"""
agent/nodes.py — LangGraph node functions, using google.genai SDK.

Rate-limit strategy:
  - _chat() retries automatically with exponential backoff on 429 errors
  - grade_documents() batches ALL chunks into ONE LLM call (was 6 calls)
  - This keeps total calls per question to ~4 regardless of chunk count
"""

from __future__ import annotations

import json
import re
import time

from google import genai
from google.genai import types
from google.genai.errors import ClientError

from config import gemini_cfg, agent_cfg
from agent.state import AgentState, RetrievedDoc
from agent.prompts import (
    ROUTER_SYSTEM, ROUTER_USER,
    GRADER_SYSTEM, GRADER_USER,
    REWRITER_SYSTEM, REWRITER_USER,
    GENERATOR_SYSTEM, GENERATOR_USER,
    CALCULATOR_DETECT_SYSTEM, CALCULATOR_DETECT_USER,
    CLARIFICATION_TEMPLATE,
)
from agent.tools import rag_tool, web_search_tool, calculator_tool


# ---------------------------------------------------------------------------
# Shared Gemini client
# ---------------------------------------------------------------------------

_client = genai.Client(api_key=gemini_cfg.api_key)


def _chat(system: str, user: str, temperature: float = 0.0, max_retries: int = 5) -> str:
    """
    Single-turn generation with automatic retry on 429 rate-limit errors.
    Waits the delay suggested by the API before retrying.
    """
    prompt = f"{system}\n\n---\n\n{user}"
    for attempt in range(max_retries):
        try:
            response = _client.models.generate_content(
                model=gemini_cfg.chat_model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=temperature),
            )
            return response.text.strip()
        except ClientError as e:
            if e.status_code == 429 and attempt < max_retries - 1:
                # Extract suggested retry delay from error, default to 15s
                wait = _parse_retry_delay(str(e), default=15)
                print(f"[rate limit] waiting {wait}s before retry (attempt {attempt + 1}/{max_retries}) …")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")


def _parse_retry_delay(error_str: str, default: int = 15) -> int:
    """Extract the retryDelay seconds from the error message string."""
    match = re.search(r"retryDelay.*?(\d+)s", error_str)
    if match:
        return int(match.group(1)) + 2   # add 2s buffer
    return default


def _parse_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    cleaned = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise


# ---------------------------------------------------------------------------
# NODE 1 — Router
# ---------------------------------------------------------------------------

def route_question(state: AgentState) -> dict:
    question = state["question"]
    raw = _chat(ROUTER_SYSTEM, ROUTER_USER.format(question=question))

    try:
        parsed = _parse_json(raw)
        route = parsed.get("route", "vectorstore")
        clarify_q = parsed.get("clarification_question", "")
    except Exception:
        route = "vectorstore"
        clarify_q = ""

    print(f"[router] → {route}")
    return {
        "route": route,
        "current_query": question,
        "rewrite_count": 0,
        "needs_clarification": route == "clarify",
        "clarification_question": clarify_q,
        "retrieved_docs": [],
        "web_results": [],
        "calculation_requested": False,
        "calculation_result": None,
        "answer": "",
        "citations": [],
    }


# ---------------------------------------------------------------------------
# NODE 2 — RAG Retrieval
# ---------------------------------------------------------------------------

def retrieve_from_rag(state: AgentState) -> dict:
    query = state["current_query"]
    print(f"[rag_retrieve] query='{query[:80]}…'")
    raw_docs = rag_tool(query)
    docs: list[RetrievedDoc] = [
        RetrievedDoc(
            text=d["text"],
            source_file=d["source_file"],
            section=d["section"],
            approx_year=d["approx_year"],
            relevance_score=d["relevance_score"],
            is_graded_relevant=False,
        )
        for d in raw_docs
    ]
    print(f"[rag_retrieve] fetched {len(docs)} chunks")
    return {"retrieved_docs": docs}


# ---------------------------------------------------------------------------
# NODE 3 — Document Grader  (ONE LLM call for all chunks)
# ---------------------------------------------------------------------------

_BATCH_GRADER_SYSTEM = """\
You are a relevance grader for a financial RAG system.

Given a user question and a list of retrieved text chunks (numbered), decide
for EACH chunk whether it contains information that would help answer the question.

Be generous: partial relevance counts as relevant.

Respond with ONLY a JSON array of booleans, one per chunk, in order.
Example for 3 chunks: [true, false, true]
No explanation, no markdown, just the JSON array.
"""

_BATCH_GRADER_USER = """\
Question: {question}

Chunks:
{chunks}
"""


def grade_documents(state: AgentState) -> dict:
    """
    Grade all chunks in ONE LLM call instead of one call per chunk.
    Reduces API calls from N to 1, staying well within rate limits.
    """
    question = state["question"]
    docs = state.get("retrieved_docs", [])

    if not docs:
        return {"docs_are_relevant": False, "grade_reason": "No documents retrieved."}

    # Build numbered chunk list for the single prompt
    chunk_lines = "\n\n".join(
        f"[{i+1}] {doc['text'][:400]}"
        for i, doc in enumerate(docs)
    )

    raw = _chat(
        _BATCH_GRADER_SYSTEM,
        _BATCH_GRADER_USER.format(question=question, chunks=chunk_lines),
    )

    # Parse the boolean array
    try:
        cleaned = re.sub(r"^```[a-z]*\n?", "", raw.strip())
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
        relevance_list: list[bool] = json.loads(cleaned)
        if len(relevance_list) != len(docs):
            # Fallback: mark all as relevant if parse gives wrong length
            relevance_list = [True] * len(docs)
    except Exception:
        relevance_list = [True] * len(docs)

    graded: list[RetrievedDoc] = []
    for doc, is_relevant in zip(docs, relevance_list):
        updated = dict(doc)
        updated["is_graded_relevant"] = bool(is_relevant)
        graded.append(updated)  # type: ignore[arg-type]

    relevant_count = sum(1 for d in graded if d["is_graded_relevant"])
    overall_relevant = relevant_count >= 1
    print(f"[grader] {relevant_count}/{len(graded)} chunks relevant → {overall_relevant}")

    return {
        "retrieved_docs": graded,
        "docs_are_relevant": overall_relevant,
        "grade_reason": f"{relevant_count} of {len(graded)} chunks are relevant.",
    }


# ---------------------------------------------------------------------------
# NODE 4 — Query Rewriter
# ---------------------------------------------------------------------------

def rewrite_query(state: AgentState) -> dict:
    count = state.get("rewrite_count", 0) + 1
    new_query = _chat(
        REWRITER_SYSTEM,
        REWRITER_USER.format(
            question=state["question"],
            failed_query=state["current_query"],
            attempt=count,
        ),
    )
    print(f"[rewriter] attempt {count}: '{new_query[:80]}'")
    return {"current_query": new_query.strip(), "rewrite_count": count}


# ---------------------------------------------------------------------------
# NODE 5 — Web Search
# ---------------------------------------------------------------------------

def search_web(state: AgentState) -> dict:
    query = state["current_query"]
    print(f"[web_search] query='{query[:80]}…'")
    results = web_search_tool(query)
    print(f"[web_search] got {len(results)} results")
    return {"web_results": results}


# ---------------------------------------------------------------------------
# NODE 6 — Calculation Detection
# ---------------------------------------------------------------------------

def detect_calculation(state: AgentState) -> dict:
    raw = _chat(
        CALCULATOR_DETECT_SYSTEM,
        CALCULATOR_DETECT_USER.format(question=state["question"]),
    )
    try:
        parsed = _parse_json(raw)
        needed = bool(parsed.get("calculation_needed", False))
        expression = parsed.get("expression", "")
    except Exception:
        needed = False
        expression = ""

    return {
        "calculation_requested": needed,
        "_pending_calc_expression": expression,
    }


# ---------------------------------------------------------------------------
# NODE 7 — Run Calculation
# ---------------------------------------------------------------------------

def run_calculation(state: AgentState) -> dict:
    expression = state.get("_pending_calc_expression", "")
    if not expression:
        return {"calculation_result": None}

    result = calculator_tool(expression)
    calc_str = (
        f"Calculation failed: {result['error']}"
        if result["error"]
        else f"{expression} = {result['result']}"
    )
    print(f"[calculator] {calc_str}")
    return {"calculation_result": calc_str}


# ---------------------------------------------------------------------------
# NODE 8 — Answer Generator
# ---------------------------------------------------------------------------

def generate_answer(state: AgentState) -> dict:
    from agent.memory import get_store

    rag_parts: list[str] = []
    citations: list[str] = []
    for doc in state.get("retrieved_docs", []):
        if doc.get("is_graded_relevant", False):
            source = f"[{doc['section'].upper()} | {doc['approx_year']} | {doc['source_file']}]"
            rag_parts.append(f"{source}\n{doc['text']}")
            if source not in citations:
                citations.append(source)

    rag_context = "\n\n---\n\n".join(rag_parts) if rag_parts else "No 10-K context available."

    web_parts: list[str] = []
    for wr in state.get("web_results", []):
        if wr.get("content"):
            web_parts.append(f"[{wr['title']}] ({wr['url']})\n{wr['content'][:500]}")
            citations.append(wr["url"])

    web_context = "\n\n---\n\n".join(web_parts) if web_parts else "No web context available."
    calc_result = state.get("calculation_result") or "N/A"
    history = state.get("conversation_history", "") or "None"

    answer = _chat(
        GENERATOR_SYSTEM,
        GENERATOR_USER.format(
            question=state["question"],
            history=history,
            rag_context=rag_context,
            web_context=web_context,
            calc_result=calc_result,
        ),
        temperature=0.2,
    )

    get_store().add_turn(
        state.get("session_id", "default"),
        state["question"],
        answer,
        state.get("route", "vectorstore"),
    )

    print(f"[generator] answer length={len(answer)}")
    return {"answer": answer, "citations": citations}


# ---------------------------------------------------------------------------
# NODE 9 — Clarification Handler
# ---------------------------------------------------------------------------

def handle_clarification(state: AgentState) -> dict:
    q = state.get("clarification_question", "Could you please provide more details?")
    return {"answer": CLARIFICATION_TEMPLATE.format(clarification_question=q)}


# ---------------------------------------------------------------------------
# NODE 10 — Answer Sufficiency Check  (self-correction)
# ---------------------------------------------------------------------------

# Phrases that indicate the generator could not answer from context
_INSUFFICIENT_PHRASES = [
    "cannot provide",
    "cannot answer",
    "not available",
    "no information",
    "does not contain",
    "not found in",
    "i don't have",
    "i do not have",
    "not included",
    "no context",
    "insufficient",
    "not mentioned",
    "not provided",
    "outside the scope",
    "beyond the scope",
]


def check_answer_sufficiency(state: AgentState) -> dict:
    """
    Checks whether the generated answer is actually useful.
    If it contains 'cannot provide' / 'not available' type phrases
    AND we haven't already done a web search, mark it as insufficient
    so the graph can route to web search for a second attempt.
    """
    answer = state.get("answer", "").lower()
    already_searched_web = bool(state.get("web_results"))

    if already_searched_web:
        # Already tried web search — accept whatever we have
        print("[sufficiency] already used web search, accepting answer")
        return {"answer_sufficient": True}

    insufficient = any(phrase in answer for phrase in _INSUFFICIENT_PHRASES)

    if insufficient:
        print("[sufficiency] answer insufficient — routing to web search")
    else:
        print("[sufficiency] answer looks good")

    return {"answer_sufficient": not insufficient}