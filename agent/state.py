"""
agent/state.py

Defines the AgentState TypedDict that flows through every node in the
LangGraph graph.  Each field is annotated with its purpose and which
node(s) write to it.
"""

from __future__ import annotations

from typing import Annotated, Optional
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# Retrieved document model
# ---------------------------------------------------------------------------

class RetrievedDoc(TypedDict):
    text: str
    source_file: str
    section: str
    approx_year: str
    relevance_score: float
    is_graded_relevant: bool          # set by grade_documents node


# ---------------------------------------------------------------------------
# Main state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    The single source of truth that travels through the entire graph.

    Immutable rule: nodes NEVER mutate the incoming state dict.
    They return a partial dict with only the keys they want to update —
    LangGraph merges these automatically.
    """

    # ── Session / memory ───────────────────────────────────────────────────
    session_id: str                     # uuid identifying the user's session
    conversation_history: str           # formatted prior turns injected by memory

    # ── Input ──────────────────────────────────────────────────────────────
    question: str                       # original user question (never rewritten)

    # ── Routing ────────────────────────────────────────────────────────────
    route: str                          # "vectorstore" | "web_search" | "clarify"

    # ── Query management ───────────────────────────────────────────────────
    current_query: str                  # may be rewritten by rewrite_query node
    rewrite_count: int                  # how many times we have rewritten so far

    # ── Retrieval ──────────────────────────────────────────────────────────
    retrieved_docs: list[RetrievedDoc]  # from RAG tool
    web_results: list[dict]             # from Tavily web search tool

    # ── Grading ────────────────────────────────────────────────────────────
    docs_are_relevant: bool             # decision from grade_documents node
    grade_reason: str                   # LLM's explanation for its grading decision

    # ── Math tool ──────────────────────────────────────────────────────────
    calculation_requested: bool
    calculation_result: Optional[str]

    # ── Clarification ──────────────────────────────────────────────────────
    needs_clarification: bool
    clarification_question: str         # question the agent wants to ask back

    # ── Final output ───────────────────────────────────────────────────────
    answer: str
    answer_sufficient: bool             # set by check_answer_sufficiency node
    citations: list[str]                # source references included in the answer