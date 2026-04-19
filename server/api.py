"""
server/api.py

FastAPI application exposing the Financial Intelligence Agent as a REST API.

Endpoints:
  POST   /chat              — synchronous, waits for full answer
  POST   /chat/stream       — Server-Sent Events, streams tokens
  GET    /sessions/{id}     — retrieve conversation history
  DELETE /sessions/{id}     — clear a session
  GET    /sessions          — list all active sessions
  GET    /health            — liveness + readiness probe

Run with:
    uv run uvicorn server.api:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

load_dotenv()

from config import validate_config
from agent.graph import compiled_graph
from agent.memory import get_store
from ingest.embedder import FinancialVectorStore
from server.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SessionInfo,
    StreamChunk,
)


# ---------------------------------------------------------------------------
# Lifespan — runs once at startup and shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    missing = validate_config()
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Set them in your .env file."
        )
    # Warm up the vector store connection (lazy init, so just touch it)
    store = FinancialVectorStore()
    print(f"✓  Vector store ready — {store.count()} documents indexed")
    yield
    print("API shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Financial Intelligence Agent",
    description="Stateful AI agent for Apple 10-K analysis, powered by LangGraph + Azure OpenAI.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Shared: build initial agent state
# ---------------------------------------------------------------------------

def _build_initial_state(request: ChatRequest) -> dict:
    memory = get_store()
    history = memory.get_history(request.session_id, max_turns=4)

    return {
        "session_id": request.session_id,
        "conversation_history": history,
        "question": request.question,
        "route": "",
        "current_query": request.question,
        "rewrite_count": 0,
        "retrieved_docs": [],
        "web_results": [],
        "docs_are_relevant": False,
        "grade_reason": "",
        "calculation_requested": False,
        "calculation_result": None,
        "needs_clarification": False,
        "clarification_question": "",
        "answer": "",
        "citations": [],
    }


# ---------------------------------------------------------------------------
# POST /chat — synchronous
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse, summary="Ask a financial question")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Submit a question and receive a complete answer synchronously.
    Suitable for non-streaming clients.
    """
    state = _build_initial_state(request)

    # Run in a thread pool so we don't block the event loop
    final = await asyncio.to_thread(compiled_graph.invoke, state)

    docs_evaluated = len(final.get("retrieved_docs", []))

    return ChatResponse(
        session_id=request.session_id,
        question=request.question,
        answer=final.get("answer", ""),
        route=final.get("route", "vectorstore"),
        citations=final.get("citations", []),
        rewrite_count=final.get("rewrite_count", 0),
        docs_evaluated=docs_evaluated,
    )


# ---------------------------------------------------------------------------
# POST /chat/stream — Server-Sent Events
# ---------------------------------------------------------------------------

async def _stream_agent(request: ChatRequest) -> AsyncGenerator[str, None]:
    """
    Yields SSE-formatted strings.
    The LangGraph stream emits per-node state updates; we detect the
    'generate' node's output and stream the answer word by word.
    """

    def _sse(chunk: StreamChunk) -> str:
        return f"data: {chunk.model_dump_json()}\n\n"

    state = _build_initial_state(request)

    # Stream node-level events
    route_emitted = False

    try:
        for event in compiled_graph.stream(state, stream_mode="updates"):
            node_name = list(event.keys())[0]
            node_output = event[node_name]

            # Emit routing decision as metadata
            if node_name == "router" and not route_emitted:
                route = node_output.get("route", "vectorstore")
                yield _sse(StreamChunk(
                    type="metadata",
                    content="",
                    metadata={"event": "routed", "route": route}
                ))
                route_emitted = True

            # Emit retrieval metadata
            if node_name == "retrieve":
                n_docs = len(node_output.get("retrieved_docs", []))
                yield _sse(StreamChunk(
                    type="metadata",
                    content="",
                    metadata={"event": "retrieved", "count": n_docs}
                ))

            # Stream the final answer word by word
            if node_name == "generate":
                answer = node_output.get("answer", "")
                citations = node_output.get("citations", [])

                # Simulate token streaming by splitting on spaces
                words = answer.split(" ")
                for i, word in enumerate(words):
                    chunk_text = word + (" " if i < len(words) - 1 else "")
                    yield _sse(StreamChunk(type="token", content=chunk_text))
                    await asyncio.sleep(0.01)   # small delay for realistic feel

                # Emit citations as final metadata
                yield _sse(StreamChunk(
                    type="metadata",
                    content="",
                    metadata={"event": "citations", "citations": citations}
                ))

        yield _sse(StreamChunk(type="done", content=""))

    except Exception as exc:
        yield _sse(StreamChunk(type="error", content=str(exc)))


@app.post("/chat/stream", summary="Stream a financial answer via SSE")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Submit a question and receive the answer as Server-Sent Events.
    Each event is a JSON `StreamChunk` with type: token | metadata | done | error.

    Example client (JavaScript):
    ```js
    const es = new EventSource('/chat/stream');
    es.onmessage = e => {
        const chunk = JSON.parse(e.data);
        if (chunk.type === 'token') process.stdout.write(chunk.content);
        if (chunk.type === 'done') es.close();
    };
    ```
    """
    return StreamingResponse(
        _stream_agent(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

@app.get("/sessions", response_model=list[dict], summary="List active sessions")
async def list_sessions():
    return get_store().list_sessions()


@app.get("/sessions/{session_id}", response_model=SessionInfo, summary="Get session history")
async def get_session(session_id: str):
    session = get_store().get_session(session_id)
    if not session.turns:
        raise HTTPException(status_code=404, detail="Session not found or empty")
    return SessionInfo(
        session_id=session.session_id,
        turn_count=len(session.turns),
        created_at=session.created_at,
        last_active=session.last_active,
        turns=[t.to_dict() for t in session.turns],
    )


@app.delete("/sessions/{session_id}", summary="Clear a session")
async def delete_session(session_id: str):
    get_store().delete_session(session_id)
    return {"deleted": session_id}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, summary="Health probe")
async def health():
    vector_store = FinancialVectorStore()
    memory = get_store()
    return HealthResponse(
        status="ok",
        vector_store_docs=vector_store.count(),
        session_count=getattr(memory, "session_count", -1),
    )