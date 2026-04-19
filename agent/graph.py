"""
agent/graph.py

Full agent workflow:
─────────────────────────────────────────────────────────────
   route_question
        │
   ┌────┼─────────────┐
   ▼    ▼             ▼
clarify retrieve   web_search ◄─────────────────────────┐
        │              │                                  │
       grade    ◄──────┘                                  │
        │                                                  │
   ┌────┼─────────┐                                        │
   ▼    ▼         ▼                                        │
rewrite web_search calc_detect                             │
   │              │                                        │
   └──► retrieve  ├── run_calc ──► generate                │
                  │                   │                    │
                  └──────────────► generate                │
                                      │                    │
                               check_sufficiency           │
                               insufficient? ──────────────┘
                               sufficient?
                                   │
                                  END
─────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import (
    route_question,
    retrieve_from_rag,
    grade_documents,
    rewrite_query,
    search_web,
    detect_calculation,
    run_calculation,
    generate_answer,
    handle_clarification,
    check_answer_sufficiency,
)
from config import agent_cfg


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def _route_after_router(state: AgentState) -> str:
    route = state.get("route", "vectorstore")
    if route == "clarify":
        return "clarify"
    if route == "web_search":
        return "web_search"
    return "retrieve"


def _route_after_grading(state: AgentState) -> str:
    if state.get("docs_are_relevant", False):
        return "calc_detect"
    if state.get("rewrite_count", 0) < agent_cfg.max_retries:
        return "rewrite"
    return "web_search"


def _route_after_rewrite(state: AgentState) -> str:
    return "retrieve"


def _route_after_calc_detect(state: AgentState) -> str:
    if state.get("calculation_requested", False):
        return "run_calc"
    return "generate"


def _route_after_sufficiency(state: AgentState) -> str:
    """
    Self-correction: if the answer was insufficient and we haven't
    searched the web yet, go to web_search for a second attempt.
    Otherwise accept the answer and end.
    """
    if not state.get("answer_sufficient", True):
        return "web_search"
    return "end"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("router",       route_question)
    graph.add_node("retrieve",     retrieve_from_rag)
    graph.add_node("grade",        grade_documents)
    graph.add_node("rewrite",      rewrite_query)
    graph.add_node("web_search",   search_web)
    graph.add_node("calc_detect",  detect_calculation)
    graph.add_node("run_calc",     run_calculation)
    graph.add_node("generate",     generate_answer)
    graph.add_node("sufficiency",  check_answer_sufficiency)
    graph.add_node("clarify",      handle_clarification)

    # Entry point
    graph.set_entry_point("router")

    # Router → first action
    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "retrieve":   "retrieve",
            "web_search": "web_search",
            "clarify":    "clarify",
        },
    )

    # Retrieve → grade
    graph.add_edge("retrieve", "grade")

    # Grade → calc_detect | rewrite | web_search
    graph.add_conditional_edges(
        "grade",
        _route_after_grading,
        {
            "calc_detect": "calc_detect",
            "rewrite":     "rewrite",
            "web_search":  "web_search",
        },
    )

    # Rewrite → back to retrieve
    graph.add_conditional_edges(
        "rewrite",
        _route_after_rewrite,
        {"retrieve": "retrieve"},
    )

    # Web search → calc_detect
    graph.add_edge("web_search", "calc_detect")

    # Calc detect → run_calc | generate
    graph.add_conditional_edges(
        "calc_detect",
        _route_after_calc_detect,
        {
            "run_calc": "run_calc",
            "generate": "generate",
        },
    )

    graph.add_edge("run_calc", "generate")

    # Generate → sufficiency check
    graph.add_edge("generate", "sufficiency")

    # Sufficiency check → web_search (retry) | END
    graph.add_conditional_edges(
        "sufficiency",
        _route_after_sufficiency,
        {
            "web_search": "web_search",
            "end":        END,
        },
    )

    # Clarify always ends
    graph.add_edge("clarify", END)

    return graph.compile()


compiled_graph = build_graph()