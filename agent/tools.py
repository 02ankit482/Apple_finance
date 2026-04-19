"""
agent/tools.py

The three tools available to the agent:
  1. rag_tool        — semantic search over ChromaDB
  2. web_search_tool — Tavily real-time web search
  3. calculator_tool — safe Python expression evaluator for YoY math
"""

from __future__ import annotations

import ast
import math
import operator
import re
from functools import lru_cache
from typing import Any

from tavily import TavilyClient

from config import tavily_cfg, agent_cfg
from ingest.embedder import FinancialVectorStore


# ---------------------------------------------------------------------------
# 1.  RAG Tool
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_vector_store() -> FinancialVectorStore:
    """Lazy singleton — initialised on first call."""
    return FinancialVectorStore()


def rag_tool(
    query: str,
    top_k: int | None = None,
    section_filter: str | None = None,
    year_filter: str | None = None,
) -> list[dict]:
    """
    Perform semantic retrieval from the Apple 10-K vector store.

    Args:
        query:          Natural-language question or keywords.
        top_k:          Number of chunks to retrieve (default from config).
        section_filter: Restrict to a specific section e.g. "item_7".
        year_filter:    Restrict to a specific fiscal year e.g. "2023".

    Returns:
        List of dicts with keys: text, source_file, section,
        approx_year, relevance_score.
    """
    store = _get_vector_store()
    if not store.collection_exists():
        raise RuntimeError(
            "Vector store is empty.  Run `python setup.py` to build the index first."
        )

    results = store.query(
        query_text=query,
        top_k=top_k or agent_cfg.top_k_retrieval,
        section_filter=section_filter,
        year_filter=year_filter,
    )
    return results


# ---------------------------------------------------------------------------
# 2.  Web Search Tool
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_tavily_client() -> TavilyClient:
    return TavilyClient(api_key=tavily_cfg.api_key)


def web_search_tool(query: str, max_results: int | None = None) -> list[dict]:
    """
    Search the live web via Tavily.

    Args:
        query:       Search query string.
        max_results: Cap on results (default from config).

    Returns:
        List of dicts with keys: title, url, content, score.
    """
    client = _get_tavily_client()
    k = max_results or agent_cfg.max_web_results

    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=k,
        include_answer=True,
    )

    results: list[dict] = []
    for r in response.get("results", []):
        results.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": round(r.get("score", 0.0), 4),
            }
        )

    # Prepend Tavily's own synthesised answer if available
    tavily_answer = response.get("answer", "")
    if tavily_answer:
        results.insert(
            0,
            {
                "title": "Tavily Answer",
                "url": "tavily-synthesis",
                "content": tavily_answer,
                "score": 1.0,
            },
        )

    return results


# ---------------------------------------------------------------------------
# 3.  Calculator Tool
# ---------------------------------------------------------------------------

# Safe operator whitelist
_SAFE_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

_SAFE_FUNCS = {
    "round": round,
    "abs": abs,
    "sqrt": math.sqrt,
    "log": math.log,
    "min": min,
    "max": max,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an AST node using only whitelisted operations."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported operator: {op_type}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _SAFE_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type}")
        return _SAFE_OPS[op_type](_safe_eval(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed.")
        func_name = node.func.id
        if func_name not in _SAFE_FUNCS:
            raise ValueError(f"Function '{func_name}' is not allowed.")
        args = [_safe_eval(a) for a in node.args]
        return _SAFE_FUNCS[func_name](*args)

    raise ValueError(f"Unsupported AST node: {type(node)}")


def calculator_tool(expression: str) -> dict:
    """
    Safely evaluate a mathematical expression.

    Supports: +, -, *, /, ** and functions: round, abs, sqrt, log, min, max.
    No imports, no arbitrary code execution.

    Returns:
        {"result": float, "expression": str, "error": str|None}
    """
    expression = expression.strip()
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return {
            "result": round(result, 4),
            "expression": expression,
            "error": None,
        }
    except Exception as exc:
        return {
            "result": None,
            "expression": expression,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Convenience: YoY growth calculator
# ---------------------------------------------------------------------------

def yoy_growth(value_current: float, value_previous: float) -> dict:
    """
    Calculate Year-over-Year percentage growth.

    Returns:
        {"growth_pct": float, "absolute_change": float}
    """
    if value_previous == 0:
        return {"growth_pct": None, "absolute_change": value_current, "error": "Division by zero"}
    growth = ((value_current - value_previous) / abs(value_previous)) * 100
    return {
        "growth_pct": round(growth, 2),
        "absolute_change": round(value_current - value_previous, 4),
        "error": None,
    }