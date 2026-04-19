"""
eval/harness.py

Evaluation harness for the Financial Intelligence Agent.

Metrics measured per question:
  - routing_correct    : did the agent use the right data source?
  - keyword_hit_rate   : fraction of expected keywords found in the answer
  - has_citation       : does the answer include at least one citation?
  - latency_s          : wall-clock seconds for the full agent run
  - calc_triggered     : was the calculator tool invoked when expected?

Aggregate report:
  - routing accuracy
  - avg keyword hit rate
  - citation rate
  - avg / p95 latency
  - pass rate (keyword_hit_rate >= 0.5 AND routing_correct)

Usage:
    uv run eval/harness.py                      # run all questions
    uv run eval/harness.py --category financials # filter by category
    uv run eval/harness.py --id fin_001         # run single question
    uv run eval/harness.py --dry-run            # validate JSON only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.graph import compiled_graph


# ---------------------------------------------------------------------------
# Load questions
# ---------------------------------------------------------------------------

QUESTIONS_FILE = Path(__file__).parent / "questions.json"


def load_questions(
    category: Optional[str] = None,
    question_id: Optional[str] = None,
) -> list[dict]:
    with open(QUESTIONS_FILE) as f:
        questions = json.load(f)

    if question_id:
        questions = [q for q in questions if q["id"] == question_id]
    elif category:
        questions = [q for q in questions if q.get("category") == category]

    return questions


# ---------------------------------------------------------------------------
# Single question runner
# ---------------------------------------------------------------------------

def _build_initial_state(question: str) -> dict:
    return {
        "session_id": "eval_session",
        "conversation_history": "",
        "question": question,
        "route": "",
        "current_query": question,
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


def run_question(question_spec: dict) -> dict:
    """
    Run a single question through the agent and compute metrics.
    Returns a result dict with all metric values.
    """
    q = question_spec["question"]
    expected_route = question_spec.get("expected_route", "vectorstore")
    required_keywords = [kw.lower() for kw in question_spec.get("required_keywords", [])]
    expects_calc = question_spec.get("expected_calculation", False)

    start = time.perf_counter()
    try:
        final = compiled_graph.invoke(_build_initial_state(q))
        latency = time.perf_counter() - start
        error = None
    except Exception as exc:
        latency = time.perf_counter() - start
        final = {}
        error = str(exc)

    actual_route = final.get("route", "")
    answer = final.get("answer", "").lower()
    citations = final.get("citations", [])
    calc_triggered = final.get("calculation_requested", False)

    routing_correct = actual_route == expected_route

    if required_keywords:
        hits = sum(1 for kw in required_keywords if kw in answer)
        keyword_hit_rate = hits / len(required_keywords)
    else:
        keyword_hit_rate = 1.0   # no keywords expected → pass

    has_citation = len(citations) > 0
    calc_correct = (calc_triggered == expects_calc) if expects_calc else True
    passed = routing_correct and keyword_hit_rate >= 0.5

    return {
        "id": question_spec["id"],
        "category": question_spec.get("category", ""),
        "question": q,
        "expected_route": expected_route,
        "actual_route": actual_route,
        "routing_correct": routing_correct,
        "keyword_hit_rate": round(keyword_hit_rate, 2),
        "has_citation": has_citation,
        "calc_correct": calc_correct,
        "latency_s": round(latency, 2),
        "passed": passed,
        "error": error,
        "answer_preview": final.get("answer", "")[:200],
    }


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

def _pct(n: int, total: int) -> str:
    return f"{n}/{total} ({100*n//total}%)"


def print_report(results: list[dict]) -> None:
    total = len(results)
    if total == 0:
        print("No results.")
        return

    passed = sum(1 for r in results if r["passed"])
    routing_ok = sum(1 for r in results if r["routing_correct"])
    with_citation = sum(1 for r in results if r["has_citation"])
    errors = sum(1 for r in results if r["error"])
    avg_kw = sum(r["keyword_hit_rate"] for r in results) / total
    latencies = [r["latency_s"] for r in results]
    avg_lat = sum(latencies) / total
    p95_lat = sorted(latencies)[int(len(latencies) * 0.95)]

    width = 62
    print("\n" + "=" * width)
    print("  EVALUATION REPORT — Financial Intelligence Agent")
    print("=" * width)
    print(f"  Questions run     : {total}")
    print(f"  Pass rate         : {_pct(passed, total)}")
    print(f"  Routing accuracy  : {_pct(routing_ok, total)}")
    print(f"  Citation rate     : {_pct(with_citation, total)}")
    print(f"  Avg keyword hits  : {avg_kw:.0%}")
    print(f"  Avg latency       : {avg_lat:.1f}s")
    print(f"  P95 latency       : {p95_lat:.1f}s")
    print(f"  Errors            : {errors}")
    print("=" * width)

    # Per-category summary
    categories: dict[str, list[dict]] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r)

    print("\n  By category:")
    for cat, cat_results in sorted(categories.items()):
        cat_pass = sum(1 for r in cat_results if r["passed"])
        print(f"    {cat:25s}  {_pct(cat_pass, len(cat_results))}")

    # Failures
    failures = [r for r in results if not r["passed"]]
    if failures:
        print("\n  Failures:")
        for r in failures:
            tag = f"[{r['id']}]"
            route_tag = (
                f"route: {r['actual_route']} (expected {r['expected_route']})"
                if not r["routing_correct"]
                else f"route: ✓"
            )
            kw_tag = f"kw: {r['keyword_hit_rate']:.0%}"
            err_tag = f"ERROR: {r['error']}" if r["error"] else ""
            print(f"    {tag:12s}  {route_tag}  {kw_tag}  {err_tag}")

    print()


# ---------------------------------------------------------------------------
# Save results to JSON
# ---------------------------------------------------------------------------

def save_results(results: list[dict], output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Financial Intelligence Agent")
    parser.add_argument("--category", help="Filter to a specific question category")
    parser.add_argument("--id", help="Run a single question by ID")
    parser.add_argument("--dry-run", action="store_true", help="Validate questions.json only")
    parser.add_argument("--output", default="eval/results.json", help="Path to save results JSON")
    args = parser.parse_args()

    questions = load_questions(category=args.category, question_id=args.id)
    if not questions:
        print("No questions matched the filter.")
        sys.exit(1)

    print(f"Loaded {len(questions)} question(s)")

    if args.dry_run:
        print("Dry-run: JSON validated successfully.")
        for q in questions:
            print(f"  [{q['id']}] {q['question'][:80]}")
        return

    results: list[dict] = []
    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q['id']}: {q['question'][:70]}…")
        result = run_question(q)
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"  {status}  route={result['actual_route']}  "
              f"kw={result['keyword_hit_rate']:.0%}  "
              f"lat={result['latency_s']}s")
        results.append(result)

    print_report(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()