"""
main.py — CLI entry point for the Financial Intelligence Agent.

Commands:
    uv run main.py setup          # build ChromaDB index (run once)
    uv run main.py chat           # interactive REPL
    uv run main.py ask "..."      # single question
    uv run main.py server         # FastAPI server on port 8000
    uv run main.py server 8080    # custom port
    uv run main.py eval           # run evaluation harness
"""

from __future__ import annotations

import sys
import textwrap
from dotenv import load_dotenv

load_dotenv()   # must happen before any config import

from config import validate_config
from agent.graph import compiled_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BANNER = textwrap.dedent("""
╔══════════════════════════════════════════════════════╗
║      Financial Intelligence Agent  —  Apple 10-K    ║
║      Powered by LangGraph + Gemini                   ║
╚══════════════════════════════════════════════════════╝
""")


def _check_env() -> None:
    missing = validate_config()
    if missing:
        print("✗  Missing required environment variables:")
        for var in missing:
            print(f"   • {var}")
        print("\nAdd them to your .env file and re-run.")
        sys.exit(1)


def _run_agent(question: str, session_id: str = "cli_session") -> str:
    """Invoke the compiled LangGraph and return the final answer."""
    initial_state = {
        "session_id": session_id,
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
        "answer_sufficient": True,
        "citations": [],
    }
    final_state = compiled_graph.invoke(initial_state)
    return final_state.get("answer", "No answer generated.")


def _print_answer(question: str, answer: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"Q: {question}")
    print(f"{'─' * 60}")
    print(f"\n{answer}\n")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_setup() -> None:
    """Build the ChromaDB index from extracted 10-K sections."""
    _check_env()
    from ingest.embedder import build_index
    print("Building vector index from sections/ directory …\n")
    build_index()
    print("\n✓  Setup complete.  Run `uv run main.py chat` to start.")


def cmd_chat() -> None:
    """Start an interactive REPL."""
    _check_env()
    print(BANNER)
    print("Type your question and press Enter.  Type 'exit' to quit.\n")

    session_id = "cli_session"
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        try:
            answer = _run_agent(question, session_id=session_id)
            _print_answer(question, answer)
        except Exception as exc:
            print(f"\n[ERROR] {exc}\n")


def cmd_ask(question: str) -> None:
    """Answer a single question and exit."""
    _check_env()
    answer = _run_agent(question)
    _print_answer(question, answer)


def cmd_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Start the FastAPI server with uvicorn."""
    _check_env()
    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed.  Run: uv add uvicorn")
        sys.exit(1)
    print(f"Starting API server on http://{host}:{port}")
    print("Swagger docs at: http://localhost:8000/docs\n")
    uvicorn.run("server.api:app", host=host, port=port, reload=reload)


def cmd_eval(category: str | None = None, question_id: str | None = None) -> None:
    """Run the evaluation harness."""
    _check_env()
    import subprocess
    cmd = [sys.executable, "eval/harness.py"]
    if category:
        cmd += ["--category", category]
    if question_id:
        cmd += ["--id", question_id]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or args[0] == "chat":
        cmd_chat()
    elif args[0] == "setup":
        cmd_setup()
    elif args[0] == "ask":
        if len(args) < 2:
            print('Usage: uv run main.py ask "your question here"')
            sys.exit(1)
        cmd_ask(" ".join(args[1:]))
    elif args[0] == "server":
        port = int(args[1]) if len(args) > 1 and args[1].isdigit() else 8000
        cmd_server(port=port, reload="--reload" in args)
    elif args[0] == "eval":
        category = args[1] if len(args) > 1 and not args[1].startswith("-") else None
        cmd_eval(category=category)
    else:
        print(f"Unknown command: {args[0]}")
        print("Available: setup | chat | ask | server | eval")
        sys.exit(1)