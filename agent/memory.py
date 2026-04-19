"""
agent/memory.py

Session-scoped conversation memory for the Financial Intelligence Agent.

Keeps a rolling window of past Q&A turns per session so the generator
node can reference prior context.  Designed to be swappable:
  - InMemoryStore  : default, no deps, single-process
  - RedisStore     : drop-in, set REDIS_URL env var

Usage in nodes.py:
    from agent.memory import get_store
    store = get_store()
    history = store.get_history(session_id)
    store.add_turn(session_id, question, answer)
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """A single Q&A exchange."""
    question: str
    answer: str
    route: str                      # "vectorstore" | "web_search" | "clarify"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        return cls(**d)


@dataclass
class Session:
    """All turns belonging to one user session."""
    session_id: str
    turns: deque[Turn] = field(default_factory=lambda: deque(maxlen=10))
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)
        self.last_active = time.time()

    def history_text(self, max_turns: int = 4) -> str:
        """
        Format the last `max_turns` exchanges as plain text for the
        generator prompt.  Returns empty string if no history.
        """
        recent = list(self.turns)[-max_turns:]
        if not recent:
            return ""
        lines = []
        for t in recent:
            lines.append(f"Q: {t.question}")
            # Truncate long answers to keep the prompt manageable
            answer_preview = t.answer[:300] + "…" if len(t.answer) > 300 else t.answer
            lines.append(f"A: {answer_preview}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "turns": [t.to_dict() for t in self.turns],
            "created_at": self.created_at,
            "last_active": self.last_active,
            "turn_count": len(self.turns),
        }


# ---------------------------------------------------------------------------
# Abstract store interface
# ---------------------------------------------------------------------------

class BaseMemoryStore(ABC):

    @abstractmethod
    def get_session(self, session_id: str) -> Session: ...

    @abstractmethod
    def add_turn(self, session_id: str, question: str, answer: str, route: str) -> None: ...

    @abstractmethod
    def get_history(self, session_id: str, max_turns: int = 4) -> str: ...

    @abstractmethod
    def delete_session(self, session_id: str) -> None: ...

    @abstractmethod
    def list_sessions(self) -> list[dict]: ...


# ---------------------------------------------------------------------------
# In-memory implementation (default)
# ---------------------------------------------------------------------------

class InMemoryStore(BaseMemoryStore):
    """
    Thread-safe enough for development and single-worker deployments.
    Evicts sessions inactive for > TTL_SECONDS.
    """
    TTL_SECONDS = 3600  # 1 hour

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def _evict_stale(self) -> None:
        cutoff = time.time() - self.TTL_SECONDS
        stale = [sid for sid, s in self._sessions.items() if s.last_active < cutoff]
        for sid in stale:
            del self._sessions[sid]

    def get_session(self, session_id: str) -> Session:
        self._evict_stale()
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
        return self._sessions[session_id]

    def add_turn(self, session_id: str, question: str, answer: str, route: str) -> None:
        session = self.get_session(session_id)
        session.add(Turn(question=question, answer=answer, route=route))

    def get_history(self, session_id: str, max_turns: int = 4) -> str:
        if session_id not in self._sessions:
            return ""
        return self._sessions[session_id].history_text(max_turns=max_turns)

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> list[dict]:
        self._evict_stale()
        return [s.to_dict() for s in self._sessions.values()]

    @property
    def session_count(self) -> int:
        return len(self._sessions)


# ---------------------------------------------------------------------------
# Redis implementation (optional, production)
# ---------------------------------------------------------------------------

class RedisStore(BaseMemoryStore):
    """
    Redis-backed store.  Install with: uv add redis
    Set REDIS_URL=redis://localhost:6379/0

    Each session is stored as a Redis hash of JSON turns.
    TTL is refreshed on every write.
    """
    TTL_SECONDS = 3600

    def __init__(self) -> None:
        try:
            import redis  # type: ignore
        except ImportError:
            raise RuntimeError("Install redis: uv add redis")
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self._r = redis.from_url(url, decode_responses=True)

    def _key(self, session_id: str) -> str:
        return f"fin_agent:session:{session_id}"

    def get_session(self, session_id: str) -> Session:
        import json
        key = self._key(session_id)
        raw = self._r.lrange(key, -10, -1)  # last 10 turns
        turns: deque[Turn] = deque(maxlen=10)
        for item in raw:
            turns.append(Turn.from_dict(json.loads(item)))
        return Session(session_id=session_id, turns=turns)

    def add_turn(self, session_id: str, question: str, answer: str, route: str) -> None:
        import json
        key = self._key(session_id)
        turn = Turn(question=question, answer=answer, route=route)
        self._r.rpush(key, json.dumps(turn.to_dict()))
        self._r.ltrim(key, -10, -1)     # keep last 10
        self._r.expire(key, self.TTL_SECONDS)

    def get_history(self, session_id: str, max_turns: int = 4) -> str:
        session = self.get_session(session_id)
        return session.history_text(max_turns=max_turns)

    def delete_session(self, session_id: str) -> None:
        self._r.delete(self._key(session_id))

    def list_sessions(self) -> list[dict]:
        keys = self._r.keys("fin_agent:session:*")
        return [{"session_id": k.split(":")[-1]} for k in keys]


# ---------------------------------------------------------------------------
# Factory — import this
# ---------------------------------------------------------------------------

_store: Optional[BaseMemoryStore] = None


def get_store() -> BaseMemoryStore:
    """
    Returns the singleton store.
    Uses Redis if REDIS_URL is set, otherwise InMemoryStore.
    """
    global _store
    if _store is None:
        if os.environ.get("REDIS_URL"):
            _store = RedisStore()
        else:
            _store = InMemoryStore()
    return _store