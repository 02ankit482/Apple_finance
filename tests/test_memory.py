"""
tests/test_memory.py

Tests for conversation memory and session management.
No API keys needed.
"""

import time
import pytest
from agent.memory import InMemoryStore, Session, Turn


class TestInMemoryStore:

    def setup_method(self):
        self.store = InMemoryStore()

    def test_new_session_is_empty(self):
        session = self.store.get_session("abc")
        assert len(session.turns) == 0

    def test_add_turn_and_retrieve(self):
        self.store.add_turn("s1", "Q1", "A1", "vectorstore")
        history = self.store.get_history("s1")
        assert "Q1" in history
        assert "A1" in history

    def test_history_uses_max_turns(self):
        for i in range(8):
            self.store.add_turn("s1", f"Q{i}", f"A{i}", "vectorstore")
        history = self.store.get_history("s1", max_turns=2)
        # Only last 2 turns should appear
        assert "Q6" in history
        assert "Q7" in history
        assert "Q0" not in history

    def test_delete_session(self):
        self.store.add_turn("s_del", "Q", "A", "vectorstore")
        self.store.delete_session("s_del")
        history = self.store.get_history("s_del")
        assert history == ""

    def test_session_count(self):
        self.store.get_session("s_a")
        self.store.get_session("s_b")
        assert self.store.session_count >= 2

    def test_rolling_window_max_10(self):
        for i in range(15):
            self.store.add_turn("s_roll", f"Q{i}", f"A{i}", "vectorstore")
        session = self.store.get_session("s_roll")
        assert len(session.turns) == 10

    def test_empty_history_for_missing_session(self):
        result = self.store.get_history("nonexistent_session")
        assert result == ""

    def test_list_sessions(self):
        self.store.add_turn("s_list", "Q", "A", "vectorstore")
        sessions = self.store.list_sessions()
        ids = [s["session_id"] for s in sessions]
        assert "s_list" in ids

    def test_stale_eviction(self):
        store = InMemoryStore()
        store.TTL_SECONDS = 0   # everything is immediately stale
        store.add_turn("stale", "Q", "A", "vectorstore")
        # Force eviction on next get
        time.sleep(0.01)
        _ = store.get_session("new_session")
        assert "stale" not in {s["session_id"] for s in store.list_sessions()}


class TestTurn:

    def test_to_dict_round_trip(self):
        t = Turn(question="Q", answer="A", route="vectorstore")
        d = t.to_dict()
        t2 = Turn.from_dict(d)
        assert t2.question == t.question
        assert t2.answer == t.answer
        assert t2.route == t.route

    def test_history_text_truncates_long_answers(self):
        session = Session(session_id="x")
        long_answer = "x" * 1000
        session.add(Turn(question="Q", answer=long_answer, route="vectorstore"))
        history = session.history_text()
        # Should be truncated with ellipsis
        assert len(history) < 1000
        assert "…" in history