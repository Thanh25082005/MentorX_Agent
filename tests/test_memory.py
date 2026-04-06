"""
tests/test_memory.py
─────────────────────
Unit tests cho Short-term Memory.
"""

import pytest
from app.memory.short_term import ShortTermMemory, InMemoryBackend
from app.models.schemas import MemoryMessage


class TestShortTermMemory:

    def setup_method(self):
        self.memory = ShortTermMemory(
            backend=InMemoryBackend(),
            max_turns=3,  # giữ tối đa 3 turns = 6 messages
        )
        self.session = "test-session"

    def test_empty_history(self):
        history = self.memory.get_history(self.session)
        assert history == []

    def test_add_and_get(self):
        self.memory.add_user_message(self.session, "Xin chào")
        self.memory.add_assistant_message(self.session, "Chào bạn!")

        history = self.memory.get_history(self.session)
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "Xin chào"
        assert history[1].role == "assistant"

    def test_sliding_window(self):
        """Test rằng sliding window cắt đúng."""
        # Thêm 5 turns = 10 messages, max_turns=3 → giữ 6 messages cuối
        for i in range(5):
            self.memory.add_user_message(self.session, f"User msg {i}")
            self.memory.add_assistant_message(self.session, f"Bot msg {i}")

        history = self.memory.get_history(self.session)
        assert len(history) == 6  # 3 turns * 2 messages
        # Messages cuối cùng phải là turn 2,3,4
        assert history[0].content == "User msg 2"
        assert history[-1].content == "Bot msg 4"

    def test_get_history_as_dicts(self):
        self.memory.add_user_message(self.session, "Hello")
        self.memory.add_assistant_message(self.session, "Hi there")

        dicts = self.memory.get_history_as_dicts(self.session)
        assert len(dicts) == 2
        assert dicts[0] == {"role": "user", "content": "Hello"}
        assert dicts[1] == {"role": "assistant", "content": "Hi there"}

    def test_clear_session(self):
        self.memory.add_user_message(self.session, "Test")
        self.memory.clear_session(self.session)
        assert self.memory.get_history(self.session) == []

    def test_separate_sessions(self):
        """Khác session không ảnh hưởng nhau."""
        self.memory.add_user_message("s1", "Hello from s1")
        self.memory.add_user_message("s2", "Hello from s2")

        h1 = self.memory.get_history("s1")
        h2 = self.memory.get_history("s2")
        assert len(h1) == 1
        assert len(h2) == 1
        assert h1[0].content == "Hello from s1"
        assert h2[0].content == "Hello from s2"
