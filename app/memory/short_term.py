"""
app/memory/short_term.py
─────────────────────────
Short-term memory — lưu lịch sử hội thoại gần đây.

Thiết kế:
  • In-memory dict (MVP)
  • Sliding window: giữ tối đa N lượt gần nhất
  • Abstract interface → dễ thay sang Redis / DynamoDB

Để nâng cấp lên Redis:
  1. pip install redis
  2. Kế thừa MemoryBackend, thay get/add bằng Redis list operations
  3. Thay ShortTermMemory._backend
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict

from loguru import logger

from app.core.config import settings
from app.models.schemas import MemoryMessage


# ╔══════════════════════════════════════════════════════════╗
# ║  Abstract Backend (để dễ swap)                           ║
# ╚══════════════════════════════════════════════════════════╝

class MemoryBackend(ABC):
    """Interface cho storage backend."""

    @abstractmethod
    def get_messages(self, session_id: str) -> list[MemoryMessage]:
        ...

    @abstractmethod
    def add_message(self, session_id: str, message: MemoryMessage) -> None:
        ...

    @abstractmethod
    def clear(self, session_id: str) -> None:
        ...


# ╔══════════════════════════════════════════════════════════╗
# ║  In-Memory Backend (MVP)                                 ║
# ╚══════════════════════════════════════════════════════════╝

class InMemoryBackend(MemoryBackend):
    """Lưu trong dict Python — mất khi restart server."""

    def __init__(self) -> None:
        self._store: dict[str, list[MemoryMessage]] = defaultdict(list)

    def get_messages(self, session_id: str) -> list[MemoryMessage]:
        return list(self._store[session_id])

    def add_message(self, session_id: str, message: MemoryMessage) -> None:
        self._store[session_id].append(message)

    def clear(self, session_id: str) -> None:
        self._store[session_id].clear()


# ╔══════════════════════════════════════════════════════════╗
# ║  Redis Backend                                           ║
# ╚══════════════════════════════════════════════════════════╝

import json
import redis
from pydantic import ValidationError

class RedisBackend(MemoryBackend):
    def __init__(self, url: str):
        self._r = redis.from_url(url, decode_responses=True)
        self._prefix = "agent:memory:"

    def get_messages(self, session_id: str) -> list[MemoryMessage]:
        raw = self._r.lrange(f"{self._prefix}{session_id}", 0, -1)
        res = []
        for m in raw:
            try:
                res.append(MemoryMessage(**json.loads(m)))
            except ValidationError:
                continue
        return res

    def add_message(self, session_id: str, message: MemoryMessage) -> None:
        self._r.rpush(f"{self._prefix}{session_id}", message.model_dump_json())

    def clear(self, session_id: str) -> None:
        self._r.delete(f"{self._prefix}{session_id}")


# ╔══════════════════════════════════════════════════════════╗
# ║  ShortTermMemory — wrapper chính                         ║
# ╚══════════════════════════════════════════════════════════╝

class ShortTermMemory:
    """
    Quản lý short-term memory với sliding window.
    Giữ tối đa `max_turns` cặp (user + assistant = 2 messages = 1 turn).
    """

    def __init__(
        self,
        backend: MemoryBackend | None = None,
        max_turns: int | None = None,
    ) -> None:
        if backend is None:
            if getattr(settings, "redis_url", None):
                self._backend = RedisBackend(url=settings.redis_url)
            else:
                self._backend = InMemoryBackend()
        else:
            self._backend = backend
            
        self._max_turns = max_turns or settings.short_term_max_turns

    def get_history(self, session_id: str) -> list[MemoryMessage]:
        """Trả về lịch sử hội thoại, đã cắt theo sliding window."""
        messages = self._backend.get_messages(session_id)
        # Mỗi turn = 2 messages (user + assistant)
        max_messages = self._max_turns * 2
        if len(messages) > max_messages:
            messages = messages[-max_messages:]
        return messages

    def get_history_as_dicts(self, session_id: str) -> list[dict[str, str]]:
        """Trả về dạng list[dict] để truyền thẳng vào Groq messages."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.get_history(session_id)
        ]

    def add_user_message(self, session_id: str, content: str) -> None:
        self._backend.add_message(
            session_id, MemoryMessage(role="user", content=content)
        )
        logger.debug(f"[Memory] Added user message to session={session_id}")

    def add_assistant_message(self, session_id: str, content: str) -> None:
        self._backend.add_message(
            session_id, MemoryMessage(role="assistant", content=content)
        )
        logger.debug(f"[Memory] Added assistant message to session={session_id}")

    def clear_session(self, session_id: str) -> None:
        self._backend.clear(session_id)
        logger.info(f"[Memory] Cleared session={session_id}")
