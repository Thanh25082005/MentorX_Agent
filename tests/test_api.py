"""
tests/test_api.py
──────────────────
Integration tests cho API endpoints.
Mock Groq client để chạy test mà không cần API key.

Chạy: pytest tests/test_api.py -v
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Mock Groq trước khi import app ───────────────────────

def _mock_groq_chat(messages, temperature=0.3, max_tokens=2048):
    """Fake chat response."""
    return "Xin chào! Tôi là trợ lý AI Academy."


def _mock_groq_chat_json(messages, temperature=0.1, max_tokens=1024):
    """Fake JSON response — trả direct_answer intent."""
    return {
        "intent": "direct_answer",
        "reasoning": "Đây là câu chào hỏi đơn giản",
        "direct_response": "Xin chào! Tôi là trợ lý AI Academy. Tôi có thể giúp gì cho bạn?",
        "rag_query": None,
        "tool_hint": None,
    }


def _mock_groq_health_check():
    """Fake health check."""
    return {"status": "ok", "model": "test-model", "latency_ms": 50}


@pytest.fixture
def client():
    """Create test client with mocked Groq."""
    with patch("app.core.groq_client.groq_client") as mock_groq:
        mock_groq.chat = MagicMock(side_effect=_mock_groq_chat)
        mock_groq.chat_json = MagicMock(side_effect=_mock_groq_chat_json)
        mock_groq.health_check = MagicMock(side_effect=_mock_groq_health_check)

        # Patch ở orchestrator vì nó import singleton
        with patch("app.agent.orchestrator.groq_client", mock_groq), \
             patch("app.agent.react_loop.groq_client", mock_groq), \
             patch("app.api.routes.groq_client", mock_groq):

            from app.main import app
            yield TestClient(app)


# ╔══════════════════════════════════════════════════════════╗
# ║  Root & Health                                           ║
# ╚══════════════════════════════════════════════════════════╝

class TestRootAndHealth:

    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "AI Academy Agent"
        assert "docs" in data

    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")
        assert "tools" in data
        assert isinstance(data["tools"], list)

    def test_tools_listing(self, client):
        resp = client.get("/api/tools")
        assert resp.status_code == 200
        data = resp.json()
        tools = data["tools"]
        assert len(tools) >= 3
        tool_names = [t["name"] for t in tools]
        assert "calculator" in tool_names
        assert "course_search" in tool_names
        assert "web_search" in tool_names


# ╔══════════════════════════════════════════════════════════╗
# ║  Chat Endpoint                                           ║
# ╚══════════════════════════════════════════════════════════╝

class TestChatEndpoint:

    def test_chat_basic(self, client):
        resp = client.post(
            "/api/chat",
            json={"message": "Xin chào"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert data["session_id"] == "default"
        assert data["intent"] == "direct_answer"

    def test_chat_with_session(self, client):
        resp = client.post(
            "/api/chat",
            json={"message": "Hello", "session_id": "test-session-123"},
        )
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "test-session-123"

    def test_chat_empty_message_rejected(self, client):
        resp = client.post(
            "/api/chat",
            json={"message": ""},
        )
        assert resp.status_code == 422  # validation error

    def test_chat_debug_mode(self, client):
        """Debug mode trả trace nếu intent = use_tools."""
        resp = client.post(
            "/api/chat?debug=true",
            json={"message": "Xin chào"},
        )
        assert resp.status_code == 200
        # direct_answer không tạo trace, nhưng field phải tồn tại
        data = resp.json()
        assert "trace" in data


# ╔══════════════════════════════════════════════════════════╗
# ║  Session Management                                      ║
# ╚══════════════════════════════════════════════════════════╝

class TestSessionManagement:

    def test_clear_session(self, client):
        # Chat trước
        client.post("/api/chat", json={"message": "Hello", "session_id": "s1"})
        # Kiểm tra history có
        resp = client.get("/api/sessions/s1/history")
        assert resp.json()["message_count"] > 0

        # Clear
        resp = client.post("/api/chat/clear", json={"session_id": "s1"})
        assert resp.status_code == 200

        # Kiểm tra history trống
        resp = client.get("/api/sessions/s1/history")
        assert resp.json()["message_count"] == 0

    def test_get_empty_history(self, client):
        resp = client.get("/api/sessions/nonexistent/history")
        assert resp.status_code == 200
        assert resp.json()["message_count"] == 0
