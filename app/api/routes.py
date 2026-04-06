"""
app/api/routes.py
──────────────────
FastAPI routes cho AI Agent.

Endpoints:
  POST /api/chat              — gửi message, nhận response
  POST /api/chat/clear        — xóa lịch sử session
  GET  /api/health            — health check (bao gồm Groq connectivity)
  GET  /api/tools             — danh sách tools có sẵn
  GET  /api/sessions/{id}/history — xem lịch sử chat
  POST /api/rag/reingest      — nạp lại tài liệu RAG
"""

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from app.agent.orchestrator import orchestrator
from app.core.groq_client import groq_client
from app.models.schemas import ChatRequest, ChatResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["Agent"])


# ── Chat ─────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    debug: bool = Query(
        default=False,
        description="Trả ReAct trace trong response (cho debugging)",
    ),
):
    """
    Endpoint chính: nhận câu hỏi → orchestrator xử lý → trả kết quả.

    Flow nội bộ:
      1. Orchestrator đọc memory
      2. Brain phân loại intent
      3. Direct / RAG / ReAct
      4. Lưu memory
      5. Trả response (kèm trace nếu debug=true)
    """
    try:
        response = orchestrator.handle_message(
            session_id=request.session_id,
            message=request.message,
            debug=debug,
        )
        return response
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}",
        )


# ── Clear session ────────────────────────────────────────

class ClearRequest(BaseModel):
    session_id: str = "default"


@router.post("/chat/clear")
def clear_session(request: ClearRequest):
    """Xóa lịch sử hội thoại của một session."""
    orchestrator.memory.clear_session(request.session_id)
    return {"status": "ok", "message": f"Session '{request.session_id}' cleared"}


# ── Health check ─────────────────────────────────────────

@router.get("/health")
def health():
    """Health check — bao gồm kiểm tra kết nối tới Groq."""
    groq_status = groq_client.health_check()
    return {
        "status": "healthy" if groq_status["status"] == "ok" else "degraded",
        "tools": list(orchestrator.tools.keys()),
        "groq": groq_status,
    }


# ── Tools listing ────────────────────────────────────────

@router.get("/tools")
def list_tools():
    """Danh sách tools có sẵn trong hệ thống."""
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
            }
            for tool in orchestrator.tools.values()
        ]
    }


# ── Session history ──────────────────────────────────────

@router.get("/sessions/{session_id}/history")
def get_session_history(session_id: str):
    """Xem lịch sử hội thoại của một session."""
    history = orchestrator.memory.get_history_as_dicts(session_id)
    return {
        "session_id": session_id,
        "message_count": len(history),
        "messages": history,
    }


# ── RAG Reingest ─────────────────────────────────────────

@router.post("/rag/reingest")
def rag_reingest():
    """Nạp lại toàn bộ tài liệu vào vector store."""
    try:
        from app.rag.retriever import rag_retriever
        rag_retriever.initialize(force_reingest=True)
        return {"status": "ok", "message": "RAG reingested successfully"}
    except Exception as e:
        logger.error(f"RAG reingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
