"""
app/api/routes.py
──────────────────
FastAPI routes cho AI Agent.

Endpoints:
  POST /api/chat              — gửi message, nhận response
    POST /api/chat/stream       — gửi message, nhận SSE stream response
    GET  /api/chat/suggestions  — gợi ý câu hỏi động từ courses.csv
  POST /api/chat/clear        — xóa lịch sử session
  GET  /api/health            — health check (bao gồm Groq connectivity)
  GET  /api/tools             — danh sách tools có sẵn
  GET  /api/sessions/{id}/history — xem lịch sử chat
  POST /api/rag/reingest      — nạp lại tài liệu RAG
"""

import asyncio
import json

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from loguru import logger
import pandas as pd

from app.agent.orchestrator import orchestrator
from app.core.config import settings
from app.core.groq_client import groq_client
from app.models.schemas import ChatRequest, ChatResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["Agent"])


def _sse_event(event: str, data: dict) -> str:
    """Format Server-Sent Event payload."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stream_answer_chunks(answer: str, chunk_words: int = 2) -> list[str]:
    """Split answer thành các token nhỏ để stream dần trên UI."""
    words = answer.split()
    if not words:
        return []

    chunks: list[str] = []
    for i in range(0, len(words), chunk_words):
        part = " ".join(words[i : i + chunk_words])
        if i + chunk_words < len(words):
            part += " "
        chunks.append(part)
    return chunks


def _build_suggestions_from_courses(max_items: int = 8) -> list[dict]:
    """Sinh gợi ý câu hỏi động từ dữ liệu khóa học hiện có."""
    try:
        df = pd.read_csv(settings.courses_csv_path, encoding="utf-8")
    except Exception as e:
        logger.error(f"Load courses.csv failed for suggestions: {e}")
        return []

    if df.empty:
        return []

    suggestions: list[dict] = []

    if "category" in df.columns:
        top_categories = (
            df["category"].dropna().astype(str).value_counts().head(4).index.tolist()
        )
        for category in top_categories:
            sample_courses = (
                df[df["category"].astype(str).str.lower() == category.lower()]["name"]
                .dropna()
                .astype(str)
                .head(2)
                .tolist()
            )
            sample_text = ", ".join(sample_courses) if sample_courses else "các khóa học phù hợp"
            suggestions.append(
                {
                    "title": f"Khóa {category}",
                    "description": f"Ví dụ: {sample_text}",
                    "prompt": (
                        f"Hiện trung tâm có những khóa học nào thuộc nhóm {category}? "
                        "Hãy gợi ý lộ trình phù hợp cho tôi."
                    ),
                    "source": "category",
                }
            )

    if "level" in df.columns:
        levels = df["level"].dropna().astype(str).str.strip().unique().tolist()
        for level in levels[:3]:
            suggestions.append(
                {
                    "title": f"Lộ trình {level}",
                    "description": f"Danh sách khóa học mức {level}",
                    "prompt": (
                        f"Tôi đang ở mức {level}. Hãy tư vấn các khóa học phù hợp "
                        "và thứ tự nên học."
                    ),
                    "source": "level",
                }
            )

    if "price_vnd" in df.columns:
        prices = pd.to_numeric(df["price_vnd"], errors="coerce").dropna()
        if not prices.empty:
            budget_low = int(prices.quantile(0.3))
            budget_high = int(prices.quantile(0.7))
            suggestions.extend(
                [
                    {
                        "title": "Khóa học tiết kiệm",
                        "description": f"Ngân sách khoảng dưới {budget_low:,} VND",
                        "prompt": (
                            f"Tôi có ngân sách dưới {budget_low:,} VND, "
                            "có những khóa học nào phù hợp?"
                        ),
                        "source": "budget",
                    },
                    {
                        "title": "Khóa học chuyên sâu",
                        "description": f"Nhóm đầu tư cao (từ {budget_high:,} VND)",
                        "prompt": (
                            f"Hãy gợi ý các khóa học chuyên sâu từ {budget_high:,} VND trở lên "
                            "và mục tiêu đầu ra."
                        ),
                        "source": "budget",
                    },
                ]
            )

    unique: list[dict] = []
    seen_prompts: set[str] = set()
    for item in suggestions:
        prompt = str(item.get("prompt", "")).strip()
        if not prompt or prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        unique.append(item)
        if len(unique) >= max_items:
            break

    return unique


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


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    debug: bool = Query(
        default=False,
        description="Trả ReAct trace trong metadata stream (cho debugging)",
    ),
):
    """
    SSE endpoint cho chat streaming.

    Event flow:
      - status: loading/thinking/streaming
      - metadata: intent/tools/rag/trace
      - token: delta text
      - final: final message + metadata
      - done
      - error
    """

    async def event_generator():
        try:
            # 1) Báo hiệu UI bắt đầu xử lý
            yield _sse_event("status", {"phase": "loading"})
            yield _sse_event("status", {"phase": "thinking"})

            task = asyncio.create_task(
                asyncio.to_thread(
                    orchestrator.handle_message,
                    request.session_id,
                    request.message,
                    debug,
                )
            )

            # 2) Giữ trạng thái thinking cho tới khi có kết quả
            while not task.done():
                yield _sse_event("status", {"phase": "thinking"})
                await asyncio.sleep(0.35)

            response = await task

            metadata = {
                "intent": response.intent,
                "tools_used": response.tools_used,
                "rag_used": response.rag_used,
                "brain_reasoning": response.brain_reasoning,
                "trace": response.trace.model_dump() if response.trace else None,
            }

            # 3) Báo metadata + bắt đầu stream token
            yield _sse_event("metadata", metadata)
            yield _sse_event("status", {"phase": "streaming"})

            streamed_text = ""
            for token in _stream_answer_chunks(response.answer, chunk_words=2):
                streamed_text += token
                yield _sse_event("token", {"delta": token})
                await asyncio.sleep(0.025)

            # 4) Kết thúc
            final_answer = streamed_text if streamed_text else response.answer
            yield _sse_event(
                "final",
                {
                    "session_id": response.session_id,
                    "answer": final_answer,
                    "metadata": metadata,
                },
            )
            yield _sse_event("done", {"ok": True})

        except Exception as e:
            logger.error(f"Chat stream endpoint error: {e}")
            yield _sse_event("error", {"message": f"Internal error: {str(e)}"})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/chat/suggestions")
def chat_suggestions(max_items: int = Query(default=8, ge=3, le=12)):
    """Gợi ý câu hỏi động dựa trên nội dung database khóa học."""
    suggestions = _build_suggestions_from_courses(max_items=max_items)
    return {
        "source": settings.courses_csv_path,
        "count": len(suggestions),
        "suggestions": suggestions,
    }


# ── Clear session ────────────────────────────────────────

class ClearRequest(BaseModel):
    session_id: str = "default"


@router.post("/chat/clear")
def clear_session(request: ClearRequest):
    """Xóa lịch sử hội thoại của một session."""
    orchestrator.clear_session(request.session_id)
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
    history = orchestrator.get_history_as_dicts(session_id)
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
