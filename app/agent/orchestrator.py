"""
app/agent/orchestrator.py
──────────────────────────
Orchestrator — bộ điều phối trung tâm của Agent.

Luồng xử lý:
  1. Nhận query từ user
  2. Đọc short-term memory (lịch sử chat)
  3. Brain (Groq) phân loại intent → structured JSON
     • direct_answer  → trả lời trực tiếp
     • use_rag        → retrieve context rồi trả lời
     • use_tools      → vào ReAct loop
  4. Tổng hợp câu trả lời cuối
  5. Ghi vào short-term memory
  6. Trả kết quả qua API

Brain phối hợp:
  • Với Memory: đọc history trước mỗi lượt
  • Với RAG: khi intent=use_rag → retrieve + augment prompt
  • Với ReAct: khi intent=use_tools → chạy react loop
"""

from __future__ import annotations

import time

from loguru import logger

from app.core.config import settings
from app.core.groq_client import groq_client
from app.memory.short_term import ShortTermMemory
from app.models.schemas import (
    BrainDecision,
    ChatResponse,
    IntentType,
    ReActTrace,
)
from app.rag.retriever import rag_retriever
from app.agent.react_loop import run_react_loop
from app.tools.base import BaseTool
from app.tools.calculator import CalculatorTool
from app.tools.course_search import CourseSearchTool
from app.tools.web_search import WebSearchTool


# ── Prompts ───────────────────────────────────────────────

INTENT_SYSTEM_PROMPT = """Bạn là trợ lý tư vấn học thuật của AI Academy. Nhiệm vụ: phân tích câu hỏi và quyết định cách xử lý.

Bạn PHẢI trả về JSON với format sau:

{{
  "intent": "direct_answer" | "use_rag" | "use_tools",
  "reasoning": "giải thích ngắn gọn tại sao chọn intent này",
  "direct_response": "câu trả lời nếu intent=direct_answer, null nếu khác",
  "rag_query":  "câu truy vấn tối ưu để tìm tài liệu nếu intent=use_rag, null nếu khác",
  "tool_hint": "gợi ý tool cần dùng nếu intent=use_tools, null nếu khác"
}}

HƯỚNG DẪN:
- "direct_answer": cho chào hỏi, hỏi thăm, kiến thức phổ thông, hoặc câu hỏi đã có đủ context trong lịch sử chat
- "use_rag": cho câu hỏi về chính sách, quy định, chương trình đào tạo, lộ trình học, chứng chỉ, giảng viên (thông tin chi tiết từ tài liệu nội bộ)
- "use_tools": cho câu hỏi cần tra cứu khóa học cụ thể (tên, giá, lịch), tính toán học phí, hoặc tìm kiếm internet

Trả lời bằng tiếng Việt. Chỉ trả JSON, không thêm text ngoài JSON."""

RAG_ANSWER_PROMPT = """Bạn là trợ lý tư vấn học thuật AI Academy.

Dựa trên context từ tài liệu nội bộ dưới đây, hãy trả lời câu hỏi của người dùng.

CONTEXT:
{context}

QUY TẮC:
- Trả lời dựa trên context, không bịa thông tin
- Nếu context không đủ, nói rõ và gợi ý liên hệ hỗ trợ
- Trả lời thân thiện, rõ ràng, có cấu trúc
- Dùng bullet points khi cần liệt kê
"""


class AgentOrchestrator:
    """Bộ điều phối trung tâm."""

    def __init__(self) -> None:
        self.memory = ShortTermMemory()

        # Khởi tạo tools
        self.tools: dict[str, BaseTool] = {
            "course_search": CourseSearchTool(),
            "calculator": CalculatorTool(),
            "web_search": WebSearchTool(),
        }

        logger.info(
            f"Orchestrator initialized with tools: {list(self.tools.keys())}"
        )

    def initialize_rag(self) -> None:
        """Khởi tạo RAG (gọi 1 lần khi startup)."""
        try:
            rag_retriever.initialize()
            logger.info("RAG initialized successfully")
        except Exception as e:
            logger.error(f"RAG initialization failed: {e}")

    # ── Entry point ──────────────────────────────────────

    def handle_message(
        self, session_id: str, message: str, debug: bool = False
    ) -> ChatResponse:
        """
        Xử lý 1 lượt hội thoại (SYNC — Groq SDK là synchronous).

        Flow:
          1. Lấy history
          2. Phân loại intent (có timing)
          3. Xử lý theo intent
          4. Lưu memory
          5. Trả response (kèm trace nếu debug=True)
        """
        total_start = time.time()
        logger.info(f"[Orchestrator] session={session_id}, query={message[:80]}")

        # 1. Lấy chat history
        history = self.memory.get_history_as_dicts(session_id)

        # 2. Phân loại intent qua Brain
        t0 = time.time()
        decision = self._classify_intent(message, history)
        t_intent = time.time() - t0
        logger.info(
            f"[Orchestrator] intent={decision.intent} "
            f"({t_intent:.2f}s), reasoning={decision.reasoning[:60]}"
        )

        # 3. Xử lý theo intent
        answer = ""
        tools_used: list[str] = []
        rag_used = False
        trace: ReActTrace | None = None

        if decision.intent == IntentType.DIRECT_ANSWER:
            answer = decision.direct_response or self._direct_chat(
                message, history
            )

        elif decision.intent == IntentType.USE_RAG:
            t0 = time.time()
            answer, rag_used = self._handle_rag(
                query=decision.rag_query or message,
                original_query=message,
                history=history,
            )
            logger.info(f"[Orchestrator] RAG completed in {time.time() - t0:.2f}s")

        elif decision.intent == IntentType.USE_TOOLS:
            t0 = time.time()
            answer, tools_used, trace = self._handle_tools(message, history)
            logger.info(
                f"[Orchestrator] ReAct completed in {time.time() - t0:.2f}s, "
                f"tools_used={tools_used}"
            )

        else:
            answer = self._direct_chat(message, history)

        # 4. Lưu vào memory
        self.memory.add_user_message(session_id, message)
        self.memory.add_assistant_message(session_id, answer)

        total_time = time.time() - total_start
        logger.info(f"[Orchestrator] Total processing time: {total_time:.2f}s")

        # 5. Trả response
        return ChatResponse(
            session_id=session_id,
            answer=answer,
            intent=decision.intent.value,
            tools_used=tools_used,
            rag_used=rag_used,
            trace=trace if (debug or settings.debug_mode) else None,
        )

    # ── Intent Classification (Brain) ────────────────────

    def _classify_intent(
        self, query: str, history: list[dict[str, str]]
    ) -> BrainDecision:
        """Dùng Groq JSON mode để phân loại intent."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT}
        ]
        # Đưa history vào context
        for msg in history[-6:]:
            messages.append(msg)
        messages.append({"role": "user", "content": query})

        try:
            data = groq_client.chat_json(messages)
            return BrainDecision(
                intent=data.get("intent", "direct_answer"),
                reasoning=data.get("reasoning", ""),
                direct_response=data.get("direct_response"),
                rag_query=data.get("rag_query"),
                tool_hint=data.get("tool_hint"),
            )
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            # Fallback: trả lời trực tiếp
            return BrainDecision(
                intent=IntentType.DIRECT_ANSWER,
                reasoning=f"Fallback do lỗi: {e}",
            )

    # ── Direct Chat ──────────────────────────────────────

    def _direct_chat(
        self, query: str, history: list[dict[str, str]]
    ) -> str:
        """Trả lời trực tiếp không cần RAG/tools (SYNC)."""
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý tư vấn học thuật AI Academy. "
                    "Trả lời thân thiện, chuyên nghiệp, bằng tiếng Việt."
                ),
            }
        ]
        for msg in history[-10:]:
            messages.append(msg)
        messages.append({"role": "user", "content": query})

        try:
            return groq_client.chat(messages)
        except Exception as e:
            logger.error(f"Direct chat failed: {e}")
            return "Xin lỗi, tôi đang gặp sự cố. Vui lòng thử lại sau."

    # ── RAG Handler ──────────────────────────────────────

    def _handle_rag(
        self,
        query: str,
        original_query: str,
        history: list[dict[str, str]],
    ) -> tuple[str, bool]:
        """Retrieve context từ knowledge base → augment prompt → trả lời."""
        # Retrieve
        context = rag_retriever.retrieve_as_context(query)

        if not context:
            logger.info("[RAG] No relevant context found")
            # Fallback: trả lời mà không có context
            messages: list[dict[str, str]] = [
                {
                    "role": "system",
                    "content": (
                        "Bạn là trợ lý tư vấn học thuật AI Academy. "
                        "Không tìm thấy thông tin trong tài liệu nội bộ. "
                        "Hãy trả lời trung thực rằng bạn chưa có thông tin này "
                        "và gợi ý liên hệ support@aiacademy.vn hoặc hotline."
                    ),
                }
            ]
            for msg in history[-6:]:
                messages.append(msg)
            messages.append({"role": "user", "content": original_query})

            try:
                answer = groq_client.chat(messages)
            except Exception as e:
                logger.error(f"RAG fallback chat failed: {e}")
                answer = (
                    "Xin lỗi, tôi chưa tìm thấy thông tin liên quan. "
                    "Vui lòng liên hệ support@aiacademy.vn để được hỗ trợ."
                )
            return answer, False

        # Augmented generation
        logger.info(f"[RAG] Found context (len={len(context)})")
        messages = [
            {
                "role": "system",
                "content": RAG_ANSWER_PROMPT.format(context=context),
            }
        ]
        for msg in history[-6:]:
            messages.append(msg)
        messages.append({"role": "user", "content": original_query})

        try:
            answer = groq_client.chat(messages, max_tokens=2048)
        except Exception as e:
            logger.error(f"RAG answer failed: {e}")
            answer = "Xin lỗi, tôi gặp lỗi khi xử lý. Vui lòng thử lại."

        return answer, True

    # ── Tool Handler (ReAct) ──────────────────────────────

    def _handle_tools(
        self, query: str, history: list[dict[str, str]]
    ) -> tuple[str, list[str], ReActTrace | None]:
        """Vào ReAct loop với tools. Trả thêm trace cho debug."""
        answer, tools_used, trace = run_react_loop(
            query=query,
            tools=self.tools,
            chat_history=history,
        )
        return answer, tools_used, trace


# Singleton
orchestrator = AgentOrchestrator()
