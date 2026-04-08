"""
app/agent/orchestrator.py
──────────────────────────
LangGraph-based Orchestrator.

Mapping từ flow cũ sang LangGraph:
    Router -> handle_message()
    Brain Intent -> node `classify_intent`
    [Direct / RAG / ReAct] -> các node xử lý có conditional edges
    Memory Update -> node `memory_update` + checkpointer của LangGraph
    Response -> ChatResponse trả về API
"""

from __future__ import annotations

import time
from typing import Annotated, Literal, TypedDict

from loguru import logger
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq

from app.core.config import settings
from app.core.groq_client import groq_client
from app.models.schemas import (
    BrainDecision,
    ChatResponse,
    IntentType,
    ReActTrace,
    ReActTraceStep,
)
from app.rag.retriever import rag_retriever
from app.agent.react_loop_langchain import run_react_loop_langchain
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
- "direct_answer": CHỈ cho chào hỏi đơn giản (xin chào, cảm ơn) hoặc câu hỏi đã có đầy đủ câu trả lời trong lịch sử chat
- "use_rag": cho câu hỏi về chính sách, quy định, chương trình đào tạo, lộ trình học, chứng chỉ, giảng viên (thông tin chi tiết từ tài liệu nội bộ)
- "use_tools": cho CÁC TRƯỜNG HỢP SAU:
  + Tra cứu khóa học cụ thể (tên, giá, lịch) → tool_hint: "course_search"
  + Tính toán học phí, số liệu → tool_hint: "calculator"
  + **CÂU HỎI VỀ KIẾN THỨC BÊN NGOÀI, tin tức, xu hướng, giá cả thị trường, công nghệ mới, so sánh framework/ngôn ngữ, bất kỳ thông tin nào KHÔNG nằm trong hệ thống khóa học nội bộ** → tool_hint: "web_search"

LƯU Ý QUAN TRỌNG: Nếu câu hỏi cần thông tin từ thế giới thực bên ngoài (ví dụ: "Python nên học gì", "xu hướng AI 2025", "giá Bitcoin", "so sánh React vs Vue"), PHẢI chọn intent="use_tools" với tool_hint="web_search". KHÔNG ĐƯỢC trả direct_answer cho những câu hỏi cần tra cứu kiến thức.

Trả lời bằng tiếng Việt. Chỉ trả JSON, không thêm text ngoài JSON."""

RAG_ANSWER_PROMPT = """Bạn là trợ lý tư vấn học thuật AI Academy.

Dựa trên context từ tài liệu nội bộ dưới đây, hãy trả lời câu hỏi của người dùng.

CONTEXT:
{context}

QUY TẮC:
- Trả lời dựa trên context, không bịa thông tin
- Nếu context không đủ, nói rõ và gợi ý liên hệ hỗ trợ
- Trả lời thân thiện, rõ ràng, có cấu trúc
- Dùng bullet points khi cần liệt kê thông thường
- Đặc biệt, KHI SO SÁNH (ví dụ hai khái niệm) hoặc trình bày dữ liệu đối chiếu, BẮT BUỘC sử dụng thẻ BẢNG MARKDOWN (Markdown Table). CHÚ Ý: Phải dùng ký tự xuống dòng (\\n) giữa các hàng của bảng.
"""


class AgentGraphState(TypedDict, total=False):
    session_id: str
    user_query: str
    messages: Annotated[list[BaseMessage], add_messages]
    decision: BrainDecision
    answer: str
    tools_used: list[str]
    rag_used: bool
    trace: ReActTrace | None
    memory_updated: bool


class AgentOrchestrator:
    """Bộ điều phối trung tâm dùng LangGraph + ChatGroq."""

    def __init__(self) -> None:
        # Khởi tạo tools
        self.tools: dict[str, BaseTool] = {
            "course_search": CourseSearchTool(),
            "calculator": CalculatorTool(),
            "web_search": WebSearchTool(),
        }

        # LangChain Groq LLM
        self.llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0.3,
            max_retries=settings.groq_max_retries,
            timeout=settings.groq_timeout,
        )

        # LangGraph memory/checkpoint
        self._checkpointer = MemorySaver()
        self._thread_map: dict[str, str] = {}
        self._graph = self._build_graph()

        logger.info(
            f"Orchestrator initialized with tools: {list(self.tools.keys())}"
        )

    def initialize_rag(self) -> bool:
        """Khởi tạo RAG (gọi 1 lần khi startup)."""
        try:
            rag_retriever.initialize()
            logger.info("RAG initialized successfully")
            return True
        except Exception as e:
            logger.error(f"RAG initialization failed: {e}")
            return False

    def _build_graph(self):
        graph = StateGraph(AgentGraphState)

        graph.add_node("classify_intent", self._node_classify_intent)
        graph.add_node("direct", self._node_direct)
        graph.add_node("rag", self._node_rag)
        graph.add_node("react", self._node_react)
        graph.add_node("memory_update", self._node_memory_update)

        graph.add_edge(START, "classify_intent")
        graph.add_conditional_edges(
            "classify_intent",
            self._route_by_intent,
            {
                "direct": "direct",
                "rag": "rag",
                "react": "react",
            },
        )
        graph.add_edge("direct", "memory_update")
        graph.add_edge("rag", "memory_update")
        graph.add_edge("react", "memory_update")
        graph.add_edge("memory_update", END)

        return graph.compile(checkpointer=self._checkpointer)

    def _session_to_thread_id(self, session_id: str) -> str:
        return self._thread_map.get(session_id, session_id)

    # ── Entry point ──────────────────────────────────────

    def handle_message(
        self, session_id: str, message: str, debug: bool = False
    ) -> ChatResponse:
        """
        Xử lý 1 lượt hội thoại (SYNC — Groq SDK là synchronous).

        Flow:
          1. Invoke LangGraph với state hiện tại (messages được persist theo thread)
          2. Graph tự route: direct / rag / react
          3. Graph update memory trong state
          4. Trả response (kèm trace nếu debug=True)
        """
        total_start = time.time()
        logger.info(f"[Orchestrator] session={session_id}, query={message[:80]}")

        thread_id = self._session_to_thread_id(session_id)
        config = {"configurable": {"thread_id": thread_id}}

        result = self._graph.invoke(
            {
                "session_id": session_id,
                "user_query": message,
                "messages": [HumanMessage(content=message)],
            },
            config=config,
        )

        decision: BrainDecision = result["decision"]
        answer: str = result.get("answer", "")
        tools_used: list[str] = result.get("tools_used", [])
        rag_used: bool = result.get("rag_used", False)
        trace: ReActTrace | None = result.get("trace")

        total_time = time.time() - total_start
        logger.info(f"[Orchestrator] Total processing time: {total_time:.2f}s")

        # 5. Trả response
        return ChatResponse(
            session_id=session_id,
            answer=answer,
            brain_reasoning=decision.reasoning,
            intent=decision.intent.value,
            tools_used=tools_used,
            rag_used=rag_used,
            trace=trace if (debug or settings.debug_mode) else None,
        )

    # ── Session helpers cho API ──────────────────────────

    def clear_session(self, session_id: str) -> None:
        """
        Reset session bằng cách map sang thread_id mới.
        (Không cần xóa dữ liệu checkpoint cũ.)
        """
        self._thread_map[session_id] = f"{session_id}:{time.time_ns()}"

    def get_history_as_dicts(self, session_id: str) -> list[dict[str, str]]:
        thread_id = self._session_to_thread_id(session_id)
        snapshot = self._graph.get_state({"configurable": {"thread_id": thread_id}})
        values = snapshot.values or {}
        messages = values.get("messages", [])

        out: list[dict[str, str]] = []
        for m in messages:
            if isinstance(m, HumanMessage):
                out.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                out.append({"role": "assistant", "content": m.content})
        return out

    # ── Graph nodes ───────────────────────────────────────

    def _node_classify_intent(self, state: AgentGraphState) -> AgentGraphState:
        history_dicts = self._to_role_dicts(state.get("messages", []))
        decision = self._classify_intent(state["user_query"], history_dicts)
        logger.info(f"[Orchestrator] intent={decision.intent}, reasoning={decision.reasoning[:80]}")
        return {"decision": decision}

    def _node_direct(self, state: AgentGraphState) -> AgentGraphState:
        decision = state["decision"]
        history_dicts = self._to_role_dicts(state.get("messages", []))
        answer = decision.direct_response or self._direct_chat(state["user_query"], history_dicts)
        
        # Tạo trace cho luồng DIRECT để FE luôn hiển thị quá trình Planning
        trace = ReActTrace(query=state["user_query"])
        trace.steps.append(ReActTraceStep(
            iteration=1,
            thought=f"Phân loại Intent: DIRECT_ANSWER. Lý do: {decision.reasoning}",
            observation="Trả lời trực tiếp không cần tra cứu tool hay tài liệu.",
        ))
        trace.final_answer = answer
        trace.total_iterations = 1
        
        return {
            "answer": answer,
            "tools_used": [],
            "rag_used": False,
            "trace": trace,
            "messages": [AIMessage(content=answer)],
        }

    def _node_rag(self, state: AgentGraphState) -> AgentGraphState:
        decision = state["decision"]
        history_dicts = self._to_role_dicts(state.get("messages", []))
        
        rag_query = decision.rag_query or state["user_query"]
        
        # Tạo trace cho luồng RAG
        trace = ReActTrace(query=state["user_query"])
        trace.steps.append(ReActTraceStep(
            iteration=1,
            thought=f"Phân loại Intent: USE_RAG. Lý do: {decision.reasoning}",
            observation=f"Đang tìm kiếm tài liệu nội bộ với truy vấn: \"{rag_query}\"",
        ))
        
        answer, rag_used = self._handle_rag(
            query=rag_query,
            original_query=state["user_query"],
            history=history_dicts,
        )
        
        trace.steps.append(ReActTraceStep(
            iteration=2,
            thought="Đã tìm thấy context từ tài liệu nội bộ. Đang tổng hợp câu trả lời." if rag_used else "Không tìm thấy tài liệu phù hợp. Trả lời dựa trên kiến thức chung.",
            observation=f"RAG retrieved: {'Có dữ liệu' if rag_used else 'Không có dữ liệu'}",
        ))
        trace.final_answer = answer
        trace.total_iterations = 2
        
        return {
            "answer": answer,
            "tools_used": [],
            "rag_used": rag_used,
            "trace": trace,
            "messages": [AIMessage(content=answer)],
        }

    def _node_react(self, state: AgentGraphState) -> AgentGraphState:
        history_dicts = self._to_role_dicts(state.get("messages", []))
        answer, tools_used, trace = self._handle_tools(state["user_query"], history_dicts)
        return {
            "answer": answer,
            "tools_used": tools_used,
            "rag_used": False,
            "trace": trace,
            "messages": [AIMessage(content=answer)],
        }

    def _node_memory_update(self, state: AgentGraphState) -> AgentGraphState:
        # LangGraph checkpointer sẽ persist state theo thread_id.
        return {"memory_updated": True}

    def _route_by_intent(self, state: AgentGraphState) -> Literal["direct", "rag", "react"]:
        intent = state["decision"].intent
        if intent == IntentType.USE_RAG:
            return "rag"
        if intent == IntentType.USE_TOOLS:
            return "react"
        return "direct"

    def _to_role_dicts(self, messages: list[BaseMessage]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for m in messages:
            if isinstance(m, HumanMessage):
                out.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                out.append({"role": "assistant", "content": m.content})
        return out

    # ── Intent Classification (Brain) ────────────────────

    def _classify_intent(
        self, query: str, history: list[dict[str, str]]
    ) -> BrainDecision:
        """Dùng ChatGroq structured output để phân loại intent."""
        lc_messages: list[BaseMessage] = [SystemMessage(content=INTENT_SYSTEM_PROMPT)]
        for msg in history[-6:]:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
        lc_messages.append(HumanMessage(content=query))

        try:
            structured = self.llm.with_structured_output(BrainDecision)
            result = structured.invoke(lc_messages)
            if isinstance(result, BrainDecision):
                return result
            return BrainDecision(**result)
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
        lc_messages: list[BaseMessage] = [
            SystemMessage(
                content=(
                    "Bạn là trợ lý tư vấn học thuật AI Academy. "
                    "Trả lời thân thiện, chuyên nghiệp, bằng tiếng Việt. "
                    "Đặc biệt, KHI SO SÁNH hoặc liệt kê đa chiều, BẮT BUỘC sử dụng thẻ BẢNG MARKDOWN (Markdown Table). CHÚ Ý: Phải dùng ký tự xuống dòng (\\n) giữa các hàng của bảng."
                )
            )
        ]
        for msg in history[-10:]:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
        lc_messages.append(HumanMessage(content=query))

        try:
            resp = self.llm.invoke(lc_messages)
            return resp.content if isinstance(resp.content, str) else str(resp.content)
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
            messages: list[BaseMessage] = [
                SystemMessage(
                    content=(
                        "Bạn là trợ lý tư vấn học thuật AI Academy. "
                        "Không tìm thấy thông tin trong tài liệu nội bộ. "
                        "Hãy trả lời trung thực rằng bạn chưa có thông tin này "
                        "và gợi ý liên hệ support@aiacademy.vn hoặc hotline."
                    )
                )
            ]
            for msg in history[-6:]:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
            messages.append(HumanMessage(content=original_query))

            try:
                answer_msg = self.llm.invoke(messages)
                answer = (
                    answer_msg.content
                    if isinstance(answer_msg.content, str)
                    else str(answer_msg.content)
                )
            except Exception as e:
                logger.error(f"RAG fallback chat failed: {e}")
                answer = (
                    "Xin lỗi, tôi chưa tìm thấy thông tin liên quan. "
                    "Vui lòng liên hệ support@aiacademy.vn để được hỗ trợ."
                )
            return answer, False

        # Augmented generation
        logger.info(f"[RAG] Found context (len={len(context)})")
        messages: list[BaseMessage] = [
            SystemMessage(content=RAG_ANSWER_PROMPT.format(context=context))
        ]
        for msg in history[-6:]:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=original_query))

        try:
            answer_msg = self.llm.invoke(messages)
            answer = (
                answer_msg.content
                if isinstance(answer_msg.content, str)
                else str(answer_msg.content)
            )
        except Exception as e:
            logger.error(f"RAG answer failed: {e}")
            answer = "Xin lỗi, tôi gặp lỗi khi xử lý. Vui lòng thử lại."

        return answer, True

    # ── Tool Handler (ReAct) ──────────────────────────────

    def _handle_tools(
        self, query: str, history: list[dict[str, str]]
    ) -> tuple[str, list[str], ReActTrace | None]:
        """Vào ReAct loop với tools. Trả thêm trace cho debug."""
        answer, tools_used, trace = run_react_loop_langchain(
            query=query,
            tools=self.tools,
            llm=self.llm,
            chat_history=history,
        )
        return answer, tools_used, trace


# Singleton
orchestrator = AgentOrchestrator()
