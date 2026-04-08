"""
app/models/schemas.py
─────────────────────
Pydantic models cho toàn bộ hệ thống:
  • API request / response
  • Brain decision (intent classification)
  • ReAct step (thought / action / observation)
  • Tool interface schemas
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ╔══════════════════════════════════════════════════════════╗
# ║  API Layer                                               ║
# ╚══════════════════════════════════════════════════════════╝

class ChatRequest(BaseModel):
    """Request gửi từ client."""
    session_id: str = Field(
        default="default",
        description="ID phiên hội thoại, dùng để quản lý memory",
    )
    message: str = Field(..., min_length=1, description="Câu hỏi của người dùng")


class ChatResponse(BaseModel):
    """Response trả về cho client."""
    session_id: str
    answer: str
    brain_reasoning: str = Field(
        default="", description="Lý do Brain chọn intent (quy trình tư duy cấp cao)"
    )
    intent: str = Field(default="unknown", description="Intent Brain phân loại được")
    tools_used: list[str] = Field(default_factory=list)
    rag_used: bool = False
    trace: Optional["ReActTrace"] = Field(
        default=None, description="ReAct trace (chỉ trả khi debug=true)"
    )


# ╔══════════════════════════════════════════════════════════╗
# ║  Brain / Intent Classification                           ║
# ╚══════════════════════════════════════════════════════════╝

class IntentType(str, Enum):
    DIRECT_ANSWER = "direct_answer"
    USE_RAG = "use_rag"
    USE_TOOLS = "use_tools"


class BrainDecision(BaseModel):
    """Kết quả phân loại intent từ Brain (structured output)."""
    intent: IntentType
    reasoning: str = ""
    direct_response: Optional[str] = None
    rag_query: Optional[str] = None
    tool_hint: Optional[str] = None


# ╔══════════════════════════════════════════════════════════╗
# ║  ReAct Loop                                              ║
# ╚══════════════════════════════════════════════════════════╝

class ToolAction(BaseModel):
    """Hành động gọi tool trong ReAct loop."""
    tool: str = Field(..., description="Tên tool cần gọi")
    input: str = Field(..., description="Input truyền vào tool")


class ReActStep(BaseModel):
    """Một bước trong ReAct loop."""
    thought: str = ""
    action: Optional[ToolAction] = None
    final_answer: Optional[str] = None


class Observation(BaseModel):
    """Kết quả trả về từ tool."""
    tool: str
    input: str
    output: str
    success: bool = True
    error: Optional[str] = None


class ReActTraceStep(BaseModel):
    """Một bước đầy đủ trong ReAct trace (thought + action + observation)."""
    iteration: int
    thought: str = ""
    action_tool: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None
    observation_success: Optional[bool] = None


class ReActTrace(BaseModel):
    """Toàn bộ trace của một ReAct loop — dùng cho debugging & Langfuse."""
    query: str
    steps: list[ReActTraceStep] = Field(default_factory=list)
    final_answer: str = ""
    total_iterations: int = 0
    tools_called: list[str] = Field(default_factory=list)


# ╔══════════════════════════════════════════════════════════╗
# ║  Memory                                                  ║
# ╚══════════════════════════════════════════════════════════╝

class MemoryMessage(BaseModel):
    """Một tin nhắn trong lịch sử hội thoại."""
    role: str = Field(..., description="'user' hoặc 'assistant'")
    content: str


# ╔══════════════════════════════════════════════════════════╗
# ║  RAG                                                     ║
# ╚══════════════════════════════════════════════════════════╝

class DocumentChunk(BaseModel):
    """Một chunk tài liệu đã được chia nhỏ."""
    text: str
    source: str = ""
    chunk_index: int = 0
    distance: Optional[float] = Field(
        default=None, description="Khoảng cách FAISS (nhỏ hơn = liên quan hơn)"
    )


class RetrievalResult(BaseModel):
    """Kết quả retrieve từ vector store."""
    chunks: list[DocumentChunk] = Field(default_factory=list)
    query: str = ""
