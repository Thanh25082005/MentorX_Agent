"""
app/agent/react_loop_langchain.py
──────────────────────────────────
ReAct loop dùng ChatGroq (LangChain) + structured output.
Giữ nguyên logic/prompt của phiên bản custom.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from loguru import logger

from app.core.config import settings
from app.models.schemas import (
    Observation,
    ReActStep,
    ReActTrace,
    ReActTraceStep,
)
from app.tools.base import BaseTool


REACT_SYSTEM_PROMPT = """Bạn là trợ lý tư vấn học thuật AI Academy. Bạn đang trong vòng lặp suy luận (ReAct loop).

Bạn có các tools sau:
{tool_descriptions}

Tại mỗi bước, bạn PHẢI trả về JSON với format:

Nếu cần gọi tool:
{{
  "thought": "Suy luận về bước tiếp theo",
  "action": {{
    "tool": "tên_tool",
    "input": "input cho tool"
  }},
  "final_answer": null
}}

Nếu đã đủ thông tin để trả lời:
{{
  "thought": "Tôi đã có đủ thông tin để trả lời",
  "action": null,
  "final_answer": "Câu trả lời hoàn chỉnh cho người dùng"
}}

QUY TẮC:
- Luôn suy nghĩ (thought) trước khi hành động
- Chọn tool phù hợp nhất:
  + course_search: CHỈ dùng khi cần tra cứu KHÓA HỌC trong hệ thống nội bộ (tên khóa, giá, lịch học)
  + calculator: tính toán số liệu
  + web_search: dùng khi cần kiến thức BÊN NGOÀI hệ thống (xu hướng công nghệ, tin tức, giá thị trường, so sánh framework, lộ trình học chung, kiến thức phổ thông mà hệ thống nội bộ KHÔNG CÓ)
- QUAN TRỌNG: Nếu câu hỏi cần thông tin ngoài dữ liệu khóa học nội bộ, PHẢI dùng web_search TRƯỚC
- Nếu tool lỗi, thử cách khác hoặc dùng tool khác
- Trả lời bằng tiếng Việt
- final_answer phải rõ ràng, đầy đủ, thân thiện
- KHI SO SÁNH HOẶC trình bày dữ liệu đa chiều, BẮT BUỘC sử dụng FORMAT BẢNG MARKDOWN (Markdown Table) để trực quan. CHÚ Ý: Phải dùng ký tự xuống dòng (\\n) giữa các hàng của bảng.
- Chỉ trả JSON, không thêm text ngoài JSON
"""


def _build_tool_descriptions(tools: dict[str, BaseTool]) -> str:
    lines = []
    for name, tool in tools.items():
        lines.append(f"- **{name}**: {tool.description}")
    return "\n".join(lines)


def _build_observation_history(observations: list[Observation]) -> str:
    if not observations:
        return ""
    parts = []
    for i, obs in enumerate(observations, 1):
        status = "✓" if obs.success else "✗"
        parts.append(
            f"[Bước {i}] {status} Tool: {obs.tool} | Input: {obs.input}\n"
            f"Kết quả: {obs.output}"
        )
    return "\n\n".join(parts)


def _to_lc_messages(history: list[dict[str, str]] | None) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for msg in (history or []):
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out


def _invoke_react_step(llm: ChatGroq, messages: list[BaseMessage]) -> ReActStep:
    structured = llm.with_structured_output(ReActStep)
    result: Any = structured.invoke(messages)
    if isinstance(result, ReActStep):
        return result
    return ReActStep(**result)


def run_react_loop_langchain(
    query: str,
    tools: dict[str, BaseTool],
    llm: ChatGroq,
    chat_history: list[dict[str, str]] | None = None,
    max_iterations: int | None = None,
) -> tuple[str, list[str], ReActTrace]:
    max_iter = max_iterations or settings.react_max_iterations
    tool_descriptions = _build_tool_descriptions(tools)
    system_prompt = REACT_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)

    observations: list[Observation] = []
    tools_used: list[str] = []
    trace = ReActTrace(query=query)

    for iteration in range(1, max_iter + 1):
        logger.info(f"── ReAct iteration {iteration}/{max_iter} ──")

        messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
        if chat_history:
            messages.extend(_to_lc_messages(chat_history[-6:]))

        user_content = f"Câu hỏi của người dùng: {query}"
        obs_history = _build_observation_history(observations)
        if obs_history:
            user_content += f"\n\nKết quả các bước trước:\n{obs_history}"
            user_content += "\n\nHãy tiếp tục suy luận và quyết định bước tiếp theo."

        messages.append(HumanMessage(content=user_content))

        try:
            step = _invoke_react_step(llm, messages)
        except Exception as e:
            logger.error(f"ReAct: LLM error at iteration {iteration}: {e}")
            trace.final_answer = "Xin lỗi, tôi gặp lỗi khi xử lý. Vui lòng thử lại sau."
            trace.total_iterations = iteration
            trace.tools_called = tools_used
            return trace.final_answer, tools_used, trace

        logger.info(f"  Thought: {step.thought[:100]}")

        trace_step = ReActTraceStep(iteration=iteration, thought=step.thought)

        if step.final_answer:
            logger.info(f"  → Final answer (len={len(step.final_answer)})")
            trace_step.observation = "[Final answer reached]"
            trace.steps.append(trace_step)
            trace.final_answer = step.final_answer
            trace.total_iterations = iteration
            trace.tools_called = tools_used
            return step.final_answer, tools_used, trace

        if step.action is None:
            logger.warning("  → No action and no final_answer, forcing finish")
            trace.steps.append(trace_step)
            final = _force_final_answer_langchain(
                llm=llm,
                query=query,
                observations=observations,
                system_prompt=system_prompt,
                chat_history=chat_history,
            )
            trace.final_answer = final
            trace.total_iterations = iteration
            trace.tools_called = tools_used
            return final, tools_used, trace

        tool_name = step.action.tool
        tool_input = step.action.input
        logger.info(f"  Action: tool={tool_name}, input={tool_input[:80]}")

        trace_step.action_tool = tool_name
        trace_step.action_input = tool_input

        if tool_name not in tools:
            obs_output = (
                f"Tool '{tool_name}' không tồn tại. "
                f"Các tool có sẵn: {', '.join(tools.keys())}"
            )
            observations.append(
                Observation(
                    tool=tool_name,
                    input=tool_input,
                    output=obs_output,
                    success=False,
                    error="Tool not found",
                )
            )
            trace_step.observation = obs_output
            trace_step.observation_success = False
            trace.steps.append(trace_step)
            continue

        output, success = tools[tool_name].safe_execute(tool_input)
        tools_used.append(tool_name)

        observations.append(
            Observation(
                tool=tool_name,
                input=tool_input,
                output=output,
                success=success,
                error=None if success else output,
            )
        )

        trace_step.observation = output
        trace_step.observation_success = success
        trace.steps.append(trace_step)

    logger.warning(f"ReAct: Max iterations ({max_iter}) reached, forcing answer")
    final = _force_final_answer_langchain(
        llm=llm,
        query=query,
        observations=observations,
        system_prompt=system_prompt,
        chat_history=chat_history,
    )
    trace.final_answer = final
    trace.total_iterations = max_iter
    trace.tools_called = tools_used
    return final, tools_used, trace


def _force_final_answer_langchain(
    llm: ChatGroq,
    query: str,
    observations: list[Observation],
    system_prompt: str,
    chat_history: list[dict[str, str]] | None,
) -> str:
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    if chat_history:
        messages.extend(_to_lc_messages(chat_history[-4:]))

    obs_text = _build_observation_history(observations)
    messages.append(
        HumanMessage(
            content=(
                f"Câu hỏi: {query}\n\n"
                f"Kết quả thu thập:\n{obs_text}\n\n"
                f"Hãy tổng hợp và đưa ra câu trả lời cuối cùng. "
                f"Trả JSON với final_answer."
            )
        )
    )

    try:
        step = _invoke_react_step(llm, messages)
        return step.final_answer or "Xin lỗi, tôi không thể tổng hợp câu trả lời."
    except Exception as e:
        logger.error(f"Force final answer failed: {e}")
        return "Xin lỗi, tôi gặp lỗi khi tổng hợp câu trả lời."
