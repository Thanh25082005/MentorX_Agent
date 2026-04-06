"""
app/agent/react_loop.py
────────────────────────
ReAct (Reasoning + Acting) loop.

Vòng lặp gồm 3 pha:
  1. Thought — Brain suy luận bước tiếp theo cần làm gì
  2. Action  — Gọi tool phù hợp
  3. Observation — Nhận kết quả từ tool

Brain tương tác qua structured JSON output (Groq JSON mode).
Loop dừng khi:
  • Brain trả final_answer
  • Đạt max_iterations
  • Lỗi không phục hồi được

Trace: Mỗi lần chạy tạo ReActTrace object chứa toàn bộ chain-of-thought
       → hữu ích cho debugging, Langfuse, evaluation.
"""

from __future__ import annotations

from loguru import logger

from app.core.config import settings
from app.core.groq_client import groq_client
from app.models.schemas import (
    Observation,
    ReActStep,
    ReActTrace,
    ReActTraceStep,
    ToolAction,
)
from app.tools.base import BaseTool


# ── System prompt cho ReAct ───────────────────────────────

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
- Chọn tool phù hợp nhất
- Nếu tool lỗi, thử cách khác hoặc dùng tool khác
- Trả lời bằng tiếng Việt
- final_answer phải rõ ràng, đầy đủ, thân thiện
- Chỉ trả JSON, không thêm text ngoài JSON
"""


def _build_tool_descriptions(tools: dict[str, BaseTool]) -> str:
    """Format danh sách tools cho system prompt."""
    lines = []
    for name, tool in tools.items():
        lines.append(f"- **{name}**: {tool.description}")
    return "\n".join(lines)


def _build_observation_history(observations: list[Observation]) -> str:
    """Format lịch sử observations thành text cho prompt."""
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


def _parse_react_step(data: dict) -> ReActStep:
    """Parse JSON response từ Groq thành ReActStep."""
    action = None
    if data.get("action"):
        action = ToolAction(
            tool=data["action"].get("tool", ""),
            input=data["action"].get("input", ""),
        )
    return ReActStep(
        thought=data.get("thought", ""),
        action=action,
        final_answer=data.get("final_answer"),
    )


def run_react_loop(
    query: str,
    tools: dict[str, BaseTool],
    chat_history: list[dict[str, str]] | None = None,
    max_iterations: int | None = None,
) -> tuple[str, list[str], ReActTrace]:
    """
    Chạy ReAct loop.

    Args:
        query: câu hỏi người dùng
        tools: dict name→tool đã khởi tạo
        chat_history: lịch sử hội thoại (optional)
        max_iterations: giới hạn vòng lặp

    Returns:
        (final_answer, list_of_tools_used, trace)
    """
    max_iter = max_iterations or settings.react_max_iterations
    tool_descriptions = _build_tool_descriptions(tools)
    system_prompt = REACT_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)

    observations: list[Observation] = []
    tools_used: list[str] = []

    # ── Trace object ─────────────────────────────────────
    trace = ReActTrace(query=query)

    for iteration in range(1, max_iter + 1):
        logger.info(f"── ReAct iteration {iteration}/{max_iter} ──")

        # ── Build messages cho Groq ──────────────────────
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        # Thêm chat history (context)
        if chat_history:
            # Chỉ lấy vài tin nhắn gần nhất để không vượt context
            for msg in chat_history[-6:]:
                messages.append(msg)

        # User query + observations
        user_content = f"Câu hỏi của người dùng: {query}"
        obs_history = _build_observation_history(observations)
        if obs_history:
            user_content += f"\n\nKết quả các bước trước:\n{obs_history}"
            user_content += "\n\nHãy tiếp tục suy luận và quyết định bước tiếp theo."

        messages.append({"role": "user", "content": user_content})

        # ── Gọi Groq JSON mode ──────────────────────────
        try:
            data = groq_client.chat_json(messages)
        except Exception as e:
            logger.error(f"ReAct: Groq error at iteration {iteration}: {e}")
            trace.final_answer = (
                "Xin lỗi, tôi gặp lỗi khi xử lý. Vui lòng thử lại sau."
            )
            trace.total_iterations = iteration
            trace.tools_called = tools_used
            return trace.final_answer, tools_used, trace

        if "error" in data:
            logger.error(f"ReAct: JSON parse error: {data}")
            trace.final_answer = (
                "Xin lỗi, tôi gặp lỗi khi phân tích. Vui lòng thử lại."
            )
            trace.total_iterations = iteration
            trace.tools_called = tools_used
            return trace.final_answer, tools_used, trace

        # ── Parse step ───────────────────────────────────
        step = _parse_react_step(data)
        logger.info(f"  Thought: {step.thought[:100]}")

        # Tạo trace step
        trace_step = ReActTraceStep(
            iteration=iteration,
            thought=step.thought,
        )

        # ── Check final answer ───────────────────────────
        if step.final_answer:
            logger.info(f"  → Final answer (len={len(step.final_answer)})")
            trace_step.observation = "[Final answer reached]"
            trace.steps.append(trace_step)
            trace.final_answer = step.final_answer
            trace.total_iterations = iteration
            trace.tools_called = tools_used
            return step.final_answer, tools_used, trace

        # ── Execute action ───────────────────────────────
        if step.action is None:
            logger.warning("  → No action and no final_answer, forcing finish")
            trace.steps.append(trace_step)
            final = _force_final_answer(
                query, observations, system_prompt, chat_history
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
            logger.warning(f"  → Unknown tool: {tool_name}")
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

        # Gọi tool qua safe_execute
        tool = tools[tool_name]
        output, success = tool.safe_execute(tool_input)
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

        logger.info(f"  Observation: success={success}, len={len(output)}")

    # ── Hết max iterations → force tổng hợp ─────────────
    logger.warning(f"ReAct: Max iterations ({max_iter}) reached, forcing answer")
    final = _force_final_answer(query, observations, system_prompt, chat_history)
    trace.final_answer = final
    trace.total_iterations = max_iter
    trace.tools_called = tools_used
    return final, tools_used, trace


def _force_final_answer(
    query: str,
    observations: list[Observation],
    system_prompt: str,
    chat_history: list[dict[str, str]] | None,
) -> str:
    """Khi hết iterations, gọi Groq 1 lần cuối để tổng hợp câu trả lời."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]
    if chat_history:
        for msg in chat_history[-4:]:
            messages.append(msg)

    obs_text = _build_observation_history(observations)
    messages.append({
        "role": "user",
        "content": (
            f"Câu hỏi: {query}\n\n"
            f"Kết quả thu thập:\n{obs_text}\n\n"
            f"Hãy tổng hợp và đưa ra câu trả lời cuối cùng. "
            f"Trả JSON với final_answer."
        ),
    })

    try:
        data = groq_client.chat_json(messages)
        return data.get(
            "final_answer",
            "Xin lỗi, tôi không thể tổng hợp câu trả lời.",
        )
    except Exception as e:
        logger.error(f"Force final answer failed: {e}")
        return "Xin lỗi, tôi gặp lỗi khi tổng hợp câu trả lời."
