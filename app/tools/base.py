"""
app/tools/base.py
─────────────────
Abstract base class cho mọi tool.
Mỗi tool cần khai báo: name, description, execute().
Orchestrator / ReAct loop dùng interface này để gọi tool một cách đồng nhất.
"""

from abc import ABC, abstractmethod
from loguru import logger


class BaseTool(ABC):
    """Interface chung cho tất cả tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tên tool (dùng trong prompt cho LLM chọn)."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Mô tả chức năng — LLM đọc description này để quyết định có gọi hay không."""
        ...

    @abstractmethod
    def execute(self, input_text: str) -> str:
        """
        Thực thi tool.

        Args:
            input_text: chuỗi input từ LLM (qua ReAct action).

        Returns:
            Chuỗi kết quả, sẽ trở thành Observation trong ReAct loop.
        """
        ...

    def safe_execute(self, input_text: str) -> tuple[str, bool]:
        """
        Wrapper an toàn: bắt exception → trả (output, success).
        ReAct loop nên gọi hàm này thay vì execute() trực tiếp.
        """
        try:
            result = self.execute(input_text)
            return result, True
        except Exception as e:
            logger.error(f"Tool '{self.name}' lỗi với input='{input_text}': {e}")
            return f"Lỗi khi thực thi tool {self.name}: {str(e)}", False
