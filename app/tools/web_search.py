"""
app/tools/web_search.py
────────────────────────
Web search tool — hiện tại là **mock/stub**.
Interface sẵn sàng để thay bằng Tavily, SerpAPI, Brave Search, v.v.

Để thay bằng Tavily:
  1. pip install tavily-python
  2. Thêm TAVILY_API_KEY vào .env
  3. Thay TavilySearchTool kế thừa BaseTool, gọi TavilyClient.search()
"""

from app.tools.base import BaseTool
from loguru import logger


class WebSearchTool(BaseTool):
    """
    Mock web search — trả về kết quả giả lập.
    Trong production, thay bằng API thật.
    """

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Tìm kiếm thông tin trên internet. "
            "Dùng khi câu hỏi cần thông tin bên ngoài mà hệ thống nội bộ không có. "
            "Input là câu truy vấn tìm kiếm, ví dụ: 'xu hướng AI 2025'."
        )

    def execute(self, input_text: str) -> str:
        query = input_text.strip()
        logger.info(f"[WebSearch MOCK] query='{query}'")

        # ── Mock response ─────────────────────────────────────
        # Trong production, thay đoạn dưới bằng API call thật
        return (
            f"[Web Search - Mock Result]\n"
            f"Query: {query}\n"
            f"---\n"
            f"Đây là kết quả mock. Trong phiên bản production, tool này sẽ gọi "
            f"API tìm kiếm thật (Tavily / SerpAPI / Brave Search) để lấy "
            f"thông tin cập nhật từ internet.\n"
            f"---\n"
            f"Gợi ý: Để tích hợp Tavily, cài `pip install tavily-python` "
            f"và thêm TAVILY_API_KEY vào .env."
        )


# ── Ví dụ implementation Tavily (uncomment khi sẵn sàng) ──────
#
# from tavily import TavilyClient
# from app.core.config import settings
#
# class TavilySearchTool(BaseTool):
#     def __init__(self):
#         self._client = TavilyClient(api_key=settings.tavily_api_key)
#
#     @property
#     def name(self) -> str:
#         return "web_search"
#
#     @property
#     def description(self) -> str:
#         return "Tìm kiếm thông tin trên internet qua Tavily."
#
#     def execute(self, input_text: str) -> str:
#         response = self._client.search(query=input_text, max_results=3)
#         results = []
#         for r in response.get("results", []):
#             results.append(f"• {r['title']}: {r['content'][:200]}")
#         return "\n".join(results) if results else "Không tìm thấy kết quả."
