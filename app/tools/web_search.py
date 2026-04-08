"""
app/tools/web_search.py
────────────────────────
Web search tool — sử dụng duckduckgo-search để lấy thông tin thực tế từ Internet.
Tool này miễn phí, không yêu cầu API key.
"""

from loguru import logger
try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

from app.tools.base import BaseTool

class WebSearchTool(BaseTool):
    """
    Tìm kiếm web thực tế lấy content bằng DuckDuckGo.
    """

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Tìm kiếm thông tin trên internet thực tế. "
            "Dùng khi câu hỏi cần thông tin cập nhật gần đây hoặc nằm ngoài dữ liệu hệ thống nội bộ. "
            "Input là câu truy vấn tìm kiếm, ví dụ: 'xu hướng học AI năm 2025'."
        )

    def execute(self, input_text: str) -> str:
        if DDGS is None:
            return "Thư viện duckduckgo-search chưa được cài đặt. Vui lòng cài bằng pip install duckduckgo-search."
            
        query = input_text.strip()
        logger.info(f"[WebSearch DDGS] query='{query}'")

        try:
            with DDGS() as ddgs:
                # max_results limit content overhead
                results = list(ddgs.text(query, max_results=5))

            if not results:
                return f"Không tìm thấy kết quả nào trên web cho từ khóa: '{query}'"

            # Format snippets into readable context
            formatted_results = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                body = r.get("body", "")
                link = r.get("href", "")
                formatted_results.append(f"{i}. **{title}**\n   {body}\n   [Nguồn]({link})")

            joined_results = "\n\n".join(formatted_results)
            return (
                f"[Kết quả Tìm kiếm Web (DuckDuckGo)]\n"
                f"Từ khóa: {query}\n"
                f"---\n"
                f"{joined_results}"
            )
        except Exception as e:
            logger.error(f"[WebSearch DDGS] Lỗi tìm kiếm: {e}")
            return f"Xảy ra lỗi khi tìm kiếm web: {str(e)}"
