"""
app/tools/course_search.py
───────────────────────────
Local search tool — đọc courses.csv bằng pandas.
Tìm kiếm theo keyword trên nhiều cột: name, category, description, level.
Dễ thay bằng PostgreSQL / Text-to-SQL sau này.
"""

import pandas as pd
from loguru import logger

from app.core.config import settings
from app.tools.base import BaseTool


class CourseSearchTool(BaseTool):

    def __init__(self) -> None:
        self._df: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return "course_search"

    @property
    def description(self) -> str:
        return (
            "Tìm kiếm khóa học trong hệ thống. "
            "Input là từ khóa tìm kiếm, ví dụ: 'Python', 'Machine Learning', 'Beginner', 'giá rẻ'. "
            "Trả về danh sách khóa học phù hợp với tên, giá, lịch, giảng viên, level."
        )

    def _load_data(self) -> pd.DataFrame:
        """Lazy-load CSV — chỉ đọc 1 lần, cache trong memory."""
        if self._df is None:
            try:
                self._df = pd.read_csv(settings.courses_csv_path, encoding="utf-8")
                logger.info(
                    f"Loaded {len(self._df)} courses from {settings.courses_csv_path}"
                )
            except FileNotFoundError:
                logger.error(f"File not found: {settings.courses_csv_path}")
                self._df = pd.DataFrame()
        return self._df

    def execute(self, input_text: str) -> str:
        df = self._load_data()
        if df.empty:
            return "Không có dữ liệu khóa học."

        query = input_text.strip().lower()
        keywords = query.split()

        # Tìm trên nhiều cột
        search_cols = ["name", "category", "description", "level", "instructor"]
        mask = pd.Series([False] * len(df), index=df.index)

        for kw in keywords:
            kw_mask = pd.Series([False] * len(df), index=df.index)
            for col in search_cols:
                if col in df.columns:
                    kw_mask |= df[col].astype(str).str.lower().str.contains(
                        kw, na=False
                    )
            mask |= kw_mask

        results = df[mask]

        if results.empty:
            return f"Không tìm thấy khóa học nào liên quan đến '{input_text}'."

        # Format kết quả
        lines: list[str] = [f"Tìm thấy {len(results)} khóa học:"]
        for _, row in results.iterrows():
            price = f"{int(row['price_vnd']):,}" if pd.notna(row.get("price_vnd")) else "N/A"
            lines.append(
                f"• [{row['id']}] {row['name']} | "
                f"Level: {row.get('level','N/A')} | "
                f"Thời lượng: {row.get('duration_weeks','N/A')} tuần | "
                f"Học phí: {price} VNĐ | "
                f"Lịch: {row.get('schedule','N/A')} | "
                f"GV: {row.get('instructor','N/A')}"
            )
        return "\n".join(lines)
