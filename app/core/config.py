"""
app/core/config.py
───────────────────
Tập trung **tất cả** cấu hình hệ thống, đọc từ .env qua pydantic-settings.
Muốn thay Redis / Qdrant / Tavily → chỉ cần thêm field ở đây.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Thư mục gốc project (2 cấp lên từ file này)
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Groq ──────────────────────────────────────────────
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_timeout: int = 30          # seconds per request
    groq_max_retries: int = 3       # retry with exponential backoff

    # ── Embedding ─────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── RAG / FAISS ───────────────────────────────────────
    rag_backend: str = "qdrant"  # qdrant | faiss
    faiss_index_path: str = str(BASE_DIR / "data" / "vector_store")
    chunk_size: int = 500
    chunk_overlap: int = 50
    rag_top_k: int = 3
    rag_distance_threshold: float = 1.5  # bỏ chunks có distance > threshold

    # ── Qdrant ─────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "academy_docs"
    qdrant_path: str = str(BASE_DIR / "data" / "qdrant")

    # ── Memory ────────────────────────────────────────────
    short_term_max_turns: int = 20
    redis_url: str | None = None

    # ── Agent ─────────────────────────────────────────────
    react_max_iterations: int = 10
    debug_mode: bool = False        # trả ReAct trace trong response

    # ── Data paths ────────────────────────────────────────
    courses_csv_path: str = str(BASE_DIR / "data" / "courses.csv")
    docs_dir: str = str(BASE_DIR / "data" / "docs")


settings = Settings()
