"""
app/main.py
────────────
Entry point của ứng dụng FastAPI.

Startup:
  1. Load .env
  2. Khởi tạo RAG (ingest documents nếu chưa có index)
  3. Mount API routes

Tính năng:
  • Request logging middleware (method, path, response time)
  • CORS support
  • Lifespan management

Chạy:
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import router
from app.agent.orchestrator import orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup & shutdown events."""
    # ── Startup ──
    logger.info("🚀 Starting AI Agent...")

    # Khởi tạo RAG pipeline (load hoặc ingest documents)
    try:
        orchestrator.initialize_rag()
        logger.info("✅ RAG initialized")
    except Exception as e:
        logger.warning(f"⚠️ RAG init failed (agent vẫn hoạt động, nhưng RAG sẽ trống): {e}")

    logger.info("✅ AI Agent ready!")
    logger.info("📚 Tools: " + ", ".join(orchestrator.tools.keys()))
    logger.info("📖 Docs: http://localhost:8000/docs")

    yield

    # ── Shutdown ──
    logger.info("🛑 Shutting down AI Agent...")


# ── FastAPI app ──────────────────────────────────────────

app = FastAPI(
    title="AI Academy Agent",
    description=(
        "AI Agent tư vấn học thuật với Groq LLM, RAG, ReAct loop, "
        "và multi-tool support."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

# CORS — cho phép frontend truy cập
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production: thay bằng domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Logging Middleware ───────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log mỗi request: method, path, response time."""
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(
        f"[HTTP] {request.method} {request.url.path} "
        f"→ {response.status_code} ({duration:.3f}s)"
    )
    return response


# Mount routes
app.include_router(router)


# ── Root route ───────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "AI Academy Agent",
        "version": "1.1.0",
        "docs": "/docs",
        "api": "/api/chat",
        "health": "/api/health",
    }
