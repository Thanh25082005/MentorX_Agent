"""
app/rag/ingest.py
──────────────────
Pipeline nạp tài liệu vào vector store:
  1. Đọc file .txt / .md / .pdf từ thư mục data/docs/
  2. Chia nhỏ thành chunks (character-based với overlap)
  3. Embed bằng sentence-transformers
  4. Lưu vào FAISS index

Chạy 1 lần khi khởi tạo hoặc khi có tài liệu mới.
Dễ thay FAISS → Qdrant / ChromaDB / Pinecone.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.models.schemas import DocumentChunk


class DocumentIngestor:
    """Đọc docs → chunk → embed → lưu FAISS."""

    def __init__(
        self,
        embedding_model_name: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self._model_name = embedding_model_name or settings.embedding_model
        self._chunk_size = chunk_size or settings.chunk_size
        self._chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load embedding model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
        return self._model

    # ── Bước 1: Đọc file ─────────────────────────────────

    def _read_file(self, path: Path) -> str:
        """Đọc nội dung file text/md/pdf."""
        suffix = path.suffix.lower()

        if suffix in (".txt", ".md"):
            return path.read_text(encoding="utf-8")

        if suffix == ".pdf":
            try:
                from PyPDF2 import PdfReader

                reader = PdfReader(str(path))
                pages = [page.extract_text() or "" for page in reader.pages]
                return "\n".join(pages)
            except ImportError:
                logger.warning("PyPDF2 not installed, skipping PDF")
                return ""

        logger.warning(f"Unsupported file type: {suffix}")
        return ""

    def _load_documents(self, docs_dir: str | None = None) -> list[tuple[str, str]]:
        """Đọc tất cả file trong thư mục → list[(content, source)]."""
        docs_path = Path(docs_dir or settings.docs_dir)
        if not docs_path.exists():
            logger.error(f"Docs directory not found: {docs_path}")
            return []

        documents: list[tuple[str, str]] = []
        for fpath in sorted(docs_path.rglob("*")):
            if fpath.is_file() and fpath.suffix.lower() in (".txt", ".md", ".pdf"):
                content = self._read_file(fpath)
                if content.strip():
                    documents.append((content, str(fpath.name)))
                    logger.info(f"Loaded: {fpath.name} ({len(content)} chars)")

        logger.info(f"Total documents loaded: {len(documents)}")
        return documents

    # ── Bước 2: Chunk ────────────────────────────────────

    def _chunk_text(self, text: str, source: str) -> list[DocumentChunk]:
        """Chia text thành chunks với overlap."""
        chunks: list[DocumentChunk] = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self._chunk_size
            chunk_text = text[start:end]

            # Cố gắng cắt tại ranh giới câu
            if end < len(text):
                last_period = chunk_text.rfind(".")
                last_newline = chunk_text.rfind("\n")
                cut_point = max(last_period, last_newline)
                if cut_point > self._chunk_size * 0.3:  # ít nhất 30% chunk
                    chunk_text = chunk_text[: cut_point + 1]
                    end = start + cut_point + 1

            chunks.append(
                DocumentChunk(text=chunk_text.strip(), source=source, chunk_index=idx)
            )
            idx += 1
            start = end - self._chunk_overlap

        return chunks

    # ── Bước 3 & 4: Embed + Index ─────────────────────────

    def ingest(self, docs_dir: str | None = None) -> tuple[faiss.IndexFlatL2, list[DocumentChunk]]:
        """
        Pipeline hoàn chỉnh: load → chunk → embed → FAISS index.

        Returns:
            (faiss_index, list_of_chunks) — chunks giữ song song với index
            để khi search có thể map vector_id → chunk text.
        """
        documents = self._load_documents(docs_dir)
        if not documents:
            logger.warning("No documents to ingest!")
            # Trả empty index
            model = self._get_model()
            dim = model.get_sentence_embedding_dimension()
            return faiss.IndexFlatL2(dim), []

        # Chunk tất cả documents
        all_chunks: list[DocumentChunk] = []
        for content, source in documents:
            chunks = self._chunk_text(content, source)
            all_chunks.extend(chunks)
        logger.info(f"Total chunks: {len(all_chunks)}")

        # Embed
        model = self._get_model()
        texts = [c.text for c in all_chunks]
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        # FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

        # Lưu ra đĩa
        self._save(index, all_chunks)

        return index, all_chunks

    def _save(self, index: faiss.IndexFlatL2, chunks: list[DocumentChunk]) -> None:
        """Persist FAISS index + chunks metadata ra đĩa."""
        store_dir = Path(settings.faiss_index_path)
        store_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(store_dir / "index.faiss"))
        with open(store_dir / "chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)

        logger.info(f"Saved FAISS index + chunks to {store_dir}")

    @staticmethod
    def load_index() -> tuple[faiss.IndexFlatL2 | None, list[DocumentChunk]]:
        """Load index + chunks từ đĩa (nếu có)."""
        store_dir = Path(settings.faiss_index_path)
        index_path = store_dir / "index.faiss"
        chunks_path = store_dir / "chunks.pkl"

        if not index_path.exists() or not chunks_path.exists():
            logger.warning("No saved FAISS index found.")
            return None, []

        index = faiss.read_index(str(index_path))
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        logger.info(f"Loaded FAISS index: {index.ntotal} vectors, {len(chunks)} chunks")
        return index, chunks
