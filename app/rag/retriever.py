"""
app/rag/retriever.py
─────────────────────
RAG Retriever — nhận query, embed, tìm top-k chunks gần nhất từ FAISS.

Luồng:
  Query → Embedding Model → FAISS similarity search → top-k DocumentChunks

Tính năng mới:
  • Distance threshold: bỏ chunks có khoảng cách quá xa (không liên quan)
  • Distance score: gắn vào DocumentChunk để đánh giá chất lượng

Dùng bởi: Orchestrator (khi Brain quyết định intent=use_rag) và ReAct loop.
"""

from __future__ import annotations

import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.models.schemas import DocumentChunk, RetrievalResult
from app.rag.ingest import DocumentIngestor


class RAGRetriever:
    """Retrieve relevant context từ FAISS vector store."""

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None
        self._index: faiss.IndexFlatL2 | None = None
        self._chunks: list[DocumentChunk] = []
        self._initialized = False

    def initialize(self, force_reingest: bool = False) -> None:
        """
        Khởi tạo retriever:
          1. Thử load index có sẵn từ đĩa
          2. Nếu không có (hoặc force) → ingest từ data/docs/
        """
        if self._initialized and not force_reingest:
            return

        # Load model
        if self._model is None:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self._model = SentenceTransformer(settings.embedding_model)

        # Thử load index đã lưu
        if not force_reingest:
            index, chunks = DocumentIngestor.load_index()
            if index is not None and index.ntotal > 0:
                self._index = index
                self._chunks = chunks
                self._initialized = True
                logger.info("RAG Retriever initialized from saved index")
                return

        # Ingest mới
        logger.info("Ingesting documents for RAG...")
        ingestor = DocumentIngestor()
        self._index, self._chunks = ingestor.ingest()
        self._initialized = True
        logger.info(f"RAG Retriever initialized: {len(self._chunks)} chunks indexed")

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        distance_threshold: float | None = None,
    ) -> RetrievalResult:
        """
        Tìm top-k chunks liên quan nhất cho query.

        Args:
            query: câu truy vấn
            top_k: số lượng chunks trả về (default từ config)
            distance_threshold: bỏ chunks có distance > threshold
                                (default từ config)

        Returns:
            RetrievalResult chứa list chunks + query gốc
        """
        if not self._initialized:
            self.initialize()

        if self._index is None or self._index.ntotal == 0:
            logger.warning("FAISS index is empty — no documents ingested")
            return RetrievalResult(query=query, chunks=[])

        k = min(top_k or settings.rag_top_k, self._index.ntotal)
        threshold = distance_threshold or settings.rag_distance_threshold

        # Embed query
        if self._model is None:
            self._model = SentenceTransformer(settings.embedding_model)

        query_vec = self._model.encode([query], convert_to_numpy=True)
        query_vec = np.array(query_vec, dtype=np.float32)

        # Search
        distances, indices = self._index.search(query_vec, k)

        # Map index → chunk, áp dụng distance threshold
        result_chunks: list[DocumentChunk] = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self._chunks) and idx >= 0:
                dist = float(distances[0][i])

                # Bỏ chunks quá xa (không liên quan)
                if dist > threshold:
                    logger.debug(
                        f"  [RAG] SKIP rank={i+1} dist={dist:.4f} > "
                        f"threshold={threshold} — not relevant"
                    )
                    continue

                chunk = self._chunks[idx]
                # Gắn distance vào chunk để downstream biết chất lượng
                enriched_chunk = DocumentChunk(
                    text=chunk.text,
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                    distance=dist,
                )
                result_chunks.append(enriched_chunk)
                logger.debug(
                    f"  [RAG] rank={i+1} dist={dist:.4f} "
                    f"source={chunk.source} chunk={chunk.chunk_index}"
                )

        logger.info(
            f"[RAG] Retrieved {len(result_chunks)}/{k} chunks "
            f"(threshold={threshold})"
        )
        return RetrievalResult(query=query, chunks=result_chunks)

    def retrieve_as_context(self, query: str, top_k: int | None = None) -> str:
        """Retrieve + format thành context string để đưa vào prompt."""
        result = self.retrieve(query, top_k)
        if not result.chunks:
            return ""

        parts: list[str] = []
        for i, chunk in enumerate(result.chunks, 1):
            dist_info = f" (relevance: {chunk.distance:.3f})" if chunk.distance is not None else ""
            parts.append(
                f"[Nguồn: {chunk.source}, đoạn {chunk.chunk_index + 1}{dist_info}]\n{chunk.text}"
            )
        return "\n\n---\n\n".join(parts)


# Singleton
rag_retriever = RAGRetriever()
