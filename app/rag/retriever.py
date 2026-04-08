"""
app/rag/retriever.py
─────────────────────
RAG Retriever — nhận query, embed, tìm top-k chunks gần nhất từ Qdrant.

Luồng:
    Query → Embedding Model → Qdrant similarity search → top-k DocumentChunks

Tính năng mới:
    • Distance threshold: bỏ chunks có khoảng cách quá xa (không liên quan, dùng Euclid)
    • Distance score: gắn vào DocumentChunk để đánh giá chất lượng

Dùng bởi: Orchestrator (khi Brain quyết định intent=use_rag) và ReAct loop.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.models.schemas import DocumentChunk, RetrievalResult
from app.rag.ingest import DocumentIngestor


class RAGRetriever:
    """Retrieve relevant context từ Qdrant vector store."""

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None
        self._using_remote = bool(settings.qdrant_url.strip())
        self._qdrant: QdrantClient | None = None
        self._collection = settings.qdrant_collection
        self._initialized = False

    def _create_qdrant_client(self) -> QdrantClient:
        if self._using_remote:
            return QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
            )

        local_path = Path(settings.qdrant_path)
        local_path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(local_path))

    def _client(self) -> QdrantClient:
        if self._qdrant is None:
            self._qdrant = self._create_qdrant_client()
        return self._qdrant

    def _ensure_qdrant_ready(self) -> None:
        """
        Kiểm tra kết nối Qdrant.
        Nếu remote không reachable thì tự động fallback sang embedded local Qdrant.
        """
        try:
            self._client().get_collections()
        except Exception as e:
            if self._using_remote:
                logger.warning(
                    f"Qdrant remote không kết nối được ({e}). "
                    f"Fallback sang embedded local tại {settings.qdrant_path}."
                )
                self._using_remote = False
                self._qdrant = self._create_qdrant_client()
                self._client().get_collections()
            else:
                raise

    def initialize(self, force_reingest: bool = False) -> None:
        """
        Khởi tạo retriever:
          1. Đảm bảo embedding model sẵn sàng
          2. Kiểm tra collection Qdrant
          3. Nếu collection chưa có dữ liệu (hoặc force) → ingest từ data/docs/
        """
        if self._initialized and not force_reingest:
            return

        self._ensure_qdrant_ready()

        # Load model
        if self._model is None:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self._model = SentenceTransformer(settings.embedding_model)

        collection_exists = self._client().collection_exists(self._collection)
        points_count = 0
        if collection_exists:
            points_count = self._client().count(
                collection_name=self._collection,
                exact=True,
            ).count

        if force_reingest or not collection_exists or points_count == 0:
            logger.info("Ingesting documents for Qdrant RAG...")
            ingestor = DocumentIngestor(qdrant_client=self._client())
            total = ingestor.ingest(recreate_collection=True)
            logger.info(f"Qdrant ingest completed: {total} chunks")

        self._initialized = True
        logger.info(f"RAG Retriever initialized on Qdrant collection '{self._collection}'")

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

        self._ensure_qdrant_ready()

        collection_exists = self._client().collection_exists(self._collection)
        if not collection_exists:
            logger.warning("Qdrant collection not found — no documents ingested")
            return RetrievalResult(query=query, chunks=[])

        points_count = self._client().count(
            collection_name=self._collection,
            exact=True,
        ).count
        if points_count == 0:
            logger.warning("Qdrant collection is empty — no documents ingested")
            return RetrievalResult(query=query, chunks=[])

        k = min(top_k or settings.rag_top_k, points_count)
        threshold = distance_threshold or settings.rag_distance_threshold

        # Embed query
        if self._model is None:
            self._model = SentenceTransformer(settings.embedding_model)

        query_vec = self._model.encode([query], normalize_embeddings=False)[0].tolist()

        # Search (Euclid distance score)
        hits = self._client().search(
            collection_name=self._collection,
            query_vector=query_vec,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )

        # Map hits → chunk, áp dụng distance threshold
        result_chunks: list[DocumentChunk] = []
        for i, hit in enumerate(hits):
            dist = float(hit.score)

            # Bỏ chunks quá xa (không liên quan)
            if dist > threshold:
                logger.debug(
                    f"  [RAG] SKIP rank={i+1} dist={dist:.4f} > "
                    f"threshold={threshold} — not relevant"
                )
                continue

            payload = hit.payload or {}
            enriched_chunk = DocumentChunk(
                text=str(payload.get("text", "")),
                source=str(payload.get("source", "unknown")),
                chunk_index=int(payload.get("chunk_index", 0)),
                distance=dist,
            )
            result_chunks.append(enriched_chunk)
            logger.debug(
                f"  [RAG] rank={i+1} dist={dist:.4f} "
                f"source={enriched_chunk.source} chunk={enriched_chunk.chunk_index}"
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
