"""
app/rag/ingest.py
──────────────────
Pipeline nạp tài liệu vào vector store:
  1. Đọc file .txt / .md / .pdf từ thư mục data/docs/
  2. Chia nhỏ thành chunks (character-based với overlap)
  3. Embed bằng sentence-transformers
    4. Upsert vào Qdrant collection

Chạy 1 lần khi khởi tạo hoặc khi có tài liệu mới.
Dễ thay Qdrant → ChromaDB / Pinecone.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.models.schemas import DocumentChunk


class DocumentIngestor:
    """Đọc docs → chunk → embed → upsert Qdrant."""

    def __init__(
        self,
        embedding_model_name: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        qdrant_client: QdrantClient | None = None,
    ) -> None:
        self._model_name = embedding_model_name or settings.embedding_model
        self._chunk_size = chunk_size or settings.chunk_size
        self._chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._collection = settings.qdrant_collection
        self._model: SentenceTransformer | None = None
        self._external_client = qdrant_client is not None
        if qdrant_client is not None:
            self._qdrant = qdrant_client
            self._using_remote = bool(settings.qdrant_url.strip())
        else:
            self._using_remote = bool(settings.qdrant_url.strip())
            self._qdrant = self._create_qdrant_client()

    def _create_qdrant_client(self) -> QdrantClient:
        if self._using_remote:
            return QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
            )

        local_path = Path(settings.qdrant_path)
        local_path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(local_path))

    def _ensure_qdrant_ready(self) -> None:
        """
        Kiểm tra kết nối Qdrant.
        Nếu remote không reachable thì tự động fallback sang embedded local Qdrant.
        """
        try:
            self._qdrant.get_collections()
        except Exception as e:
            if self._external_client:
                raise
            if self._using_remote:
                logger.warning(
                    f"Qdrant remote không kết nối được ({e}). "
                    f"Fallback sang embedded local tại {settings.qdrant_path}."
                )
                self._using_remote = False
                self._qdrant = self._create_qdrant_client()
                self._qdrant.get_collections()
            else:
                raise

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

    # ── Bước 3 & 4: Embed + Upsert Qdrant ────────────────

    def _ensure_collection(self, recreate: bool = False) -> None:
        self._ensure_qdrant_ready()
        model = self._get_model()
        dim = model.get_sentence_embedding_dimension()

        exists = self._qdrant.collection_exists(self._collection)
        if exists and recreate:
            self._qdrant.delete_collection(self._collection)
            exists = False

        if not exists:
            self._qdrant.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=dim, distance=Distance.EUCLID),
            )
            logger.info(
                f"Created Qdrant collection '{self._collection}' (dim={dim}, distance=euclid)"
            )

    def ingest(self, docs_dir: str | None = None, recreate_collection: bool = True) -> int:
        """
        Pipeline hoàn chỉnh: load → chunk → embed → Qdrant upsert.

        Returns:
            Số lượng chunks đã upsert.
        """
        documents = self._load_documents(docs_dir)
        if not documents:
            logger.warning("No documents to ingest!")
            return 0

        # Chunk tất cả documents
        all_chunks: list[DocumentChunk] = []
        for content, source in documents:
            chunks = self._chunk_text(content, source)
            all_chunks.extend(chunks)
        logger.info(f"Total chunks: {len(all_chunks)}")

        # Embed
        model = self._get_model()
        texts = [c.text for c in all_chunks]
        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=False)

        self._ensure_collection(recreate=recreate_collection)
        self._ensure_qdrant_ready()

        batch_size = 64
        total = len(all_chunks)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            points: list[PointStruct] = []

            for i in range(start, end):
                chunk = all_chunks[i]
                vec = embeddings[i].tolist()
                points.append(
                    PointStruct(
                        id=i,
                        vector=vec,
                        payload={
                            "text": chunk.text,
                            "source": chunk.source,
                            "chunk_index": chunk.chunk_index,
                        },
                    )
                )

            self._qdrant.upsert(collection_name=self._collection, points=points)

        logger.info(
            f"Upserted {total} chunks into Qdrant collection '{self._collection}'"
        )
        return total
