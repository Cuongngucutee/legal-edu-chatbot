from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List

from app.core.config import AppConfig
from app.core.logger import log_retrieval_trace, setup_logging
from app.rag.embeddings import EmbeddingService
from app.rag.schemas import RetrievalHit
from app.rag.vector_store import FaissVectorStore


@dataclass(frozen=True)
class RagResponse:
    question: str
    hits: List[RetrievalHit]
    elapsed_ms: int


class RagService:
    def __init__(self, config: AppConfig):
        config.ensure_dirs()
        self.config = config
        self.logger = setup_logging(config.app_log_file)

        self.embedding_service = EmbeddingService(config.embedding_model)
        self.vector_store = FaissVectorStore(config.index_file, config.metadata_file)
        self.vector_store.load()

        self.logger.info("RAG service initialized with index=%s", config.index_file)

    def retrieve(self, question: str, top_k: int | None = None) -> List[RetrievalHit]:
        k = top_k or self.config.retrieval_top_k
        query_embedding = self.embedding_service.embed([question], normalize=True)
        hits = self.vector_store.search(query_embedding, top_k=k)
        self.logger.info("Retrieved %s chunks for question", len(hits))
        return hits

    def ask(self, question: str) -> RagResponse:
        start = time.time()
        hits = self.retrieve(question)
        elapsed_ms = int((time.time() - start) * 1000)
        self._write_trace(question=question, hits=hits, elapsed_ms=elapsed_ms)

        return RagResponse(question=question, hits=hits, elapsed_ms=elapsed_ms)

    def _write_trace(self, question: str, hits: List[RetrievalHit], elapsed_ms: int) -> None:
        payload: Dict[str, object] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "retrieval_top_k": self.config.retrieval_top_k,
            "elapsed_ms": elapsed_ms,
            "hits": [asdict(hit) for hit in hits],
        }
        log_retrieval_trace(self.config.retrieval_log_file, payload)
