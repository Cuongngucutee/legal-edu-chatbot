"""
Dense Retrieval – Vector Store using Qdrant (in-memory).

Encodes chunks with sentence-transformers and indexes them in Qdrant
for semantic similarity search.
"""
from __future__ import annotations

import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, EMBEDDING_DIM, QDRANT_COLLECTION, DENSE_TOP_K
from src.models import ChunkRecord, RetrievedChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Qdrant-backed dense vector store for semantic retrieval."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        logger.info(f"Loading embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.client = QdrantClient(":memory:")  # in-memory for simplicity
        self._chunk_map: dict[str, ChunkRecord] = {}
        self._collection_ready = False

    # ── Indexing ───────────────────────────────────────────────────

    def index_chunks(self, chunks: list[ChunkRecord], batch_size: int = 64) -> None:
        """Encode and index all chunks."""
        if not chunks:
            logger.warning("No chunks to index")
            return

        # Create collection
        self.client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )

        # Encode in batches
        texts = [c.text for c in chunks]
        logger.info(f"Encoding {len(texts)} chunks...")

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self.encoder.encode(batch, show_progress_bar=False)
            all_embeddings.extend(embeddings)

        # Upsert to Qdrant
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
            self._chunk_map[chunk.chunk_id] = chunk
            points.append(
                PointStruct(
                    id=idx,
                    vector=embedding.tolist(),
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "node_id": chunk.node_id,
                        "source": chunk.source,
                        "chapter": chunk.chapter,
                        "article_number": chunk.article_number,
                        "article_title": chunk.article_title,
                        "clause_number": chunk.clause_number,
                        "point_label": chunk.point_label,
                        "split_level": chunk.split_level,
                    },
                )
            )

        # Upsert in batches
        for i in range(0, len(points), 100):
            batch = points[i : i + 100]
            self.client.upsert(collection_name=QDRANT_COLLECTION, points=batch)

        self._collection_ready = True
        logger.info(f"Indexed {len(points)} vectors in Qdrant")

    # ── Search ─────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
        source_filter: Optional[str] = None,
        article_filter: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """Semantic similarity search."""
        if not self._collection_ready:
            logger.warning("Vector store not indexed yet")
            return []

        query_vector = self.encoder.encode(query).tolist()

        # Build filter
        conditions = []
        if source_filter:
            conditions.append(
                FieldCondition(key="source", match=MatchValue(value=source_filter))
            )
        if article_filter:
            conditions.append(
                FieldCondition(
                    key="article_number", match=MatchValue(value=article_filter)
                )
            )

        search_filter = Filter(must=conditions) if conditions else None

        results = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=top_k,
            query_filter=search_filter,
        )

        retrieved = []
        for hit in results.points:
            chunk_id = hit.payload.get("chunk_id", "")
            chunk = self._chunk_map.get(chunk_id)
            if chunk:
                retrieved.append(
                    RetrievedChunk(
                        chunk=chunk,
                        score=hit.score,
                        source_method="dense",
                    )
                )

        return retrieved

    def encode_query(self, query: str) -> list[float]:
        """Encode a query string to a vector."""
        return self.encoder.encode(query).tolist()
