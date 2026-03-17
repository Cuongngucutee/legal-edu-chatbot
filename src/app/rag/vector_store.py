from __future__ import annotations

import json
from pathlib import Path
from typing import List

import faiss
import numpy as np

from app.rag.schemas import LegalChunk, RetrievalHit


class FaissVectorStore:
    def __init__(self, index_file: Path, metadata_file: Path):
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.index: faiss.Index | None = None
        self.chunks: List[LegalChunk] = []

    def build(self, embeddings: np.ndarray, chunks: List[LegalChunk]) -> None:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix")
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Embeddings rows must match chunk count")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_file))

        serialized = [
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "article_number": c.article_number,
                "article_title": c.article_title,
                "chapter": c.chapter,
                "section": c.section,
                "text": c.text,
            }
            for c in chunks
        ]
        with self.metadata_file.open("w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False, indent=2)

        self.index = index
        self.chunks = chunks

    def load(self) -> None:
        if not self.index_file.exists() or not self.metadata_file.exists():
            raise FileNotFoundError("FAISS index or metadata file is missing")

        self.index = faiss.read_index(str(self.index_file))
        with self.metadata_file.open("r", encoding="utf-8") as f:
            records = json.load(f)

        self.chunks = [
            LegalChunk(
                chunk_id=rec["chunk_id"],
                source=rec["source"],
                article_number=rec["article_number"],
                article_title=rec["article_title"],
                chapter=rec["chapter"],
                section=rec["section"],
                text=rec["text"],
            )
            for rec in records
        ]

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievalHit]:
        if self.index is None:
            raise RuntimeError("FAISS index is not loaded")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding.astype("float32"), top_k)
        hits: List[RetrievalHit] = []

        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.chunks):
                continue
            hits.append(RetrievalHit(rank=rank, score=float(score), chunk=self.chunks[idx]))
        return hits
