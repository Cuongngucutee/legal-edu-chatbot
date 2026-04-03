"""
Sparse Retrieval – BM25 index for keyword-based search.

Uses rank_bm25 with simple Vietnamese tokenization.
Supports metadata filtering by source, article_number, etc.
"""
from __future__ import annotations

import re
import logging
from typing import Optional

from rank_bm25 import BM25Okapi

from src.config import SPARSE_TOP_K
from src.models import ChunkRecord, RetrievedChunk

logger = logging.getLogger(__name__)


def _tokenize_vi(text: str) -> list[str]:
    """Simple Vietnamese tokenizer: lowercase, split on whitespace/punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]", " ", text)
    tokens = text.split()
    # Remove very short tokens
    return [t for t in tokens if len(t) > 1]


class SparseRetrieval:
    """BM25-based sparse retrieval index."""

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._chunks: list[ChunkRecord] = []
        self._tokenized_corpus: list[list[str]] = []

    # ── Indexing ───────────────────────────────────────────────────

    def index_chunks(self, chunks: list[ChunkRecord]) -> None:
        """Build BM25 index from chunks."""
        self._chunks = chunks
        self._tokenized_corpus = [_tokenize_vi(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"BM25 index built with {len(chunks)} documents")

    # ── Search ─────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = SPARSE_TOP_K,
        source_filter: Optional[str] = None,
        article_filter: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """BM25 keyword search with optional metadata filtering."""
        if self._bm25 is None:
            logger.warning("BM25 index not built yet")
            return []

        query_tokens = _tokenize_vi(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Pair scores with chunks and apply filters
        scored_chunks = []
        for idx, (score, chunk) in enumerate(zip(scores, self._chunks)):
            if score <= 0:
                continue
            if source_filter and chunk.source != source_filter:
                continue
            if article_filter and chunk.article_number != article_filter:
                continue
            scored_chunks.append((score, chunk))

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, chunk in scored_chunks[:top_k]:
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(score),
                    source_method="sparse",
                )
            )

        return results
