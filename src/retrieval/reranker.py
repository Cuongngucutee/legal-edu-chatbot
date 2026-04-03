"""
Cross-Encoder Reranking Module

Uses a cross-encoder model to rerank retrieval results
for higher precision. Falls back to heuristic scoring
if the model is unavailable.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.models import RetrievedChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks retrieval results using a cross-encoder model.
    Falls back to heuristic scoring if model unavailable.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model = None
        self._model_name = model_name
        self._load_attempted = False

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            logger.info(f"Loading cross-encoder: {self._model_name}")
            self._model = CrossEncoder(self._model_name)
            logger.info("Cross-encoder loaded successfully")
        except Exception as e:
            logger.warning(f"Cross-encoder unavailable: {e}. Using heuristic reranking.")
            self._model = None

    def rerank(
        self,
        query: str,
        results: list[RetrievedChunk],
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """Rerank results using cross-encoder or heuristic fallback."""
        if not results:
            return []

        self._load_model()

        if self._model is not None:
            return self._cross_encoder_rerank(query, results, top_k)
        else:
            return self._heuristic_rerank(query, results, top_k)

    def _cross_encoder_rerank(
        self,
        query: str,
        results: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Rerank using cross-encoder model scores."""
        pairs = [(query, r.chunk.text) for r in results]
        scores = self._model.predict(pairs)

        # Combine with original score (weighted)
        for r, ce_score in zip(results, scores):
            # 70% cross-encoder, 30% original RRF
            r.score = 0.7 * float(ce_score) + 0.3 * r.score

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _heuristic_rerank(
        self,
        query: str,
        results: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Heuristic reranking based on text overlap and structure."""
        query_terms = set(query.lower().split())

        for r in results:
            text_lower = r.chunk.text.lower()
            title_lower = r.chunk.article_title.lower()

            boost = 0.0

            # Term overlap boost
            text_terms = set(text_lower.split())
            overlap = len(query_terms & text_terms) / max(len(query_terms), 1)
            boost += overlap * 0.3

            # Title match boost
            title_terms = set(title_lower.split())
            title_overlap = len(query_terms & title_terms) / max(len(query_terms), 1)
            boost += title_overlap * 0.2

            # "Giải thích từ ngữ" boost for definition queries
            if "là gì" in query.lower() and "giải thích từ ngữ" in title_lower:
                boost += 0.3

            # Exact phrase match boost
            for term in query_terms:
                if len(term) > 3 and term in text_lower:
                    boost += 0.05

            # Multi-source boost (retrieved from multiple methods)
            if "+" in r.source_method:
                boost += 0.1 * r.source_method.count("+")

            r.score += boost

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
