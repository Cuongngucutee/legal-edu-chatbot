"""
Hybrid Retrieval & Reranking

Combines dense (vector) + sparse (BM25) + KG retrieval using
Reciprocal Rank Fusion (RRF), then optionally reranks with a
cross-encoder-like heuristic.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from src.config import RRF_K, RERANK_TOP_K, DENSE_TOP_K, SPARSE_TOP_K
from src.models import (
    ParsedQuery, RetrievedChunk, ChunkRecord, QueryIntent,
)
from src.retrieval.vector_store import VectorStore
from src.retrieval.sparse_retrieval import SparseRetrieval
from src.retrieval.reranker import CrossEncoderReranker
from src.bookindex.kg_builder import KnowledgeGraph
from src.bookindex.tree_builder import HierarchicalTree

logger = logging.getLogger(__name__)


class HybridRetrieval:
    """Combines dense, sparse, and KG retrieval with fusion & reranking."""

    def __init__(
        self,
        vector_store: VectorStore,
        sparse_index: SparseRetrieval,
        kg: KnowledgeGraph,
        tree: HierarchicalTree,
    ):
        self.vector_store = vector_store
        self.sparse_index = sparse_index
        self.kg = kg
        self.tree = tree
        self.reranker = CrossEncoderReranker()

    def retrieve(
        self,
        parsed_query: ParsedQuery,
        top_k: int = RERANK_TOP_K,
    ) -> list[RetrievedChunk]:
        """Full hybrid retrieval pipeline."""

        # ── Determine filters from parsed query ───────────────────
        source_filter = None
        article_filter = None

        if parsed_query.law_references:
            # Try to match a source name
            source_filter = self._match_source(parsed_query.law_references[0])

        if parsed_query.article_references:
            # Extract article number
            import re
            m = re.search(r"(\d+[a-z]?)", parsed_query.article_references[0])
            if m:
                article_filter = m.group(1)

        query_text = parsed_query.normalized

        # ── 1. Dense retrieval ────────────────────────────────────
        dense_results = self.vector_store.search(
            query=query_text,
            top_k=DENSE_TOP_K,
            source_filter=source_filter,
            article_filter=article_filter,
        )

        # ── 2. Sparse retrieval ───────────────────────────────────
        sparse_results = self.sparse_index.search(
            query=query_text,
            top_k=SPARSE_TOP_K,
            source_filter=source_filter,
            article_filter=article_filter,
        )

        # ── 3. KG retrieval ───────────────────────────────────────
        kg_results = self._kg_retrieve(parsed_query)

        # ── 4. Reciprocal Rank Fusion ─────────────────────────────
        fused = self._rrf_fusion(dense_results, sparse_results, kg_results)

        # ── 5. Intent-aware boosting ──────────────────────────────
        fused = self._intent_boost(fused, parsed_query)

        # ── 6. Deduplication ──────────────────────────────────────
        fused = self._deduplicate(fused)

        # ── 7. Cross-encoder reranking ────────────────────────────
        fused = self.reranker.rerank(query_text, fused, top_k=top_k * 2)

        # ── 8. Tree-aware context expansion ───────────────────────
        fused = self._expand_context(fused, parsed_query)

        # Return top_k
        return fused[:top_k]

    # ── KG Retrieval ───────────────────────────────────────────────

    def _kg_retrieve(self, parsed_query: ParsedQuery) -> list[RetrievedChunk]:
        """Retrieve chunks via knowledge graph traversal."""
        results = []
        chunk_map = {c.chunk_id: c for c in self.tree.get_leaf_chunks()}

        # Search by concepts
        for concept in parsed_query.key_concepts:
            chunk_ids = self.kg.find_related_chunks(concept, max_hops=2, limit=10)
            for cid in chunk_ids:
                if cid in chunk_map:
                    results.append(
                        RetrievedChunk(
                            chunk=chunk_map[cid],
                            score=1.0,  # normalized later by RRF
                            source_method="kg",
                        )
                    )

        # Search by entity names from query
        for entity in parsed_query.key_concepts[:3]:
            node_ids = self.kg.find_entity_articles(entity)
            for node_id in node_ids[:5]:
                # Find chunks with matching node_id
                for chunk in self.tree.get_leaf_chunks():
                    if chunk.node_id == node_id and chunk.chunk_id not in [
                        r.chunk.chunk_id for r in results
                    ]:
                        results.append(
                            RetrievedChunk(
                                chunk=chunk,
                                score=0.8,
                                source_method="kg",
                            )
                        )

        return results[:SPARSE_TOP_K]

    # ── RRF Fusion ─────────────────────────────────────────────────

    def _rrf_fusion(
        self,
        dense: list[RetrievedChunk],
        sparse: list[RetrievedChunk],
        kg: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion across multiple result lists."""
        scores: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, RetrievedChunk] = {}
        sources: dict[str, list[str]] = defaultdict(list)

        for rank, result in enumerate(dense):
            cid = result.chunk.chunk_id
            scores[cid] += 1.0 / (RRF_K + rank + 1)
            chunk_map[cid] = result
            sources[cid].append("dense")

        for rank, result in enumerate(sparse):
            cid = result.chunk.chunk_id
            scores[cid] += 1.0 / (RRF_K + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = result
            sources[cid].append("sparse")

        for rank, result in enumerate(kg):
            cid = result.chunk.chunk_id
            scores[cid] += 0.8 / (RRF_K + rank + 1)  # slight discount for KG
            if cid not in chunk_map:
                chunk_map[cid] = result
            sources[cid].append("kg")

        # Sort by fused score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        fused = []
        for cid in sorted_ids:
            rc = chunk_map[cid]
            fused.append(
                RetrievedChunk(
                    chunk=rc.chunk,
                    score=scores[cid],
                    source_method="+".join(sources[cid]),
                )
            )

        return fused

    # ── Intent Boosting ────────────────────────────────────────────

    def _intent_boost(
        self, results: list[RetrievedChunk], parsed_query: ParsedQuery
    ) -> list[RetrievedChunk]:
        """Boost scores based on query intent."""
        for r in results:
            text_lower = r.chunk.text.lower()
            title_lower = r.chunk.article_title.lower()

            if parsed_query.intent == QueryIntent.DEFINITION:
                # Boost "Giải thích từ ngữ" articles
                if "giải thích từ ngữ" in title_lower:
                    r.score *= 1.5
                if " là " in text_lower and any(
                    c in text_lower for c in parsed_query.key_concepts
                ):
                    r.score *= 1.3

            elif parsed_query.intent == QueryIntent.PROCEDURE:
                if any(
                    w in text_lower
                    for w in ["thủ tục", "trình tự", "hồ sơ", "quy trình"]
                ):
                    r.score *= 1.3

            elif parsed_query.intent == QueryIntent.CONDITION:
                if any(
                    w in text_lower
                    for w in ["điều kiện", "tiêu chuẩn", "yêu cầu"]
                ):
                    r.score *= 1.3

        # Re-sort
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    # ── Deduplication ──────────────────────────────────────────────

    def _deduplicate(self, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Remove near-duplicate chunks (same article_full_text + same level)."""
        seen_texts: set[str] = set()
        deduped = []

        for r in results:
            # Use a fingerprint: source + article_number + clause + point
            fp = (
                f"{r.chunk.source}|{r.chunk.article_number}|"
                f"{r.chunk.clause_number}|{r.chunk.point_label}|"
                f"{r.chunk.split_level}"
            )
            if fp not in seen_texts:
                seen_texts.add(fp)
                deduped.append(r)

        return deduped

    # ── Context Expansion ──────────────────────────────────────────

    def _expand_context(
        self, results: list[RetrievedChunk], parsed_query: ParsedQuery
    ) -> list[RetrievedChunk]:
        """For comparison queries, ensure we have chunks from multiple sources."""
        if parsed_query.intent != QueryIntent.COMPARISON:
            return results

        # Ensure diversity across sources
        sources_seen: dict[str, int] = defaultdict(int)
        expanded = []
        deferred = []

        for r in results:
            src = r.chunk.source
            if sources_seen[src] < 5:
                expanded.append(r)
                sources_seen[src] += 1
            else:
                deferred.append(r)

        expanded.extend(deferred)
        return expanded

    # ── Helpers ─────────────────────────────────────────────────────

    def _match_source(self, law_ref: str) -> Optional[str]:
        """Try to match a law reference to an actual source name in the corpus."""
        law_ref_lower = law_ref.lower()

        for chunk in self.tree.get_leaf_chunks():
            if not chunk.source:
                continue
            src_lower = chunk.source.lower()

            # Direct match
            if law_ref_lower in src_lower or src_lower in law_ref_lower:
                return chunk.source

            # Number match: "Luật 08" matches "Luật 08/2012/QH13"
            import re
            nums = re.findall(r"\d+", law_ref)
            if nums and nums[0] in chunk.source:
                return chunk.source

        return None
