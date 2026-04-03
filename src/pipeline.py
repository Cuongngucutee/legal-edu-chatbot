"""
Pipeline Orchestrator

End-to-end pipeline that connects all components:
    Query → Processing → Retrieval → Reasoning → Generation → Formatting → Answer
"""
from __future__ import annotations

import logging
from typing import Optional

from src.config import MAX_CONTEXT_CHUNKS, MAX_RETRIES
from src.models import ParsedQuery, PipelineAnswer, RetrievedChunk
from src.query.processor import QueryProcessor
from src.retrieval.vector_store import VectorStore
from src.retrieval.sparse_retrieval import SparseRetrieval
from src.retrieval.hybrid_retrieval import HybridRetrieval
from src.bookindex.tree_builder import HierarchicalTree
from src.bookindex.kg_builder import KnowledgeGraph
from src.reasoning.reasoning_engine import ReasoningEngine
from src.generation.answer_generator import AnswerGenerator
from src.generation.formatter import Formatter

logger = logging.getLogger(__name__)


class BookRAGPipeline:
    """Main pipeline orchestrator for the BookRAG system."""

    def __init__(
        self,
        tree: HierarchicalTree,
        kg: KnowledgeGraph,
        vector_store: VectorStore,
        sparse_index: SparseRetrieval,
    ):
        self.tree = tree
        self.kg = kg
        self.vector_store = vector_store
        self.sparse_index = sparse_index

        # Initialize components
        self.query_processor = QueryProcessor()
        self.hybrid_retrieval = HybridRetrieval(
            vector_store=vector_store,
            sparse_index=sparse_index,
            kg=kg,
            tree=tree,
        )
        self.reasoning_engine = ReasoningEngine()
        self.answer_generator = AnswerGenerator()
        self.formatter = Formatter()

    def query(
        self,
        question: str,
        top_k: int = MAX_CONTEXT_CHUNKS,
        show_reasoning: bool = False,
    ) -> str:
        """
        Process a question end-to-end and return formatted Markdown answer.

        Args:
            question: User's question in Vietnamese
            top_k: Number of chunks to retrieve
            show_reasoning: Whether to include reasoning trace in output

        Returns:
            Formatted Markdown answer string
        """
        # 1. Query Processing
        logger.info(f"═══ Processing query: {question[:80]}...")
        parsed_query = self.query_processor.process(question)

        # 2. Retrieval
        logger.info("═══ Retrieving relevant chunks...")
        retrieved_chunks = self.hybrid_retrieval.retrieve(
            parsed_query=parsed_query,
            top_k=top_k,
        )
        logger.info(f"    Retrieved {len(retrieved_chunks)} chunks")

        # 3. Reasoning
        logger.info("═══ Running chain-of-thought reasoning...")
        reasoning = self.reasoning_engine.reason(
            parsed_query=parsed_query,
            retrieved_chunks=retrieved_chunks,
        )

        # 3.5 Retry loop if reasoning says more info needed
        if reasoning.needs_more_info and reasoning.additional_queries:
            for retry in range(MAX_RETRIES):
                logger.info(f"    Retry {retry+1}: fetching additional info...")
                for extra_q in reasoning.additional_queries[:2]:
                    extra_parsed = self.query_processor.process(extra_q)
                    extra_chunks = self.hybrid_retrieval.retrieve(
                        parsed_query=extra_parsed, top_k=5,
                    )
                    # Merge (avoid duplicates)
                    existing_ids = {rc.chunk.chunk_id for rc in retrieved_chunks}
                    for ec in extra_chunks:
                        if ec.chunk.chunk_id not in existing_ids:
                            retrieved_chunks.append(ec)
                            existing_ids.add(ec.chunk.chunk_id)

                # Re-run reasoning with expanded evidence
                reasoning = self.reasoning_engine.reason(
                    parsed_query=parsed_query,
                    retrieved_chunks=retrieved_chunks[:top_k],
                )
                if not reasoning.needs_more_info:
                    break

        # 4. Answer Generation
        logger.info("═══ Generating answer...")
        answer = self.answer_generator.generate(
            parsed_query=parsed_query,
            retrieved_chunks=retrieved_chunks[:top_k],
            reasoning=reasoning,
        )

        # 5. Formatting
        formatted = self.formatter.format(answer, show_reasoning=show_reasoning)

        logger.info("═══ Done!")
        return formatted

    def query_raw(
        self,
        question: str,
        top_k: int = MAX_CONTEXT_CHUNKS,
    ) -> PipelineAnswer:
        """
        Process a question and return the raw PipelineAnswer object.
        Useful for programmatic access.
        """
        parsed_query = self.query_processor.process(question)

        retrieved_chunks = self.hybrid_retrieval.retrieve(
            parsed_query=parsed_query,
            top_k=top_k,
        )

        reasoning = self.reasoning_engine.reason(
            parsed_query=parsed_query,
            retrieved_chunks=retrieved_chunks,
        )

        answer = self.answer_generator.generate(
            parsed_query=parsed_query,
            retrieved_chunks=retrieved_chunks[:top_k],
            reasoning=reasoning,
        )
        answer.retrieved_chunks = retrieved_chunks[:top_k]

        return answer

    def get_tree_info(self) -> str:
        """Return tree structure info for debugging."""
        return self.tree.print_tree(max_depth=3)

    def get_stats(self) -> dict:
        """Return pipeline statistics."""
        return {
            "tree_nodes": len(self.tree.nodes),
            "total_chunks": len(self.tree.chunks),
            "kg_nodes": self.kg.graph.number_of_nodes(),
            "kg_edges": self.kg.graph.number_of_edges(),
        }
