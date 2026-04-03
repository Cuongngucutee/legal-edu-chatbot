"""
Indexing Script – Build all indexes from raw JSON data.

Run once to:
1. Build hierarchical tree from data/final/
2. Build knowledge graph
3. Build dense vector index (Qdrant)
4. Build sparse BM25 index
5. Return initialized pipeline ready for queries
"""
from __future__ import annotations

import logging
import time

from src.config import DATA_DIR
from src.bookindex.tree_builder import HierarchicalTree
from src.bookindex.kg_builder import KnowledgeGraph
from src.retrieval.vector_store import VectorStore
from src.retrieval.sparse_retrieval import SparseRetrieval
from src.pipeline import BookRAGPipeline

logger = logging.getLogger(__name__)


def build_pipeline(verbose: bool = True) -> BookRAGPipeline:
    """
    Build the full BookRAG pipeline from raw data.

    Returns:
        Initialized BookRAGPipeline ready for queries.
    """
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s │ %(levelname)-5s │ %(message)s",
            datefmt="%H:%M:%S",
        )

    total_start = time.time()

    # ── Step 1: Build Hierarchical Tree ────────────────────────────
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║  Step 1/4: Building Hierarchical Tree        ║")
    logger.info("╚══════════════════════════════════════════════╝")
    t0 = time.time()

    tree = HierarchicalTree()
    tree.build_from_directory(DATA_DIR)

    logger.info(f"  ✓ Tree: {len(tree.nodes)} nodes, {len(tree.chunks)} chunks")
    logger.info(f"  ⏱ {time.time() - t0:.1f}s")

    # Print tree preview
    preview = tree.print_tree(max_depth=2)
    for line in preview.split("\n")[:15]:
        logger.info(f"  {line}")

    # ── Step 2: Build Knowledge Graph ──────────────────────────────
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║  Step 2/4: Building Knowledge Graph          ║")
    logger.info("╚══════════════════════════════════════════════╝")
    t0 = time.time()

    kg = KnowledgeGraph()
    kg.build_from_tree(tree)

    logger.info(
        f"  ✓ KG: {kg.graph.number_of_nodes()} nodes, "
        f"{kg.graph.number_of_edges()} edges"
    )
    logger.info(f"  ⏱ {time.time() - t0:.1f}s")
    
    # Export KG to physical file
    import os
    os.makedirs("indexes", exist_ok=True)
    kg.export_graphml("indexes/kg.graphml")

    # ── Step 3: Build Vector Store ─────────────────────────────────
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║  Step 3/4: Building Vector Store (Dense)     ║")
    logger.info("╚══════════════════════════════════════════════╝")
    t0 = time.time()

    vector_store = VectorStore()
    chunks = tree.get_leaf_chunks()
    vector_store.index_chunks(chunks)

    logger.info(f"  ✓ Indexed {len(chunks)} vectors")
    logger.info(f"  ⏱ {time.time() - t0:.1f}s")

    # ── Step 4: Build BM25 Index ───────────────────────────────────
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║  Step 4/4: Building BM25 Index (Sparse)      ║")
    logger.info("╚══════════════════════════════════════════════╝")
    t0 = time.time()

    sparse_index = SparseRetrieval()
    sparse_index.index_chunks(chunks)

    logger.info(f"  ✓ BM25 index: {len(chunks)} documents")
    logger.info(f"  ⏱ {time.time() - t0:.1f}s")

    # ── Create Pipeline ────────────────────────────────────────────
    pipeline = BookRAGPipeline(
        tree=tree,
        kg=kg,
        vector_store=vector_store,
        sparse_index=sparse_index,
    )

    total_time = time.time() - total_start
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║  ✅ Pipeline ready!                          ║")
    logger.info(f"║  Total time: {total_time:.1f}s                          ║")
    logger.info("╚══════════════════════════════════════════════╝")

    stats = pipeline.get_stats()
    logger.info(f"  📊 Stats: {stats}")

    return pipeline


if __name__ == "__main__":
    pipeline = build_pipeline()

    # Quick test
    print("\n" + "═" * 60)
    print("  Quick test: 'Giáo dục chính quy là gì?'")
    print("═" * 60 + "\n")

    result = pipeline.query(
        "Giáo dục chính quy là gì?",
        show_reasoning=True,
    )
    print(result)
