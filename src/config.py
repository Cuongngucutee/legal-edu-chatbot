"""
Configuration and constants for the BookRAG pipeline.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "final"
INDEX_DIR = PROJECT_ROOT / "indexes"
INDEX_DIR.mkdir(exist_ok=True)

# ── LLM ────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ── Embedding ──────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
EMBEDDING_DIM = 384  # dimension for MiniLM-L12-v2

# ── Retrieval ──────────────────────────────────────────────────────
DENSE_TOP_K = 20          # candidates from dense retrieval
SPARSE_TOP_K = 20         # candidates from BM25
RERANK_TOP_K = 10         # final results after reranking
RRF_K = 60                # RRF constant

# ── Qdrant ─────────────────────────────────────────────────────────
QDRANT_COLLECTION = "bookrag_legal"

# ── Reasoning ──────────────────────────────────────────────────────
MAX_CONTEXT_CHUNKS = 8    # max chunks fed to LLM for reasoning
MAX_RETRIES = 1           # max retrieval retries in reasoning loop
