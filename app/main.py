"""
LawEdu AI — FastAPI Application Entry Point.
Initializes all components and starts the server.

Architecture (from diagram):
CLIENT → API GATEWAY → CHAT ORCHESTRATOR → QUERY REWRITE → RAG PIPELINE
    → HYBRID SEARCH (BM25 + Vector) → RE-RANKER → CONTEXT BUILDER
    → LLM GENERATOR (320B API) → TOOL CALLING → RESPONSE (Cache)
"""
import os
import sys
import logging

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

# Setup paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from app.config import settings
from app.logging_monitor.logger import setup_logging, logger, metrics

setup_logging()

# ══════════════════════════════════════════════════════════════════
# FastAPI App
# ══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="LawEdu AI",
    version="5.0",
    description="Legal Education Chatbot — 320B LLM + RAG Pipeline",
)

# ── Middleware ──
from app.gateway.auth import AuthMiddleware
from app.gateway.rate_limiter import RateLimiterMiddleware

app.add_middleware(RateLimiterMiddleware, requests_per_minute=settings.api.rate_limit_per_minute)
app.add_middleware(AuthMiddleware, api_key=settings.api.api_key)

# ── Router ──
from app.gateway.router import router, init_router
app.include_router(router)


# ── Global Pipeline State ──
PIPELINE = None


def init_pipeline():
    """Initialize the full pipeline: Index → Retriever → LLM → Pipeline."""
    global PIPELINE

    if PIPELINE is not None:
        return

    logger.info("🚀 Initializing LawEdu AI Pipeline v5.0...")
    logger.info(f"   LLM: {settings.llm.model} @ {settings.llm.api_base}")

    # 1. Load BookIndex (KG + FAISS + BM25)
    logger.info("📚 Loading BookIndex...")
    from app.index.book_index import BookIndex
    index = BookIndex(settings.data.data_dir, settings.data.kg_path)
    index.load_index()
    logger.info(f"   Nodes: {index.graph.number_of_nodes()}, Docs: {len(index.doc_registry)}")

    # 2. Load BookRAGRetriever (CrossEncoder)
    logger.info("🔍 Loading Retriever + CrossEncoder...")
    from app.rag.hybrid_search import BookRAGRetriever
    retriever = BookRAGRetriever(index)

    # Initialize Agentic LLM (320B)
    logger.info("🤖 Initializing 320B LLM Client (Agentic)...")
    from app.llm.client import LLMClient
    agentic_llm = LLMClient(
        api_base=settings.llm.api_base,
        api_key=settings.llm.api_key,
        model=settings.llm.model,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
    )
    logger.info(f"   Model: {agentic_llm.model}")
    
    # Initialize Generator LLM (7B Local)
    logger.info("🤖 Initializing 7B LLM Client (Generator)...")
    generator_llm = LLMClient(
        api_base=settings.generator_llm.api_base,
        api_key=settings.generator_llm.api_key,
        model=settings.generator_llm.model,
        temperature=settings.generator_llm.temperature,
        max_tokens=settings.generator_llm.max_tokens,
    )
    logger.info(f"   Model: {generator_llm.model}")

    # 4. Build Pipeline
    logger.info("⚡ Building LawEduPipeline...")
    from app.rag.pipeline import LawEduPipeline
    PIPELINE = LawEduPipeline(retriever, agentic_llm, generator_llm)

    # 5. Initialize Cache
    logger.info("📦 Initializing Cache...")
    from app.cache.redis_cache import QueryCache
    cache = QueryCache(redis_url=settings.redis.url, ttl=settings.redis.ttl)

    # 6. Initialize Router
    init_router(PIPELINE, cache, metrics)

    logger.info("✅ Pipeline ready! All components initialized.")


@app.on_event("startup")
async def startup():
    init_pipeline()


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    static_path = os.path.join(ROOT_DIR, "static", "index.html")
    if os.path.exists(static_path):
        return FileResponse(static_path)
    return HTMLResponse("<h1>LawEdu AI v5.0</h1><p>API ready at /api/health</p>")


# Mount static files
static_dir = os.path.join(ROOT_DIR, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=True,
    )
