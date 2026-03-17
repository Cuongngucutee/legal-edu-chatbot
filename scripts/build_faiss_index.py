from __future__ import annotations

from app.core.config import AppConfig
from app.core.logger import setup_logging
from app.rag.data_loader import LegalDataLoader
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import FaissVectorStore


def main() -> None:
    config = AppConfig.from_env()
    config.ensure_dirs()

    logger = setup_logging(config.app_log_file)
    logger.info("Building FAISS index from data_dir=%s", config.data_dir)

    loader = LegalDataLoader(config.data_dir)
    chunks = loader.load_chunks()
    logger.info("Loaded %s legal chunks", len(chunks))

    embedder = EmbeddingService(config.embedding_model)
    vectors = embedder.embed([chunk.text for chunk in chunks], normalize=True)
    logger.info("Generated embeddings with shape=%s", vectors.shape)

    vector_store = FaissVectorStore(config.index_file, config.metadata_file)
    vector_store.build(vectors, chunks)

    logger.info("Index build completed: %s", config.index_file)
    print("FAISS index built successfully")


if __name__ == "__main__":
    main()
