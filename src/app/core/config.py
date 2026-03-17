from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    data_dir: Path
    index_dir: Path
    index_file: Path
    metadata_file: Path
    logs_dir: Path
    retrieval_log_file: Path
    app_log_file: Path
    embedding_model: str
    retrieval_top_k: int

    @classmethod
    def from_env(cls) -> "AppConfig":
        project_root = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[3]))
        data_dir = Path(os.getenv("DATA_DIR", project_root / "data" / "final"))
        index_dir = Path(os.getenv("INDEX_DIR", project_root / "outputs" / "indexes" / "faiss"))
        logs_dir = Path(os.getenv("LOGS_DIR", project_root / "logs"))

        embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        return cls(
            project_root=project_root,
            data_dir=data_dir,
            index_dir=index_dir,
            index_file=index_dir / "legal_faiss.index",
            metadata_file=index_dir / "legal_chunks_metadata.json",
            logs_dir=logs_dir,
            retrieval_log_file=logs_dir / "retrieval_trace.log",
            app_log_file=logs_dir / "app.log",
            embedding_model=embedding_model,
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
        )

    def ensure_dirs(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
