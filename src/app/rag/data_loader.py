from __future__ import annotations

import json
from pathlib import Path
from typing import List

from app.rag.schemas import LegalChunk


class LegalDataLoader:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def load_chunks(self) -> List[LegalChunk]:
        files = sorted(self.data_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No data files found in {self.data_dir}")

        chunks: List[LegalChunk] = []
        for file_path in files:
            with file_path.open("r", encoding="utf-8") as f:
                records = json.load(f)

            for record in records:
                metadata = record.get("metadata", {})
                content = record.get("content", {})
                full_text = str(content.get("full_text", "")).strip()
                if not full_text:
                    continue

                chunks.append(
                    LegalChunk(
                        chunk_id=str(record.get("id", "")),
                        source=str(metadata.get("source", file_path.stem)),
                        article_number=str(metadata.get("article_number", "")),
                        article_title=str(metadata.get("article_title", "")),
                        chapter=str(metadata.get("chapter", "")),
                        section=str(metadata.get("section", "")),
                        text=full_text,
                    )
                )
        return chunks
