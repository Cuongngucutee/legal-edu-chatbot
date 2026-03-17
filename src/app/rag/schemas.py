from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class LegalChunk:
    chunk_id: str
    source: str
    article_number: str
    article_title: str
    chapter: str
    section: str
    text: str


@dataclass(frozen=True)
class RetrievalHit:
    rank: int
    score: float
    chunk: LegalChunk

    def to_debug_dict(self) -> Dict[str, str]:
        return {
            "rank": str(self.rank),
            "score": f"{self.score:.6f}",
            "chunk_id": self.chunk.chunk_id,
            "source": self.chunk.source,
            "article_number": self.chunk.article_number,
            "article_title": self.chunk.article_title,
            "chapter": self.chunk.chapter,
            "section": self.chunk.section,
            "text": self.chunk.text,
        }
