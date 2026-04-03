"""
Pydantic data models used throughout the BookRAG pipeline.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────

class NodeLevel(str, Enum):
    ROOT = "root"
    DOCUMENT = "document"
    CHAPTER = "chapter"
    SECTION = "section"
    ARTICLE = "article"
    CLAUSE = "clause"
    POINT = "point"


class QueryIntent(str, Enum):
    DEFINITION = "definition"
    PROCEDURE = "procedure"
    CONDITION = "condition"
    COMPARISON = "comparison"
    GENERAL = "general"


# ── Tree Nodes ─────────────────────────────────────────────────────

class TreeNode(BaseModel):
    """A node in the BookRAG hierarchical tree."""
    node_id: str
    level: NodeLevel
    title: str = ""
    summary: str = ""                # short summary for navigation
    full_text: str = ""              # full content for leaf nodes
    chunk_text: str = ""             # chunk-level text (for retrieval)
    article_full_text: str = ""      # complete article text (for context)
    parent_id: Optional[str] = None
    children_ids: list[str] = Field(default_factory=list)

    # Original metadata
    source: str = ""                 # e.g. "Luật 08/2012/QH13"
    chapter: str = ""
    section: str = ""
    article_number: str = ""
    article_title: str = ""
    clause_number: str = ""
    point_label: str = ""
    tags: list[str] = Field(default_factory=list)
    original_chunk_id: str = ""      # id from the JSON data


# ── Chunk for retrieval ────────────────────────────────────────────

class ChunkRecord(BaseModel):
    """A flat chunk record for indexing & retrieval."""
    chunk_id: str
    node_id: str                     # maps back to tree node
    text: str                        # text used for embedding / BM25
    source: str = ""
    chapter: str = ""
    section: str = ""
    article_number: str = ""
    article_title: str = ""
    clause_number: str = ""
    point_label: str = ""
    split_level: str = ""
    article_full_text: str = ""
    tags: list[str] = Field(default_factory=list)


# ── Query models ───────────────────────────────────────────────────

class ParsedQuery(BaseModel):
    """Result of query processing."""
    original: str
    normalized: str
    intent: QueryIntent = QueryIntent.GENERAL
    # Extracted entities
    law_references: list[str] = Field(default_factory=list)       # e.g. ["Luật 08/2012/QH13"]
    article_references: list[str] = Field(default_factory=list)   # e.g. ["Điều 4"]
    clause_references: list[str] = Field(default_factory=list)    # e.g. ["Khoản 2"]
    point_references: list[str] = Field(default_factory=list)     # e.g. ["Điểm a"]
    key_concepts: list[str] = Field(default_factory=list)         # e.g. ["tự chủ", "giáo dục đại học"]


# ── Retrieval result ───────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """A single retrieval result with score."""
    chunk: ChunkRecord
    score: float = 0.0
    source_method: str = ""          # "dense", "sparse", "kg"


# ── Reasoning ──────────────────────────────────────────────────────

class ReasoningStep(BaseModel):
    """A single step in the chain-of-thought reasoning."""
    step_name: str
    content: str


class ReasoningResult(BaseModel):
    """Full reasoning output."""
    query_analysis: str = ""
    plan: str = ""
    evidence_alignment: str = ""
    judgment: str = ""
    needs_more_info: bool = False
    additional_queries: list[str] = Field(default_factory=list)


# ── Final answer ───────────────────────────────────────────────────

class Citation(BaseModel):
    """A citation reference in the answer."""
    source: str           # e.g. "Luật 08/2012/QH13"
    article: str = ""     # e.g. "Điều 4"
    clause: str = ""      # e.g. "Khoản 2"
    point: str = ""       # e.g. "Điểm a"
    text_excerpt: str = ""

    def format(self) -> str:
        parts = [self.source]
        if self.article:
            parts.append(self.article)
        if self.clause:
            parts.append(self.clause)
        if self.point:
            parts.append(self.point)
        return ", ".join(parts)


class PipelineAnswer(BaseModel):
    """The final answer returned to the user."""
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0          # 0.0 – 1.0
    reasoning_trace: list[ReasoningStep] = Field(default_factory=list)
    related_questions: list[str] = Field(default_factory=list)
    warning: str = ""
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
