"""
BookIndex – Hierarchical Tree Builder

Transforms flat JSON chunks from data/final/ into a hierarchical tree
structure following the BookRAG approach:

    Root (Corpus)
    ├── Document (Luật 08/2012/QH13)
    │   ├── Chapter (Chương I)
    │   │   ├── Section (Mục 1)  [optional]
    │   │   │   ├── Article (Điều X)
    │   │   │   │   ├── Clause (Khoản 1)
    │   │   │   │   │   ├── Point (Điểm a)
    │   │   │   │   │   └── Point (Điểm b)
    │   │   │   │   └── Clause (Khoản 2)
    │   │   │   └── ...
    │   │   └── Article (Điều Y)  [no section]
    │   └── Chapter (Chương II)
    └── Document (Luật 34/2018/QH14)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Optional

from src.models import TreeNode, NodeLevel, ChunkRecord

logger = logging.getLogger(__name__)


class HierarchicalTree:
    """In-memory hierarchical tree built from chunked legal JSON data."""

    def __init__(self):
        self.nodes: dict[str, TreeNode] = {}
        self.chunks: list[ChunkRecord] = []
        self._root_id = "ROOT"
        # Create root
        root = TreeNode(
            node_id=self._root_id,
            level=NodeLevel.ROOT,
            title="Corpus Luật Giáo dục Việt Nam",
        )
        self.nodes[self._root_id] = root

    # ── Public API ─────────────────────────────────────────────────

    def build_from_directory(self, data_dir: Path) -> None:
        """Load all JSON files and build the tree."""
        json_files = sorted(data_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files in {data_dir}")

        for json_path in json_files:
            self._load_file(json_path)

        # Build summaries bottom-up
        self._build_summaries(self._root_id)
        logger.info(
            f"Tree built: {len(self.nodes)} nodes, {len(self.chunks)} chunks"
        )

    def get_node(self, node_id: str) -> Optional[TreeNode]:
        return self.nodes.get(node_id)

    def get_children(self, node_id: str) -> list[TreeNode]:
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[cid] for cid in node.children_ids if cid in self.nodes]

    def get_ancestors(self, node_id: str) -> list[TreeNode]:
        """Return ancestors from node up to root (excluding root)."""
        ancestors = []
        current = self.nodes.get(node_id)
        while current and current.parent_id and current.parent_id != self._root_id:
            parent = self.nodes.get(current.parent_id)
            if parent:
                ancestors.append(parent)
            current = parent
        return ancestors

    def get_siblings(self, node_id: str) -> list[TreeNode]:
        node = self.nodes.get(node_id)
        if not node or not node.parent_id:
            return []
        parent = self.nodes.get(node.parent_id)
        if not parent:
            return []
        return [
            self.nodes[cid]
            for cid in parent.children_ids
            if cid in self.nodes and cid != node_id
        ]

    def get_leaf_chunks(self) -> list[ChunkRecord]:
        """Return all leaf-level chunks for indexing."""
        return self.chunks

    def get_article_node_for_chunk(self, chunk_id: str) -> Optional[TreeNode]:
        """Given a chunk_id, walk up the tree to find the article-level node."""
        # Find the node by chunk_id
        for node in self.nodes.values():
            if node.original_chunk_id == chunk_id:
                current = node
                while current:
                    if current.level == NodeLevel.ARTICLE:
                        return current
                    if current.parent_id:
                        current = self.nodes.get(current.parent_id)
                    else:
                        break
                break
        return None

    def print_tree(self, node_id: str = None, indent: int = 0, max_depth: int = 4) -> str:
        """Pretty-print the tree for debugging."""
        if node_id is None:
            node_id = self._root_id
        node = self.nodes.get(node_id)
        if not node or indent > max_depth * 4:
            return ""

        prefix = "│   " * (indent // 4) + ("├── " if indent > 0 else "")
        lines = [f"{prefix}[{node.level.value}] {node.title or node.node_id}"]

        for child_id in node.children_ids:
            lines.append(self.print_tree(child_id, indent + 4, max_depth))

        return "\n".join(line for line in lines if line)

    # ── Internal ───────────────────────────────────────────────────

    def _load_file(self, json_path: Path) -> None:
        """Load a single JSON file and add its chunks to the tree."""
        with open(json_path, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)

        logger.info(f"Loading {json_path.name}: {len(raw_chunks)} chunks")

        for raw in raw_chunks:
            self._process_chunk(raw)

    def _process_chunk(self, raw: dict) -> None:
        """Process a single raw chunk dict and insert into tree."""
        chunk_id = raw["id"]
        meta = raw["metadata"]
        content = raw["content"]

        source = meta.get("source", "")
        chapter = meta.get("chapter", "")
        section = meta.get("section", "")
        article_number = meta.get("article_number", "")
        article_title = meta.get("article_title", "")
        is_sub_split = meta.get("is_sub_split", False)
        split_level = meta.get("split_level", "article")
        clause_number = meta.get("clause_number", "")
        point_label = meta.get("point_label", "")
        tags = meta.get("tags", [])

        chunk_text = content.get("chunk_text", "")
        full_text = content.get("full_text", "")
        article_full_text = content.get("article_full_text", "")

        # ── Ensure Document node ──
        doc_id = self._sanitize_id(source)
        if doc_id not in self.nodes:
            doc_node = TreeNode(
                node_id=doc_id,
                level=NodeLevel.DOCUMENT,
                title=source,
                source=source,
                parent_id=self._root_id,
            )
            self.nodes[doc_id] = doc_node
            self.nodes[self._root_id].children_ids.append(doc_id)

        # ── Ensure Chapter node ──
        chapter_id = f"{doc_id}__{self._sanitize_id(chapter)}" if chapter else doc_id
        if chapter and chapter_id not in self.nodes:
            chapter_node = TreeNode(
                node_id=chapter_id,
                level=NodeLevel.CHAPTER,
                title=chapter,
                source=source,
                chapter=chapter,
                parent_id=doc_id,
            )
            self.nodes[chapter_id] = chapter_node
            self.nodes[doc_id].children_ids.append(chapter_id)

        parent_for_article = chapter_id

        # ── Ensure Section node (if present) ──
        if section:
            section_id = f"{chapter_id}__{self._sanitize_id(section)}"
            if section_id not in self.nodes:
                section_node = TreeNode(
                    node_id=section_id,
                    level=NodeLevel.SECTION,
                    title=section,
                    source=source,
                    chapter=chapter,
                    section=section,
                    parent_id=chapter_id,
                )
                self.nodes[section_id] = section_node
                self.nodes[chapter_id].children_ids.append(section_id)
            parent_for_article = section_id

        # ── Ensure Article node ──
        # For deep-split docs (like Luật 34), use target_article_number
        effective_article = meta.get("target_article_number", article_number)
        article_heading = meta.get("article_heading", article_title)

        article_id = f"{parent_for_article}__D{effective_article}"
        if article_id not in self.nodes:
            article_node = TreeNode(
                node_id=article_id,
                level=NodeLevel.ARTICLE,
                title=article_heading or f"Điều {effective_article}",
                source=source,
                chapter=chapter,
                section=section,
                article_number=effective_article,
                article_title=article_heading or article_title,
                tags=tags,
                parent_id=parent_for_article,
                article_full_text=article_full_text,
            )
            self.nodes[article_id] = article_node
            if article_id not in self.nodes[parent_for_article].children_ids:
                self.nodes[parent_for_article].children_ids.append(article_id)

        # ── Handle leaf vs sub-split ──
        if not is_sub_split:
            # Article-level chunk (no further split)
            leaf_node_id = article_id
            node = self.nodes[leaf_node_id]
            node.full_text = full_text
            node.chunk_text = chunk_text
            node.article_full_text = article_full_text
            node.original_chunk_id = chunk_id
        else:
            # Sub-split: clause or point
            if split_level == "clause" and clause_number:
                clause_id = f"{article_id}__K{clause_number}"
                if clause_id not in self.nodes:
                    clause_node = TreeNode(
                        node_id=clause_id,
                        level=NodeLevel.CLAUSE,
                        title=f"Khoản {clause_number}",
                        full_text=full_text,
                        chunk_text=chunk_text,
                        article_full_text=article_full_text,
                        source=source,
                        chapter=chapter,
                        section=section,
                        article_number=effective_article,
                        article_title=article_heading or article_title,
                        clause_number=clause_number,
                        tags=tags,
                        parent_id=article_id,
                        original_chunk_id=chunk_id,
                    )
                    self.nodes[clause_id] = clause_node
                    if clause_id not in self.nodes[article_id].children_ids:
                        self.nodes[article_id].children_ids.append(clause_id)

                leaf_node_id = clause_id

            elif split_level == "point" and point_label:
                # Point is child of clause
                clause_id = f"{article_id}__K{clause_number}"
                # Ensure clause node exists
                if clause_id not in self.nodes:
                    clause_node = TreeNode(
                        node_id=clause_id,
                        level=NodeLevel.CLAUSE,
                        title=f"Khoản {clause_number}",
                        source=source,
                        chapter=chapter,
                        section=section,
                        article_number=effective_article,
                        article_title=article_heading or article_title,
                        clause_number=clause_number,
                        tags=tags,
                        parent_id=article_id,
                        article_full_text=article_full_text,
                    )
                    self.nodes[clause_id] = clause_node
                    if clause_id not in self.nodes[article_id].children_ids:
                        self.nodes[article_id].children_ids.append(clause_id)

                point_id = f"{clause_id}__P{point_label}"
                if point_id not in self.nodes:
                    point_node = TreeNode(
                        node_id=point_id,
                        level=NodeLevel.POINT,
                        title=f"Điểm {point_label}",
                        full_text=full_text,
                        chunk_text=chunk_text,
                        article_full_text=article_full_text,
                        source=source,
                        chapter=chapter,
                        section=section,
                        article_number=effective_article,
                        article_title=article_heading or article_title,
                        clause_number=clause_number,
                        point_label=point_label,
                        tags=tags,
                        parent_id=clause_id,
                        original_chunk_id=chunk_id,
                    )
                    self.nodes[point_id] = point_node
                    if point_id not in self.nodes[clause_id].children_ids:
                        self.nodes[clause_id].children_ids.append(point_id)

                leaf_node_id = point_id
            else:
                leaf_node_id = article_id

        # ── Create ChunkRecord for retrieval ──
        # Use chunk_text for retrieval; enrich with article context
        enriched_text = self._build_enriched_text(
            source, chapter, section, article_heading or article_title,
            clause_number, point_label, chunk_text,
        )

        chunk_record = ChunkRecord(
            chunk_id=chunk_id,
            node_id=leaf_node_id,
            text=enriched_text,
            source=source,
            chapter=chapter,
            section=section,
            article_number=effective_article,
            article_title=article_heading or article_title,
            clause_number=clause_number,
            point_label=point_label,
            split_level=split_level,
            article_full_text=article_full_text,
            tags=tags,
        )
        self.chunks.append(chunk_record)

    def _build_enriched_text(
        self, source: str, chapter: str, section: str,
        article_title: str, clause_number: str, point_label: str,
        chunk_text: str,
    ) -> str:
        """Build enriched text for better retrieval by prepending context."""
        parts = []
        if source:
            parts.append(f"[{source}]")
        if chapter:
            parts.append(f"[{chapter}]")
        if section:
            parts.append(f"[{section}]")
        if article_title:
            parts.append(f"[{article_title}]")
        if clause_number:
            parts.append(f"[Khoản {clause_number}]")
        if point_label:
            parts.append(f"[Điểm {point_label}]")

        header = " ".join(parts)
        if header:
            return f"{header}\n{chunk_text}"
        return chunk_text

    def _build_summaries(self, node_id: str) -> str:
        """Recursively build summaries bottom-up."""
        node = self.nodes.get(node_id)
        if not node:
            return ""

        if not node.children_ids:
            # Leaf node — summary is first 200 chars of chunk_text
            node.summary = (node.chunk_text or node.full_text or node.title)[:200]
            return node.summary

        child_summaries = []
        for child_id in node.children_ids:
            s = self._build_summaries(child_id)
            if s:
                child_summaries.append(s)

        # Combine child summaries (truncated)
        combined = "; ".join(child_summaries)
        node.summary = combined[:300] if combined else node.title
        return node.summary

    @staticmethod
    def _sanitize_id(text: str) -> str:
        """Create a safe ID string from text."""
        return (
            text.replace("/", "_")
            .replace(" ", "_")
            .replace(".", "_")
            .replace(",", "")
            .strip("_")
        )
