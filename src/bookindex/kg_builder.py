"""
BookIndex – Knowledge Graph Builder

Builds a NetworkX knowledge graph from the chunked legal data.
Uses rule-based extraction for entities and relations.

Entity types:
  - Document, Chapter, Article, LegalConcept, Organization

Relation types:
  - BELONGS_TO, DEFINES, AMENDS, REFERENCES, PARENT_OF
"""
from __future__ import annotations

import re
import logging
from typing import Optional

import networkx as nx

from src.models import TreeNode, NodeLevel, ChunkRecord
from src.bookindex.tree_builder import HierarchicalTree

logger = logging.getLogger(__name__)

# ── Known legal concepts (rule-based seed list) ────────────────────
_KNOWN_CONCEPTS = [
    "giáo dục chính quy", "giáo dục thường xuyên", "đào tạo từ xa",
    "liên thông", "tín chỉ", "tự chủ", "kiểm định chất lượng",
    "chương trình đào tạo", "nghiên cứu khoa học", "hợp tác quốc tế",
    "cơ sở giáo dục đại học", "trường đại học", "học viện", "đại học",
    "đại học quốc gia", "đại học vùng", "cao đẳng", "thạc sĩ", "tiến sĩ",
    "giảng viên", "người học", "sinh viên", "nghiên cứu sinh",
    "hội đồng trường", "hội đồng đại học", "hiệu trưởng", "giám đốc",
    "tuyển sinh", "văn bằng", "chứng chỉ", "học phí",
    "ngân sách nhà nước", "tài chính", "tài sản",
    "giáo dục đại học", "giáo dục mầm non", "giáo dục phổ thông",
    "giáo dục nghề nghiệp", "giáo dục thường xuyên",
    "nhà giáo", "cán bộ quản lý giáo dục", "người học",
    "quyền tự chủ", "trách nhiệm giải trình",
    "công lập", "tư thục", "không vì lợi nhuận",
    "chuyên ngành", "ngành đào tạo", "lĩnh vực",
]

_KNOWN_ORGS = [
    "Bộ Giáo dục và Đào tạo", "Chính phủ", "Quốc hội", "Thủ tướng",
    "Ủy ban nhân dân", "Hội đồng nhân dân", "Bộ Tài chính",
    "Bộ Nội vụ", "Bộ Khoa học và Công nghệ",
]


class KnowledgeGraph:
    """Rule-based knowledge graph for Vietnamese legal education corpus."""

    def __init__(self):
        self.graph = nx.DiGraph()

    # ── Build ──────────────────────────────────────────────────────

    def build_from_tree(self, tree: HierarchicalTree) -> None:
        """Extract entities and relations from the hierarchical tree."""
        # 1) Add structural nodes
        for node_id, node in tree.nodes.items():
            if node.level in (NodeLevel.ROOT,):
                continue
            self._add_structural_node(node)

        # 2) Extract concept/org entities from leaf chunks
        for chunk in tree.get_leaf_chunks():
            self._extract_entities_from_chunk(chunk)

        # 3) Detect amendment relations between documents
        self._detect_amendments(tree)

        logger.info(
            f"KG built: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

    # ── Query ──────────────────────────────────────────────────────

    def find_related_chunks(
        self, concept: str, max_hops: int = 2, limit: int = 20
    ) -> list[str]:
        """Find chunk_ids related to a concept via graph traversal."""
        concept_lower = concept.lower().strip()

        # Find matching concept nodes
        start_nodes = []
        for n, data in self.graph.nodes(data=True):
            if data.get("type") == "concept":
                if concept_lower in n.lower() or n.lower() in concept_lower:
                    start_nodes.append(n)

        if not start_nodes:
            return []

        # BFS traversal
        visited_chunks: list[str] = []
        visited: set[str] = set()

        for start in start_nodes:
            queue = [(start, 0)]
            while queue and len(visited_chunks) < limit:
                current, depth = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)

                node_data = self.graph.nodes.get(current, {})
                if node_data.get("type") == "chunk":
                    chunk_id = node_data.get("chunk_id", current)
                    if chunk_id not in visited_chunks:
                        visited_chunks.append(chunk_id)

                if depth < max_hops:
                    # Follow both directions
                    neighbors = list(self.graph.successors(current)) + list(
                        self.graph.predecessors(current)
                    )
                    for nb in neighbors:
                        if nb not in visited:
                            queue.append((nb, depth + 1))

        return visited_chunks

    def find_entity_articles(self, entity_name: str) -> list[str]:
        """Find article node_ids that mention a given entity."""
        entity_lower = entity_name.lower().strip()
        results = []

        for n, data in self.graph.nodes(data=True):
            if data.get("type") in ("concept", "organization"):
                if entity_lower in n.lower() or n.lower() in entity_lower:
                    # Get connected article/chunk nodes
                    for nb in list(self.graph.predecessors(n)) + list(
                        self.graph.successors(n)
                    ):
                        nb_data = self.graph.nodes.get(nb, {})
                        if nb_data.get("type") in ("article", "chunk"):
                            node_id = nb_data.get("node_id", nb)
                            if node_id not in results:
                                results.append(node_id)

        return results

    def get_document_amendments(self, doc_source: str) -> list[dict]:
        """Get list of amendments for a document."""
        amendments = []
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation") == "AMENDS":
                if doc_source.lower() in u.lower() or doc_source.lower() in v.lower():
                    amendments.append({
                        "amending": u,
                        "amended": v,
                        "relation": "AMENDS",
                    })
        return amendments

    def export_graphml(self, filepath: str) -> None:
        """Export the knowledge graph to GraphML format for visualization."""
        try:
            # GraphML does not support dict/list as attributes. Convert them to strings.
            # Create a copy to sanitize attributes
            g_copy = self.graph.copy()
            for n, data in g_copy.nodes(data=True):
                for k, v in data.items():
                    if isinstance(v, (dict, list, set)):
                        g_copy.nodes[n][k] = str(v)
            for u, v, data in g_copy.edges(data=True):
                for k, val in data.items():
                    if isinstance(val, (dict, list, set)):
                        g_copy.edges[u, v][k] = str(val)
                        
            nx.write_graphml(g_copy, filepath)
            logger.info(f"Knowledge Graph exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export KG to GraphML: {e}")

    # ── Internal ───────────────────────────────────────────────────

    def _add_structural_node(self, node: TreeNode) -> None:
        """Add a tree node to the graph as a structural entity."""
        node_type = node.level.value  # "document", "chapter", "article", ...

        self.graph.add_node(
            node.node_id,
            type=node_type,
            title=node.title,
            source=node.source,
            node_id=node.node_id,
        )

        # Add PARENT_OF edge
        if node.parent_id and node.parent_id != "ROOT":
            self.graph.add_edge(
                node.parent_id, node.node_id, relation="PARENT_OF"
            )

    def _extract_entities_from_chunk(self, chunk: ChunkRecord) -> None:
        """Extract concept and organization entities from a chunk."""
        text_lower = chunk.text.lower()

        # Add chunk node
        self.graph.add_node(
            f"chunk:{chunk.chunk_id}",
            type="chunk",
            chunk_id=chunk.chunk_id,
            node_id=chunk.node_id,
            source=chunk.source,
        )

        # Link chunk to its tree node
        if chunk.node_id in self.graph:
            self.graph.add_edge(
                chunk.node_id, f"chunk:{chunk.chunk_id}", relation="CONTAINS"
            )

        # Extract legal concepts
        for concept in _KNOWN_CONCEPTS:
            if concept in text_lower:
                concept_node_id = f"concept:{concept}"
                if concept_node_id not in self.graph:
                    self.graph.add_node(
                        concept_node_id, type="concept", name=concept
                    )

                # Determine relation type
                relation = "MENTIONS"
                if self._is_definition_context(text_lower, concept):
                    relation = "DEFINES"
                elif self._is_regulation_context(text_lower):
                    relation = "REGULATES"

                self.graph.add_edge(
                    f"chunk:{chunk.chunk_id}",
                    concept_node_id,
                    relation=relation,
                )

        # Extract organizations
        for org in _KNOWN_ORGS:
            if org.lower() in text_lower:
                org_node_id = f"org:{org}"
                if org_node_id not in self.graph:
                    self.graph.add_node(
                        org_node_id, type="organization", name=org
                    )
                self.graph.add_edge(
                    f"chunk:{chunk.chunk_id}",
                    org_node_id,
                    relation="MENTIONS",
                )

        # Extract cross-references (e.g., "Điều 32", "Khoản 3 Điều 15")
        self._extract_cross_references(chunk)

    def _extract_cross_references(self, chunk: ChunkRecord) -> None:
        """Detect and create cross-reference edges."""
        text = chunk.text

        # Pattern: "Điều X" (with optional Khoản/Điểm before)
        # We look for references to other articles
        ref_pattern = r"(?:theo |tại |quy định tại |nêu tại )?[Đđ]iều\s+(\d+[a-z]?)"
        matches = re.findall(ref_pattern, text)

        for art_num in matches:
            # Don't self-reference
            if art_num == chunk.article_number:
                continue

            ref_node_id = f"ref:Dieu_{art_num}"
            if ref_node_id not in self.graph:
                self.graph.add_node(
                    ref_node_id,
                    type="reference",
                    article_number=art_num,
                )

            self.graph.add_edge(
                f"chunk:{chunk.chunk_id}",
                ref_node_id,
                relation="REFERENCES",
            )

    def _detect_amendments(self, tree: HierarchicalTree) -> None:
        """Detect amendment (sửa đổi, bổ sung) relations between documents."""
        for node in tree.nodes.values():
            if node.level != NodeLevel.DOCUMENT:
                continue

            title_lower = node.title.lower()
            if "sửa đổi" in title_lower or "bổ sung" in title_lower:
                # This document amends others
                for other_node in tree.nodes.values():
                    if (
                        other_node.level == NodeLevel.DOCUMENT
                        and other_node.node_id != node.node_id
                    ):
                        # Check if this doc's chunks reference the other
                        self.graph.add_edge(
                            node.node_id,
                            other_node.node_id,
                            relation="AMENDS",
                        )

        # Also check is_deep_split chunks (Luật 34 amends Luật 08)
        amendment_docs = set()
        for chunk in tree.get_leaf_chunks():
            if "sửa đổi" in chunk.article_title.lower():
                amendment_docs.add(chunk.source)

        for doc_source in amendment_docs:
            doc_id = tree._sanitize_id(doc_source)
            for other_id, other_node in tree.nodes.items():
                if (
                    other_node.level == NodeLevel.DOCUMENT
                    and other_id != doc_id
                    and other_id in self.graph
                    and doc_id in self.graph
                ):
                    if not self.graph.has_edge(doc_id, other_id):
                        self.graph.add_edge(
                            doc_id, other_id, relation="POTENTIALLY_AMENDS"
                        )

    @staticmethod
    def _is_definition_context(text: str, concept: str) -> bool:
        """Check if the text defines the concept (e.g., 'X là ...', 'X được hiểu là')."""
        patterns = [
            f"{concept} là ",
            f"{concept} được hiểu là",
            f"{concept} bao gồm",
            f"{concept} gồm",
            "giải thích từ ngữ",
        ]
        return any(p in text for p in patterns)

    @staticmethod
    def _is_regulation_context(text: str) -> bool:
        """Check if the text is a regulatory provision."""
        indicators = [
            "quy định", "được phép", "không được", "phải",
            "có trách nhiệm", "có quyền", "nghĩa vụ",
            "điều kiện", "thủ tục", "hồ sơ",
        ]
        return any(ind in text for ind in indicators)
