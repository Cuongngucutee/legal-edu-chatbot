"""
LLM-Assisted Knowledge Graph Enhancement

Uses LLM to extract richer entities and relationships from
legal text chunks, enhancing the rule-based KG with:
- Fine-grained legal concept extraction
- Causal/conditional relationship detection
- Cross-document reference resolution
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

import networkx as nx
import google.generativeai as genai

from src.config import GEMINI_API_KEY, GEMINI_MODEL
from src.models import ChunkRecord
from src.bookindex.kg_builder import KnowledgeGraph

logger = logging.getLogger(__name__)


_EXTRACTION_PROMPT = """Bạn là chuyên gia trích xuất thông tin từ văn bản pháp luật giáo dục Việt Nam.

Cho đoạn văn bản luật sau, hãy trích xuất các thực thể và quan hệ theo format JSON:

## Đoạn văn bản:
{text}

## Metadata:
- Nguồn: {source}
- Điều: {article}
- Khoản: {clause}

## Yêu cầu trích xuất:

Trả về JSON với format:
```json
{{
  "entities": [
    {{"name": "tên thực thể", "type": "concept|organization|role|document|process", "description": "mô tả ngắn"}}
  ],
  "relations": [
    {{"subject": "thực thể 1", "predicate": "DEFINES|REGULATES|REQUIRES|GRANTS|PROHIBITS|AMENDS|REFERENCES", "object": "thực thể 2", "context": "ngữ cảnh ngắn"}}
  ]
}}
```

Chỉ trích xuất các thực thể và quan hệ RÕ RÀNG trong văn bản. Không suy đoán.
Trả về **chỉ JSON**, không giải thích thêm."""


class KGEnhancer:
    """Enhances knowledge graph with LLM-extracted entities and relations."""

    def __init__(
        self,
        kg: KnowledgeGraph,
        api_key: str = GEMINI_API_KEY,
        model: str = GEMINI_MODEL,
    ):
        self.kg = kg
        self.model_name = model
        self.api_key = api_key
        if api_key:
            genai.configure(api_key=api_key)

    def enhance_from_chunks(
        self,
        chunks: list[ChunkRecord],
        batch_size: int = 5,
        max_chunks: Optional[int] = None,
    ) -> dict:
        """
        Enhance KG with LLM-extracted entities from chunks.

        Args:
            chunks: List of chunks to process
            batch_size: Number of chunks per LLM call
            max_chunks: Maximum chunks to process (None = all)

        Returns:
            Stats dict with counts of added entities/relations
        """
        if not self.api_key:
            logger.warning("No LLM client available for KG enhancement")
            return {"entities_added": 0, "relations_added": 0, "error": "No API key"}

        chunks_to_process = chunks[:max_chunks] if max_chunks else chunks
        total_entities = 0
        total_relations = 0
        errors = 0

        logger.info(f"Enhancing KG from {len(chunks_to_process)} chunks...")

        for i in range(0, len(chunks_to_process), batch_size):
            batch = chunks_to_process[i:i + batch_size]

            for chunk in batch:
                try:
                    result = self._extract_from_chunk(chunk)
                    if result:
                        e, r = self._add_to_graph(result, chunk)
                        total_entities += e
                        total_relations += r
                except Exception as e:
                    logger.warning(f"Error processing chunk {chunk.chunk_id}: {e}")
                    errors += 1

            logger.info(
                f"  Processed {min(i + batch_size, len(chunks_to_process))}/"
                f"{len(chunks_to_process)} chunks"
            )

        stats = {
            "entities_added": total_entities,
            "relations_added": total_relations,
            "chunks_processed": len(chunks_to_process),
            "errors": errors,
            "total_kg_nodes": self.kg.graph.number_of_nodes(),
            "total_kg_edges": self.kg.graph.number_of_edges(),
        }

        logger.info(f"KG enhancement complete: {stats}")
        return stats

    def _extract_from_chunk(self, chunk: ChunkRecord) -> Optional[dict]:
        """Extract entities and relations from a single chunk using LLM."""
        # Skip very short chunks
        if len(chunk.text) < 50:
            return None

        prompt = _EXTRACTION_PROMPT.format(
            text=chunk.text[:1500],  # Limit text length
            source=chunk.source,
            article=chunk.article_title,
            clause=f"Khoản {chunk.clause_number}" if chunk.clause_number else "N/A",
        )

        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(temperature=0.0, max_output_tokens=1000),
            )
            content = response.text or ""

            # Parse JSON from response
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return None

        except json.JSONDecodeError:
            logger.debug(f"Failed to parse JSON from LLM for chunk {chunk.chunk_id}")
            return None
        except Exception as e:
            logger.warning(f"LLM extraction error: {e}")
            return None

    def _add_to_graph(self, extraction: dict, chunk: ChunkRecord) -> tuple[int, int]:
        """Add extracted entities and relations to the knowledge graph."""
        entities_added = 0
        relations_added = 0

        entities = extraction.get("entities", [])
        relations = extraction.get("relations", [])

        # Add entities
        for entity in entities:
            name = entity.get("name", "").strip()
            etype = entity.get("type", "concept")
            description = entity.get("description", "")

            if not name or len(name) < 2:
                continue

            node_id = f"llm:{etype}:{name.lower()}"
            if node_id not in self.kg.graph:
                self.kg.graph.add_node(
                    node_id,
                    type=etype,
                    name=name,
                    description=description,
                    source="llm",
                )
                entities_added += 1

            # Link entity to chunk
            chunk_node = f"chunk:{chunk.chunk_id}"
            if chunk_node in self.kg.graph:
                self.kg.graph.add_edge(
                    chunk_node, node_id, relation="MENTIONS", source="llm"
                )
                relations_added += 1

        # Add relations
        for rel in relations:
            subject = rel.get("subject", "").strip()
            predicate = rel.get("predicate", "RELATED_TO")
            obj = rel.get("object", "").strip()
            context = rel.get("context", "")

            if not subject or not obj:
                continue

            # Find or create subject/object nodes
            subj_id = self._find_or_create_node(subject)
            obj_id = self._find_or_create_node(obj)

            if subj_id and obj_id:
                self.kg.graph.add_edge(
                    subj_id, obj_id,
                    relation=predicate,
                    context=context,
                    source="llm",
                    chunk_id=chunk.chunk_id,
                )
                relations_added += 1

        return entities_added, relations_added

    def _find_or_create_node(self, name: str) -> Optional[str]:
        """Find an existing node by name or create a new one."""
        name_lower = name.lower().strip()

        # Search existing nodes
        for node_id, data in self.kg.graph.nodes(data=True):
            node_name = data.get("name", "").lower()
            if node_name == name_lower:
                return node_id

        # Create new concept node
        node_id = f"llm:concept:{name_lower}"
        self.kg.graph.add_node(
            node_id,
            type="concept",
            name=name,
            source="llm",
        )
        return node_id
