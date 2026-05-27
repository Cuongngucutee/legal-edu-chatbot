import json
import os
import re
import string
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple

# CRITICAL: phải set TRƯỚC khi import torch (qua sentence_transformers)
# Tắt MPS memory watermark để tránh segfault khi load nhiều model cùng lúc
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import faiss
import networkx as nx
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

class BookIndex:
    def __init__(self, data_dir: str, kg_path: str, embed_model_name: str = "namnguyenba2003/Vietnamese_Law_Embedding_finetuned_v3_256dims"):
        self.data_dir = Path(data_dir)
        self.kg_path = Path(kg_path)
        self.graph = nx.DiGraph()
        
        print(f"Loading embedding model: {embed_model_name}...")
        self.encoder = SentenceTransformer(embed_model_name)
        
        # Determine embedding dimension
        dummy_emb = self.encoder.encode("test")
        self.embed_dim = dummy_emb.shape[0]
        
        # Vector index for nodes
        self.faiss_index = faiss.IndexFlatIP(self.embed_dim) # Inner product (Cosine sim if normalized)
        self.node_mapping = {} # index in faiss -> node_id
        
        # Lexical (BM25) index
        self.bm25_index = None
        self.bm25_nodes = [] # align with BM25 corpus
        
        # Document registry for entity resolution (so_hieu variants → canonical)
        self.doc_registry = {}  # "81/2021" → "01/2021/TT-BGDĐT"
        self.doc_nodes = {}     # so_hieu → set of node_ids belonging to that doc
        self.doc_faiss_map = {} # canonical → set of FAISS indices (for scoped search)
        
        # Document Catalog for statistical/listing queries
        self.doc_catalog = []       # List[dict] — one entry per document
        self.doc_catalog_faiss = None  # Separate FAISS for doc-level semantic search
        self.doc_catalog_bm25 = None   # Separate BM25 for doc-level lexical search
        self.doc_catalog_mapping = {}  # FAISS idx → catalog idx

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()
        
    def _slug(self, text: str) -> str:
        text = str(text).lower().strip()
        text = re.sub(r'\s+', '_', text)
        text = re.sub(r'[^\w_]', '', text, flags=re.UNICODE)
        return text.strip('_') or "unknown"

    def load_index(self):
        print("Loading Knowledge Graph...")
        with open(self.kg_path, 'r', encoding='utf-8') as f:
            kg_data = json.load(f)
            
        kg_root = kg_data.get("bookrag_entity_graph", {})
        entities = kg_root.get("entities", [])
        relations = kg_root.get("relations", [])
        mappings = kg_root.get("mappings", [])
        
        # Add entities as nodes
        for ent in entities:
            node_id = ent.get("id")
            ent_type = ent.get("type", "Entity")
            name = ent.get("name", "")
            
            search_text = f"{ent_type}: {name} - {ent.get('description', '')}"
            self.graph.add_node(node_id, search_text=search_text, **ent)
            
        # Add relations as edges
        for rel in relations:
            sub = rel.get("source")
            predicate = rel.get("relation")
            obj = rel.get("target")
            if sub and obj and predicate:
                self.graph.add_edge(sub, obj, type=predicate)
                self.graph.add_edge(obj, sub, type=f"REV_{predicate}")

        # Add mappings (Tree Node <-> Entity)
        for mapping in mappings:
            tree_node_id = mapping.get("tree_node")
            entity_id = mapping.get("entity")
            if tree_node_id and entity_id:
                self.graph.add_edge(tree_node_id, entity_id, type="MENTIONS")
                self.graph.add_edge(entity_id, tree_node_id, type="MENTIONED_IN")

        print("Augmenting with Structural Data from JSONs...")
        self._augment_with_json_tree()

        print("Building Document Registry...")
        self._build_doc_registry()

        print("Building Vector Index...")
        self._build_vector_index()
        
        print("Building Doc-FAISS Map...")
        self._build_doc_faiss_map()
        
        print("Building Document Catalog...")
        self._build_doc_catalog()
        
    def _augment_with_json_tree(self):
        # Read JSON chunks to get full texts and hierarchical structures (chapters, sections)
        if not self.data_dir.exists():
            print(f"Data dir {self.data_dir} does not exist. Skipping augmentation.")
            return
            
        json_files = list(self.data_dir.glob("*.json"))
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                
                # We need to map chunks back to Article nodes from the Knowledge Graph
                for chunk in chunks:
                    meta = chunk.get("metadata", {})
                    content = chunk.get("content", {})
                    
                    doc_source = meta.get("source", file_path.stem)
                    doc_slug = self._slug(doc_source)
                    doc_id = f"doc:{doc_slug}"
                    
                    chunk_id = str(chunk.get("id", "")).strip()
                    if chunk_id:
                        article_id = f"node:{chunk_id}"
                    else:
                        art_num = str(meta.get("article_number", "")).strip()
                        target = str(meta.get("target_article", "")).strip()
                        suffix = self._slug(target) if target else f"d{art_num}"
                        article_id = f"node:{doc_slug}:{suffix}"
                    
                    full_text = content.get("full_text", "")
                    chapter_name = meta.get("chapter", "").strip()
                    section_name = meta.get("section", "").strip()
                    
                    # Ensure Document Node has title
                    if doc_id in self.graph.nodes:
                        self.graph.nodes[doc_id]['search_text'] = self.graph.nodes[doc_id].get('name', doc_source)

                    # Augment Article node with full_text and Tree parents
                    # Tạo node mới nếu chưa tồn tại (quan trọng: KG chỉ có ent: nodes)
                    article_name = meta.get("target_article", "") or f"Điều {meta.get('article_number', '?')}"
                    if article_id not in self.graph.nodes:
                        self.graph.add_node(article_id, type="Article", name=article_name)
                    
                    self.graph.nodes[article_id]['full_text'] = full_text
                    # Gắn search_text để FAISS embed trực tiếp nội dung Điều khoản
                    # Truncate 512 chars (đủ chứa tiêu đề + nội dung chính)
                    self.graph.nodes[article_id]['search_text'] = full_text[:512]
                    
                    # Link article → document
                    if doc_id in self.graph.nodes:
                        self.graph.add_edge(article_id, doc_id, type="THUOC_VAN_BAN")
                        self.graph.add_edge(doc_id, article_id, type="REV_THUOC_VAN_BAN")
                        
                    # Add Chapter Node
                    if chapter_name:
                        chapter_id = f"chap:{doc_slug}:{self._slug(chapter_name)}"
                        if chapter_id not in self.graph.nodes:
                            self.graph.add_node(chapter_id, type="Chapter", name=chapter_name, search_text=f"Chapter: {chapter_name} of {doc_source}")
                            self.graph.add_edge(chapter_id, doc_id, type="THUOC_VAN_BAN")
                            self.graph.add_edge(doc_id, chapter_id, type="REV_THUOC_VAN_BAN")
                        
                        self.graph.add_edge(article_id, chapter_id, type="THUOC_CHUONG")
                        self.graph.add_edge(chapter_id, article_id, type="REV_THUOC_CHUONG")
                        
                    # Add Section Node
                    if section_name:
                        section_id = f"sec:{doc_slug}:{self._slug(section_name)}"
                        if section_id not in self.graph.nodes:
                            self.graph.add_node(section_id, type="Section", name=section_name, search_text=f"Section: {section_name} of {doc_source}")
                            if chapter_name:
                                chapter_id = f"chap:{doc_slug}:{self._slug(chapter_name)}"
                                self.graph.add_edge(section_id, chapter_id, type="THUOC_CHUONG")
                                self.graph.add_edge(chapter_id, section_id, type="REV_THUOC_CHUONG")
                            else:
                                self.graph.add_edge(section_id, doc_id, type="THUOC_VAN_BAN")
                                self.graph.add_edge(doc_id, section_id, type="REV_THUOC_VAN_BAN")
                        
                        self.graph.add_edge(article_id, section_id, type="THUOC_MUC")
                        self.graph.add_edge(section_id, article_id, type="REV_THUOC_MUC")
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")

    def _build_doc_registry(self):
        """Build reverse mapping: so_hieu variants → canonical so_hieu.
        Also builds doc_nodes: so_hieu → set of node_ids in that document."""
        self.doc_registry = {}
        self.doc_nodes = {}
        
        json_files = list(self.data_dir.glob("*.json"))
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                if not chunks:
                    continue
                
                so_hieu = chunks[0].get("metadata", {}).get("so_hieu", "") or ""
                source = chunks[0].get("metadata", {}).get("source", "") or ""
                
                # Nếu không có so_hieu, trích xuất từ source
                # VD: source="Luật 43/2019/QH14" → canonical="43/2019/QH14"
                canonical = so_hieu
                if not canonical and source:
                    m = re.search(r'(\d+[\/-]\d{4}[\/-]?[\w\-]*)', source)
                    if m:
                        canonical = m.group(1)
                    else:
                        canonical = source  # Fallback dùng source làm canonical
                
                if not canonical:
                    continue
                
                # Build doc_nodes set
                if canonical not in self.doc_nodes:
                    self.doc_nodes[canonical] = set()
                
                for chunk in chunks:
                    chunk_id = str(chunk.get("id", "")).strip()
                    if chunk_id:
                        node_id = f"node:{chunk_id}"
                        self.doc_nodes[canonical].add(node_id)
                
                # Register canonical
                self.doc_registry[canonical] = canonical
                self.doc_registry[canonical.lower()] = canonical
                
                # Register source name and variants
                if source:
                    self.doc_registry[source] = canonical
                    self.doc_registry[source.lower()] = canonical
                
                # Register number variants: "81/2021", "81"
                parts = canonical.split("/")
                if len(parts) >= 2:
                    short = f"{parts[0]}/{parts[1]}"  # "81/2021"
                    self.doc_registry[short] = canonical
                    self.doc_registry[short.lower()] = canonical
                if parts:
                    # Only store number if it's not ambiguous (>= 2 digits)
                    num = parts[0].strip()
                    if len(num) >= 2 and num not in self.doc_registry:
                        self.doc_registry[num] = canonical
                
            except Exception as e:
                print(f"[BookIndex] Warning: skipping {file_path.name}: {e}")

        
        print(f"  Document Registry: {len(self.doc_registry)} entries → {len(self.doc_nodes)} documents")

    
    def get_doc_node_ids(self, so_hieu: str) -> set:
        """Get all node IDs belonging to a document by so_hieu."""
        canonical = self.doc_registry.get(so_hieu, so_hieu)
        return self.doc_nodes.get(canonical, set())

    def build_toc(self, so_hieu: str) -> str:
        """Build a structured Table of Contents (TOC) for a document.
        
        Returns a formatted string listing all articles grouped by chapter/section,
        using only article titles (not full text) to keep context compact.
        """
        node_ids = self.get_doc_node_ids(so_hieu)
        if not node_ids:
            return ""
        
        # Collect article info with their chapter/section from graph edges
        articles = []
        for node_id in node_ids:
            if node_id not in self.graph.nodes:
                continue
            
            # Skip CAN_CU nodes (căn cứ pháp lý, không phải điều luật)
            if "_CAN_CU" in node_id:
                continue
            
            node_data = self.graph.nodes[node_id]
            
            # Get article title: prefer 'name', fallback to first line of full_text
            name = node_data.get("name", "")
            if not name:
                full_text = node_data.get("full_text", "") or node_data.get("search_text", "")
                if full_text:
                    # First line = article title (e.g. "Điều 1. Phạm vi điều chỉnh")
                    first_line = full_text.split("\n")[0].strip()
                    # Truncate long titles
                    name = first_line[:120] if len(first_line) > 120 else first_line
            
            if not name:
                continue
            
            # Get chapter via THUOC_CHUONG edge
            chapter = ""
            for neighbor in self.graph.neighbors(node_id):
                edge_type = self.graph.edges[node_id, neighbor].get("type", "")
                if edge_type == "THUOC_CHUONG":
                    chapter = self.graph.nodes[neighbor].get("name", "")
                    break
            
            # Get section via THUOC_MUC edge
            section = ""
            for neighbor in self.graph.neighbors(node_id):
                edge_type = self.graph.edges[node_id, neighbor].get("type", "")
                if edge_type == "THUOC_MUC":
                    section = self.graph.nodes[neighbor].get("name", "")
                    break
            
            # Extract article number for sorting
            art_num = 0
            m = re.search(r'Điều\s+(\d+)', name)
            if m:
                art_num = int(m.group(1))
            
            articles.append({
                "name": name,
                "chapter": chapter,
                "section": section,
                "art_num": art_num,
            })
        
        if not articles:
            return ""
        
        # Sort by article number
        articles.sort(key=lambda a: a["art_num"])
        
        # Build structured TOC grouped by chapter → section
        canonical = self.doc_registry.get(so_hieu, so_hieu)
        toc_lines = [f"=== MỤC LỤC: {canonical} ==="]
        
        current_chapter = None
        current_section = None
        
        for art in articles:
            if art["chapter"] and art["chapter"] != current_chapter:
                current_chapter = art["chapter"]
                current_section = None  # Reset section when chapter changes
                toc_lines.append(f"\n{current_chapter}")
            
            if art["section"] and art["section"] != current_section:
                current_section = art["section"]
                toc_lines.append(f"  {current_section}")
            
            indent = "    " if current_section else "  "
            toc_lines.append(f"{indent}- {art['name']}")
        
        toc_lines.append(f"\nTổng cộng: {len(articles)} điều")
        return "\n".join(toc_lines)

    def build_enriched_toc(self, so_hieu: str, snippet_len: int = 200) -> str:
        """Build an enriched TOC with content snippets for each article.
        
        Unlike build_toc() which only shows article titles, this method includes
        the first `snippet_len` characters of each article's content. This helps
        the LLM distinguish between articles with similar titles during Stage 2
        clause selection.
        
        Args:
            so_hieu: Document identifier (so_hieu or canonical).
            snippet_len: Number of characters to include as snippet per article.
            
        Returns:
            Formatted string with article titles + content snippets.
        """
        node_ids = self.get_doc_node_ids(so_hieu)
        if not node_ids:
            return ""
        
        # Selective TOC Enrichment: skip snippets for general procedural documents (distractors)
        canonical = self.doc_registry.get(so_hieu, so_hieu).upper()
        distractors = ["13/2024", "125/2024", "238/2025", "04/2021"]
        should_enrich = not any(d in canonical for d in distractors)
        
        articles = []
        for node_id in node_ids:
            if node_id not in self.graph.nodes:
                continue
            if "_CAN_CU" in node_id:
                continue
            
            node_data = self.graph.nodes[node_id]
            
            # Get article title
            name = node_data.get("name", "")
            full_text = node_data.get("full_text", "") or node_data.get("search_text", "")
            if not name and full_text:
                first_line = full_text.split("\n")[0].strip()
                name = first_line[:120] if len(first_line) > 120 else first_line
            
            if not name:
                continue
            
            # Extract content snippet (skip the title line)
            snippet = ""
            if should_enrich and full_text:
                lines = full_text.split("\n")
                # Skip first line (title) and get content
                content_lines = [l.strip() for l in lines[1:] if l.strip()]
                content_text = " ".join(content_lines)
                if content_text:
                    snippet = content_text[:snippet_len].strip()
                    if len(content_text) > snippet_len:
                        snippet += "..."
            
            # Extract article number for sorting
            art_num = 0
            m = re.search(r'Điều\s+(\d+)', name)
            if m:
                art_num = int(m.group(1))
            
            # Get chapter via THUOC_CHUONG edge
            chapter = ""
            for neighbor in self.graph.neighbors(node_id):
                edge_type = self.graph.edges[node_id, neighbor].get("type", "")
                if edge_type == "THUOC_CHUONG":
                    chapter = self.graph.nodes[neighbor].get("name", "")
                    break
            
            articles.append({
                "name": name,
                "snippet": snippet,
                "chapter": chapter,
                "art_num": art_num,
            })
        
        if not articles:
            return ""
        
        articles.sort(key=lambda a: a["art_num"])
        
        canonical = self.doc_registry.get(so_hieu, so_hieu)
        toc_lines = [f"=== MỤC LỤC CHI TIẾT: {canonical} ==="]
        
        current_chapter = None
        for art in articles:
            if art["chapter"] and art["chapter"] != current_chapter:
                current_chapter = art["chapter"]
                toc_lines.append(f"\n{current_chapter}")
            
            toc_lines.append(f"  - {art['name']}")
            if art["snippet"]:
                toc_lines.append(f"    → {art['snippet']}")
        
        toc_lines.append(f"\nTổng cộng: {len(articles)} điều")
        return "\n".join(toc_lines)
    
    # ─── Document Catalog (Statistical Queries) ────────────────────────

    def _build_doc_catalog(self):
        """Build document-level catalog with metadata + summaries.
        
        Each entry contains: source, so_hieu, doc_type, year, summary, canonical.
        Also builds a doc-level FAISS index on summaries for semantic search.
        """
        self.doc_catalog = []
        json_files = list(self.data_dir.glob("*.json"))
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                if not isinstance(chunks, list) or not chunks:
                    continue
                
                meta = chunks[0].get("metadata", {})
                source = meta.get("source", "") or ""
                so_hieu = meta.get("so_hieu", "") or ""
                
                if not source:
                    continue
                
                # Extract doc_type from source prefix
                source_lower = source.lower()
                if source_lower.startswith("nghị định"):
                    doc_type = "Nghị định"
                elif source_lower.startswith("thông tư"):
                    doc_type = "Thông tư"
                elif source_lower.startswith("luật"):
                    doc_type = "Luật"
                else:
                    doc_type = "Khác"
                
                # Extract year from so_hieu (e.g. '238/2025/NĐ-CP' → 2025)
                year = ""
                m = re.search(r'(\d{4})', so_hieu)
                if m:
                    year = m.group(1)
                
                # Extract summary from Điều 1 (Phạm vi điều chỉnh)
                summary = ""
                for chunk in chunks:
                    c_meta = chunk.get("metadata", {})
                    art_num = str(c_meta.get("article_number", "")).strip()
                    if art_num == "1":
                        full_text = chunk.get("content", {}).get("full_text", "")
                        # Take first 300 chars as summary
                        summary = full_text[:300].strip()
                        break
                
                # Fallback summary from CAN_CU or first chunk
                if not summary:
                    for chunk in chunks:
                        full_text = chunk.get("content", {}).get("full_text", "")
                        if full_text and "_CAN_CU" not in str(chunk.get("id", "")):
                            summary = full_text[:300].strip()
                            break
                
                canonical = self.doc_registry.get(so_hieu, so_hieu)
                
                self.doc_catalog.append({
                    "source": source,
                    "so_hieu": so_hieu,
                    "canonical": canonical,
                    "doc_type": doc_type,
                    "year": year,
                    "summary": summary,
                })
            except Exception as e:
                print(f"[Catalog] Warning: skipping {file_path.name}: {e}")
        
        # Build doc-level Hybrid index on summaries (FAISS + BM25)
        if self.doc_catalog:
            texts = [f"{entry['source']}: {entry['summary']}" for entry in self.doc_catalog]
            embeddings = self.encoder.encode(texts, normalize_embeddings=True)
            embeddings = np.array(embeddings).astype('float32')
            
            self.doc_catalog_faiss = faiss.IndexFlatIP(self.embed_dim)
            self.doc_catalog_faiss.add(embeddings)
            
            tokenized_corpus = [self._tokenize(text) for text in texts]
            self.doc_catalog_bm25 = BM25Okapi(tokenized_corpus)
            
            self.doc_catalog_mapping = {i: i for i in range(len(self.doc_catalog))}
        
        print(f"  Document Catalog: {len(self.doc_catalog)} documents indexed")

    def query_catalog(self, query: str, 
                      doc_type: str = None, 
                      year: str = None,
                      semantic_top_k: int = 20,
                      max_results: int = 15) -> str:
        """Query the document catalog using metadata filters + semantic search.
        
        Args:
            query: User query for semantic matching
            doc_type: Filter by type ('Nghị định', 'Thông tư', 'Luật')
            year: Filter by year ('2025', '2026', etc.)
            semantic_top_k: Number of semantic matches to consider
            max_results: Max documents to return
        
        Returns:
            Formatted string listing matching documents
        """
        if not self.doc_catalog:
            return ""
        
        # Step 1: Metadata filter
        candidates = self.doc_catalog
        
        if doc_type:
            candidates = [d for d in candidates if d["doc_type"].lower() == doc_type.lower()]
        
        if year:
            candidates = [d for d in candidates if d["year"] == year]
        
        # Step 2: If no metadata filter matched or we want semantic ranking too
        if self.doc_catalog_faiss and self.doc_catalog_bm25 and query:
            # 1. FAISS Search
            query_emb = self.encoder.encode([query], normalize_embeddings=True)
            query_emb = np.array(query_emb).astype('float32')
            k_faiss = min(semantic_top_k * 2, self.doc_catalog_faiss.ntotal)
            faiss_scores, faiss_indices = self.doc_catalog_faiss.search(query_emb, k_faiss)
            
            # 2. BM25 Search
            tokenized_query = self._tokenize(query)
            bm25_all_scores = self.doc_catalog_bm25.get_scores(tokenized_query)
            
            # 3. Reciprocal Rank Fusion (RRF)
            rrf_scores = {}
            k_rrf = 60
            
            # Accumulate FAISS rank
            for rank, idx in enumerate(faiss_indices[0]):
                if idx >= 0 and faiss_scores[0][rank] > 0.15:  # Basic sanity threshold
                    rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0.0) + 1.0 / (k_rrf + rank + 1)
            
            # Accumulate BM25 rank
            k_bm25 = min(semantic_top_k * 2, len(self.doc_catalog))
            bm25_indices = np.argsort(bm25_all_scores)[::-1][:k_bm25]
            for rank, idx in enumerate(bm25_indices):
                score = bm25_all_scores[idx]
                if score > 0:  # Only count if BM25 actually matches something
                    rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0.0) + 1.0 / (k_rrf + rank + 1)
            
            # Build set of semantically relevant catalog indices (sorted by RRF)
            sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
            semantic_set = set(sorted_indices[:semantic_top_k])
            
            if candidates == self.doc_catalog:
                # No metadata filter → use semantic results directly
                candidates = [self.doc_catalog[i] for i in sorted(semantic_set) 
                              if i < len(self.doc_catalog)]
            else:
                # Intersect metadata-filtered + semantic, but keep all metadata matches
                # Prioritize those in semantic_set
                candidate_set = {d["so_hieu"] for d in candidates}
                semantic_docs = [self.doc_catalog[i] for i in sorted(semantic_set)
                                 if i < len(self.doc_catalog) 
                                 and self.doc_catalog[i]["so_hieu"] in candidate_set]
                
                # If intersection is too small, just keep all metadata-filtered
                if len(semantic_docs) >= 3:
                    candidates = semantic_docs
        
        # Limit results
        candidates = candidates[:max_results]
        
        if not candidates:
            return ""
        
        # Format output
        lines = [f"=== KẾT QUẢ TÌM KIẾM VĂN BẢN ({len(candidates)} kết quả) ==="]
        for i, doc in enumerate(candidates, 1):
            summary_preview = doc['summary'].split('\n')[0][:150] if doc['summary'] else 'Không có tóm tắt'
            lines.append(f"\n{i}. {doc['source']}")
            lines.append(f"   Loại: {doc['doc_type']} | Năm: {doc['year']}")
            lines.append(f"   Nội dung: {summary_preview}")
        
        return "\n".join(lines)

    def _build_doc_faiss_map(self):
        """Build mapping: canonical doc → set of FAISS indices.
        Cho phép scoped_vector_search tra cứu trực tiếp."""
        self.doc_faiss_map = {}
        
        # Reverse mapping: node_id → faiss_idx
        rev_mapping = {}
        for idx, node_id in self.node_mapping.items():
            rev_mapping[node_id] = idx
        
        for canonical, node_ids in self.doc_nodes.items():
            faiss_indices = []
            for nid in node_ids:
                if nid in rev_mapping:
                    faiss_indices.append(rev_mapping[nid])
            if faiss_indices:
                self.doc_faiss_map[canonical] = faiss_indices
        
        total_mapped = sum(len(v) for v in self.doc_faiss_map.values())
        print(f"  Doc-FAISS Map: {len(self.doc_faiss_map)} documents, {total_mapped} article vectors")
    
    def scoped_vector_search(self, query: str, doc_so_hieu: str, top_k: int = 10) -> List[str]:
        """Vector search CHỈ trong phạm vi 1 document — không quét toàn bộ 20k+ nodes."""
        canonical = self.doc_registry.get(doc_so_hieu, doc_so_hieu)
        faiss_indices = self.doc_faiss_map.get(canonical, [])
        
        if not faiss_indices:
            return []
        
        # Lấy vectors của document từ FAISS
        indices_arr = sorted(faiss_indices)
        doc_vectors = np.array([
            self.faiss_index.reconstruct(int(i)) for i in indices_arr
        ]).astype('float32')
        
        # Encode query
        query_emb = self.encoder.encode([query], normalize_embeddings=True)
        query_emb = np.array(query_emb).astype('float32')
        
        # Cosine similarity trực tiếp trên subset
        scores = np.dot(doc_vectors, query_emb.T).flatten()
        actual_k = min(top_k, len(indices_arr))
        top_local = np.argsort(scores)[::-1][:actual_k]
        
        return [self.node_mapping[indices_arr[i]] for i in top_local if scores[i] > 0]

    def _build_vector_index(self):
        cache_dir = self.kg_path.parent / "index_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        faiss_path = cache_dir / "faiss.index"
        bm25_path = cache_dir / "bm25.pkl"
        mapping_path = cache_dir / "mapping.pkl"
        
        nodes_to_embed = []
        texts_to_embed = []
        
        for node_id, data in self.graph.nodes(data=True):
            # Embed all nodes that have search_text
            text = data.get("search_text", "")
            if text:
                nodes_to_embed.append(node_id)
                texts_to_embed.append(text)
                
        if not texts_to_embed:
            return
            
        # Check if cache exists and matches the number of nodes
        if faiss_path.exists() and bm25_path.exists() and mapping_path.exists():
            try:
                print(" -> Loading Cached Indexes...")
                self.faiss_index = faiss.read_index(str(faiss_path))
                with open(mapping_path, "rb") as f:
                    mappings = pickle.load(f)
                    self.node_mapping = mappings["node_mapping"]
                    self.bm25_nodes = mappings["bm25_nodes"]
                with open(bm25_path, "rb") as f:
                    self.bm25_index = pickle.load(f)
                    
                if len(self.node_mapping) == len(nodes_to_embed):
                    print(f"Loaded {len(self.node_mapping)} nodes from cache.")
                    return
                else:
                    print(" -> Data changed. Rebuilding index...")
            except Exception as e:
                print(f" -> Cache load failed: {e}. Rebuilding index...")
                
        print(" -> Generating Dense Embeddings (FAISS)...")
        embeddings = self.encoder.encode(texts_to_embed, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # recreate faiss index in case it was dirty
        self.faiss_index = faiss.IndexFlatIP(self.embed_dim)
        self.faiss_index.add(embeddings)
        self.node_mapping = {i: node_id for i, node_id in enumerate(nodes_to_embed)}
        
        print(" -> Generating Lexical Embeddings (BM25)...")
        tokenized_corpus = [self._tokenize(doc) for doc in texts_to_embed]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        self.bm25_nodes = nodes_to_embed
        
        print(f"Added {len(nodes_to_embed)} nodes to the Hybrid Index (FAISS + BM25).")
        
        # Save to cache
        print(" -> Saving Indexes to cache...")
        faiss.write_index(self.faiss_index, str(faiss_path))
        with open(mapping_path, "wb") as f:
            pickle.dump({
                "node_mapping": self.node_mapping,
                "bm25_nodes": self.bm25_nodes
            }, f)
        with open(bm25_path, "wb") as f:
            pickle.dump(self.bm25_index, f)

if __name__ == "__main__":
    import sys
    data_dir = "./data/final"
    kg_path = "./outputs/knowledge_graph/entity_graph.json"
    index = BookIndex(data_dir, kg_path)
    index.load_index()
    print("Done building index!")
