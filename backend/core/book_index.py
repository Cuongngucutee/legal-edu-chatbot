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
                    # Extract "43/2019/QH14" from "Luật 43/2019/QH14"
                    import re as _re
                    m = _re.search(r'(\d+[\/-]\d{4}[\/-]?[\w\-]*)', source)
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
                pass  # Skip silently
        
        # Bridge catalog doc_ids to corpus canonicals
        self._build_catalog_bridge()
        
        print(f"  Document Registry: {len(self.doc_registry)} entries → {len(self.doc_nodes)} documents")
    
    def _build_catalog_bridge(self):
        """Map catalog doc_id format sang file-based canonical format.
        
        Examples:
          catalog: "01/2014/TT-BGDĐT"    → corpus: "01/2014/TT-BGDĐT" (direct match)
          catalog: "Luật 43/2019/QH14"    → corpus via file stem "Luật_43_2019_QH14"
          catalog: "23/2024/TT-BGDĐT (Sửa đổi)" → clean "23/2024/TT-BGDĐT"
        """
        catalog_path = self.data_dir.parent / "outputs" / "document_catalog_final.json"
        if not catalog_path.exists():
            return
        
        try:
            with open(catalog_path, 'r', encoding='utf-8') as f:
                catalog = json.load(f)
        except Exception:
            return
        
        bridged = 0
        for entry in catalog:
            cat_doc_id = entry["doc_id"]
            
            # Already registered?
            if cat_doc_id in self.doc_registry:
                continue
            
            # Clean suffixes like "(Sửa đổi)", "(Cập nhật)"
            clean_id = re.sub(r'\s*\(.*?\)\s*', '', cat_doc_id).strip()
            
            # Try direct match with clean version
            if clean_id in self.doc_registry:
                self.doc_registry[cat_doc_id] = self.doc_registry[clean_id]
                self.doc_registry[cat_doc_id.lower()] = self.doc_registry[clean_id]
                bridged += 1
                continue
            
            # Try file stem matching: "01/2014/TT-BGDĐT" → "01_2014_TT_BGDĐT"
            stem = clean_id.replace('/', '_').replace('-', '_')
            for canonical in self.doc_nodes:
                canonical_stem = canonical.replace('/', '_').replace('-', '_')
                if stem == canonical_stem or stem.lower() == canonical_stem.lower():
                    self.doc_registry[cat_doc_id] = canonical
                    self.doc_registry[cat_doc_id.lower()] = canonical
                    self.doc_registry[clean_id] = canonical
                    self.doc_registry[clean_id.lower()] = canonical
                    bridged += 1
                    break
            else:
                # Handle "Luật XX/YYYY/QHZZ" format
                # Catalog: "Luật 43/2019/QH14" → file: "Luật_43_2019_QH14"
                if clean_id.startswith("Luật "):
                    luat_stem = "Luật_" + clean_id[5:].replace('/', '_').replace('-', '_')
                    for canonical in self.doc_nodes:
                        canonical_stem = canonical.replace('/', '_').replace('-', '_')
                        if luat_stem == canonical_stem or luat_stem.lower() == canonical_stem.lower():
                            self.doc_registry[cat_doc_id] = canonical
                            self.doc_registry[cat_doc_id.lower()] = canonical
                            self.doc_registry[clean_id] = canonical
                            self.doc_registry[clean_id.lower()] = canonical
                            # Also register without "Luật " prefix  
                            num_part = clean_id[5:]  # "43/2019/QH14"
                            if num_part not in self.doc_registry:
                                self.doc_registry[num_part] = canonical
                            bridged += 1
                            break
        
        if bridged > 0:
            print(f"  Catalog Bridge: {bridged} additional mappings registered")
    
    def get_doc_node_ids(self, so_hieu: str) -> set:
        """Get all node IDs belonging to a document by so_hieu."""
        canonical = self.doc_registry.get(so_hieu, so_hieu)
        return self.doc_nodes.get(canonical, set())
    
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
        import os
        import pickle
        import faiss
        
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
