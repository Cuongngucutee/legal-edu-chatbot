import json
import os
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple

import faiss
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

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

        print("Building Vector Index...")
        self._build_vector_index()
        
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
                    if article_id in self.graph.nodes:
                        self.graph.nodes[article_id]['full_text'] = full_text
                        
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

    def _build_vector_index(self):
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
            
        embeddings = self.encoder.encode(texts_to_embed, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype('float32')
        
        self.faiss_index.add(embeddings)
        self.node_mapping = {i: node_id for i, node_id in enumerate(nodes_to_embed)}
        
        print(f"Added {len(nodes_to_embed)} nodes to the Vector Index.")

if __name__ == "__main__":
    import sys
    data_dir = "d:/legal-edu-chatbot/data/final"
    kg_path = "d:/legal-edu-chatbot/outputs/knowledge_graph/entity_graph.json"
    index = BookIndex(data_dir, kg_path)
    index.load_index()
    print("Done building index!")
