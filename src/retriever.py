import numpy as np
import networkx as nx
from typing import List, Dict, Set

from book_index import BookIndex

class BookRAGRetriever:
    def __init__(self, book_index: BookIndex):
        self.index = book_index

    def retrieve(self, query: str, top_k: int = 3, max_hops: int = 1) -> str:
        # Step 1: Embed query
        query_emb = self.index.encoder.encode([query], normalize_embeddings=True)
        query_emb = np.array(query_emb).astype('float32')
        
        # Step 2: Search FAISS index for Entry Nodes
        distances, indices = self.index.faiss_index.search(query_emb, top_k)
        
        entry_nodes = []
        for idx in indices[0]:
            if idx != -1 and idx in self.index.node_mapping:
                entry_nodes.append(self.index.node_mapping[idx])
                
        # Step 3: Graph Traversal to expand context
        expanded_nodes = set(entry_nodes)
        current_layer = set(entry_nodes)
        
        for _ in range(max_hops):
            next_layer = set()
            for node in current_layer:
                neighbors = list(self.index.graph.neighbors(node))
                for neighbor in neighbors:
                    if neighbor not in expanded_nodes:
                        next_layer.add(neighbor)
                        expanded_nodes.add(neighbor)
            current_layer = next_layer
            
        return self._format_context(expanded_nodes)

    def _format_context(self, nodes: Set[str]) -> str:
        # Step 4: Construct hierarchical context
        # We want to present Articles and Clauses with their parent hierarchy
        articles_and_clauses = [n for n in nodes if str(n).startswith("node:")]
        
        structured_contexts = []
        
        for node in articles_and_clauses:
            node_data = self.index.graph.nodes[node]
            node_type = node_data.get("type", "Node")
            
            # Find ancestors (Chapter, Section, Document) structurally
            hierarchy = []
            
            chaps = [n for n in self.index.graph.neighbors(node) if self.index.graph.edges[node, n].get("type") == "THUOC_CHUONG"]
            if chaps:
                hierarchy.append(self.index.graph.nodes[chaps[0]].get("name", chaps[0]))
                    
            content = node_data.get("full_text") or node_data.get("text", "")
            if not content:
                continue
                
            path_str = " -> ".join(reversed(hierarchy)) if hierarchy else "Document"
            title = node_data.get("name") or node_data.get("search_text", node)
            
            structured_contexts.append(f"[{path_str} -> {title}]\n{content}\n")
            
        return "\n".join(set(structured_contexts)) # deduplicate
