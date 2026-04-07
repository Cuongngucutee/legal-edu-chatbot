import numpy as np
import networkx as nx
from typing import List, Dict, Set

from sentence_transformers import CrossEncoder
from book_index import BookIndex

class BookRAGRetriever:
    def __init__(self, book_index: BookIndex, cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.index = book_index
        print(f"Loading CrossEncoder: {cross_encoder_model}...")
        self.reranker = CrossEncoder(cross_encoder_model)

    def retrieve(self, query: str, query_type: str = "tong_hop", top_k: int = 5, max_hops: int = 1) -> str:
        # Tùy chỉnh tham số Hybrid Search theo Intent (chiến lược Adaptive)
        if query_type == "tra_cuu":
            faiss_k = top_k
            bm25_k = top_k * 2  # Trọng số lexcial cao hơn vì tra cứu cần chuẩn xác từ khóa
            max_hops = max(1, max_hops)
        elif query_type == "so_sanh":
            faiss_k = top_k * 2
            bm25_k = top_k
            max_hops = max(2, max_hops)  # Cần liên kết rộng hơn qua Node đồ thị
        elif query_type == "thu_tuc":
            faiss_k = top_k
            bm25_k = top_k
            max_hops = max(1, max_hops)
        else: # tong_hop
            faiss_k = top_k * 2
            bm25_k = top_k
            max_hops = max(1, max_hops)

        # 1. Semantic FAISS Retrieval
        query_emb = self.index.encoder.encode([query], normalize_embeddings=True)
        query_emb = np.array(query_emb).astype('float32')
        _, indices = self.index.faiss_index.search(query_emb, faiss_k)
        faiss_nodes = [self.index.node_mapping[idx] for idx in indices[0] if idx != -1 and idx in self.index.node_mapping]
        
        # 2. Lexical BM25 Retrieval
        tokenized_query = self.index._tokenize(query)
        bm25_nodes = []
        if self.index.bm25_index is not None:
            bm25_scores = self.index.bm25_index.get_scores(tokenized_query)
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:bm25_k]
            bm25_nodes = [self.index.bm25_nodes[idx] for idx in top_bm25_indices if bm25_scores[idx] > 0]
                
        # Combine Entry Nodes
        entry_nodes = list(set(faiss_nodes + bm25_nodes))
        
        # 3. Graph Traversal to expand context
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
            
        # 4. CrossEncoder Reranking
        articles_and_clauses = [n for n in expanded_nodes if str(n).startswith("node:")]
        candidate_pairs = []
        node_contexts = []
        for node in articles_and_clauses:
            text = self.index.graph.nodes[node].get("full_text") or self.index.graph.nodes[node].get("text", "")
            if text:
                candidate_pairs.append([query, text])
                node_contexts.append(node)
                
        if not candidate_pairs:
            return ""

        scores = self.reranker.predict(candidate_pairs)
        ranked_indices = np.argsort(scores)[::-1]
        
        # Giữ lại Top N có điểm cao nhất để gừi LLM (tránh nhiễu context)
        final_top = min(top_k * 2, len(ranked_indices))
        selected_nodes = [node_contexts[i] for i in ranked_indices[:final_top]]

        return self._format_context(set(selected_nodes))

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
