import re
import numpy as np
import networkx as nx
from typing import List, Dict, Set, Optional

from sentence_transformers import CrossEncoder
from book_index import BookIndex
from query_intent import QueryIntent, intent_to_legacy_type

# Patterns nhận diện câu hỏi tóm tắt/tổng quan về văn bản
_SUMMARY_PATTERNS = [
    r"nói về (gì|điều gì|cái gì|vấn đề gì)",
    r"quy định (gì|những gì|về gì|về vấn đề gì)",
    r"gồm (những gì|bao nhiêu|mấy)",
    r"bao gồm (những gì|nội dung gì)",
    r"tóm tắt",
    r"nội dung (chính|chủ yếu|tổng quan|cơ bản)",
    r"có (bao nhiêu|mấy) (chương|điều|mục|phần)",
    r"tổng quan",
    r"giới thiệu",
    r"cấu trúc",
    r"phạm vi điều chỉnh",
]

# Patterns nhận diện câu hỏi thống kê/liệt kê văn bản
_STAT_PATTERNS = [
    r"những (nghị định|thông tư|luật|văn bản|quy định) nào",
    r"liệt kê.*(nghị định|thông tư|luật|văn bản)",
    r"có (bao nhiêu|mấy) (nghị định|thông tư|luật|văn bản)",
    r"danh sách.*(nghị định|thông tư|luật|văn bản)",
    r"(nghị định|thông tư|luật|văn bản) nào.*(liên quan|nói về|quy định về|đề cập)",
    r"(năm|trong năm)\s+20\d{2}.*(nghị định|thông tư|luật|văn bản|ban hành|ra)",
    r"(nghị định|thông tư|luật|văn bản).*(năm|trong năm)\s+20\d{2}",
    r"ban hành.*(năm|trong năm)\s+20\d{2}",
    r"(ra|được ban hành).*(năm|trong)\s+20\d{2}",
]


class BookRAGRetriever:
    def __init__(self, book_index: BookIndex, cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.index = book_index
        print(f"Loading CrossEncoder: {cross_encoder_model}...")
        self.reranker = CrossEncoder(cross_encoder_model)

    # ─── Public API ───────────────────────────────────────────────────────

    def retrieve(self, query: str, intent: QueryIntent = None, query_type: str = "tong_hop", top_k: int = 5, max_hops: int = 1) -> str:
        """
        Main retrieval entry point.
        
        Nếu có intent (QueryIntent) → dùng chiến lược nâng cao theo type.
        Nếu không có intent → fallback về logic cũ dùng query_type string.
        """
        if intent is not None:
            return self._retrieve_with_intent(query, intent, top_k)
        else:
            return self._retrieve_legacy(query, query_type, top_k, max_hops)

    # ─── Intent-based Retrieval (MỚI) ────────────────────────────────────

    def _retrieve_with_intent(self, query: str, intent: QueryIntent, top_k: int = 5) -> str:
        """Dispatch retrieval strategy theo QueryIntent.type."""
        
        strategy_map = {
            "single_lookup":     self._retrieve_single_lookup,
            "comparison":        self._retrieve_comparison,
            "listing":           self._retrieve_listing,
            "eligibility_check": self._retrieve_eligibility,
            "procedure":         self._retrieve_procedure,
            "cross_document":    self._retrieve_cross_document,
            "definition":        self._retrieve_definition,
        }
        
        strategy = strategy_map.get(intent.type, self._retrieve_default)
        result = strategy(query, intent, top_k)
        
        # Fallback: nếu context rỗng và có documents → thử global search
        if (not result or not result.strip()) and intent.documents:
            print(f"[Retriever] Strategy '{intent.type}' returned empty. Trying global fallback...")
            all_nodes = self._global_retrieve(query, faiss_k=top_k * 3, bm25_k=top_k * 2)
            if all_nodes:
                expanded = self._expand_graph(set(all_nodes), hops=1)
                selected = self._rerank_select(query, expanded, top_k * 2, boost_keywords=intent.keywords)
                result = self._format_context(selected)
        
        # TOC Injection: nếu có document cụ thể + câu hỏi dạng tóm tắt
        # → prepend mục lục vào context (retrieval vẫn chạy bình thường)
        if intent.documents and self._is_summary_query(query):
            toc_parts = []
            for doc_ref in intent.documents:
                toc = self.index.build_toc(doc_ref)
                if toc:
                    toc_parts.append(toc)
            if toc_parts:
                toc_context = "\n\n".join(toc_parts)
                print(f"[Retriever] TOC injected for {len(toc_parts)} document(s)")
                result = toc_context + "\n\n" + (result or "")
        
        # Catalog Injection: câu hỏi thống kê/liệt kê văn bản
        # → truy vấn Document Catalog (metadata + semantic), prepend vào context
        if self._is_statistical_query(query):
            year_filter = self._extract_year_filter(query)
            doc_type_filter = self._extract_doc_type_filter(query)
            
            catalog_result = self.index.query_catalog(
                query=query,
                doc_type=doc_type_filter,
                year=year_filter,
            )
            if catalog_result:
                print(f"[Retriever] Catalog injected (year={year_filter}, type={doc_type_filter})")
                result = catalog_result + "\n\n" + (result or "")
        
        return result

    @staticmethod
    def _is_summary_query(query: str) -> bool:
        """Kiểm tra câu hỏi có phải dạng tóm tắt/tổng quan không."""
        query_lower = query.lower()
        return any(re.search(p, query_lower) for p in _SUMMARY_PATTERNS)

    @staticmethod
    def _is_statistical_query(query: str) -> bool:
        """Kiểm tra câu hỏi có phải dạng thống kê/liệt kê văn bản không."""
        query_lower = query.lower()
        return any(re.search(p, query_lower) for p in _STAT_PATTERNS)

    @staticmethod
    def _extract_year_filter(query: str) -> str:
        """Trích xuất năm từ câu hỏi (nếu có)."""
        m = re.search(r'(20\d{2})', query)
        return m.group(1) if m else None

    @staticmethod
    def _extract_doc_type_filter(query: str) -> str:
        """Trích xuất loại văn bản từ câu hỏi (nếu có)."""
        query_lower = query.lower()
        if "nghị định" in query_lower:
            return "Nghị định"
        elif "thông tư" in query_lower:
            return "Thông tư"
        elif "luật" in query_lower:
            return "Luật"
        return None

    # ── Multi-Query Retrieval (Tận dụng sub_queries) ─────────────────────

    def _multi_query_retrieve(self, query: str, intent: QueryIntent,
                               faiss_k: int = 10, bm25_k: int = 10,
                               scoped: bool = True) -> List[str]:
        """
        Tìm kiếm song song với từng sub_query (nếu có) rồi merge kết quả.
        
        Luật:
        - Nếu sub_queries chỉ có 1 → search 2 lần: sub_query + topic, rồi merge
        - Nếu sub_queries >= 2 → chạy riêng từng sub_query
        """
        sub_queries = intent.sub_queries or [query]
        # Deduplicate sub_queries
        seen_sq = set()
        unique_sqs = []
        for sq in sub_queries:
            sq_lower = sq.strip().lower()
            if sq_lower and sq_lower not in seen_sq:
                seen_sq.add(sq_lower)
                unique_sqs.append(sq)
        if not unique_sqs:
            unique_sqs = [query]
        
        all_nodes = []
        
        if len(unique_sqs) == 1:
            # ── 1 sub_query → search 2 lần: sub_query + topic, rồi merge ──
            sub_q = unique_sqs[0]
            topic = (intent.topic or "").strip()
            
            # Lần 1: Search bằng sub_query
            if scoped and intent.documents:
                for doc in intent.documents:
                    nodes = self._scoped_retrieve(sub_q, doc, faiss_k=faiss_k, bm25_k=bm25_k)
                    all_nodes.extend(nodes)
            else:
                nodes = self._global_retrieve(sub_q, faiss_k=faiss_k, bm25_k=bm25_k)
                all_nodes.extend(nodes)
            
            # Lần 2: Search bằng topic (nếu topic khác sub_query)
            if topic and topic.lower() != sub_q.strip().lower():
                if scoped and intent.documents:
                    for doc in intent.documents:
                        nodes = self._scoped_retrieve(topic, doc, faiss_k=faiss_k, bm25_k=bm25_k)
                        all_nodes.extend(nodes)
                else:
                    nodes = self._global_retrieve(topic, faiss_k=faiss_k, bm25_k=bm25_k)
                    all_nodes.extend(nodes)
        else:
            # ── Nhiều sub_queries → chạy từng sub_query riêng, KHÔNG thêm query gốc ──
            per_query_faiss = max(3, faiss_k // len(unique_sqs))
            per_query_bm25 = max(3, bm25_k // len(unique_sqs))
            
            for sq in unique_sqs:
                if scoped and intent.documents:
                    for doc in intent.documents:
                        nodes = self._scoped_retrieve(sq, doc, faiss_k=per_query_faiss, bm25_k=per_query_bm25)
                        all_nodes.extend(nodes)
                else:
                    nodes = self._global_retrieve(sq, faiss_k=per_query_faiss, bm25_k=per_query_bm25)
                    all_nodes.extend(nodes)
        
        return list(set(all_nodes))

    # ── Strategy: single_lookup ───────────────────────────────────────────

    def _retrieve_single_lookup(self, query: str, intent: QueryIntent, top_k: int) -> str:
        """Tra cứu 1 điều khoản cụ thể. Nếu có article_hint → direct lookup."""
        
        # Nếu có article_hint + document → direct lookup (zero latency)
        if intent.article_hint and intent.documents:
            context = self._article_direct_lookup(intent.documents[0], intent.article_hint)
            if context:
                return context
        
        # Multi-query retrieval với sub_queries
        all_nodes = self._multi_query_retrieve(query, intent, faiss_k=top_k, bm25_k=top_k * 2, scoped=bool(intent.documents))
        
        if all_nodes:
            expanded = self._expand_graph(set(all_nodes), hops=1)
            selected = self._rerank_select(query, expanded, top_k * 2, boost_keywords=intent.keywords)
            return self._format_context(selected)
        
        # Không có document → standard search
        return self._retrieve_default(query, intent, top_k)

    # ── Strategy: comparison ──────────────────────────────────────────────

    def _retrieve_comparison(self, query: str, intent: QueryIntent, top_k: int) -> str:
        """So sánh giữa 2+ văn bản. Parallel scoped search + balanced merge."""
        
        # Deduplicate documents (Qwen đôi khi trả duplicate)
        seen_docs = set()
        unique_docs = []
        for doc in intent.documents:
            canonical = self.index.doc_registry.get(doc, doc)
            if canonical not in seen_docs:
                seen_docs.add(canonical)
                unique_docs.append(doc)
        
        if len(unique_docs) < 2:
            # Không đủ 2 docs → fallback default với mở rộng
            return self._retrieve_default(query, intent, top_k * 2)
        
        # Parallel scoped search cho mỗi document
        per_doc_nodes = {}
        sub_queries = intent.sub_queries or [query] * len(unique_docs)
        
        for i, doc in enumerate(unique_docs):
            sub_q = sub_queries[i] if i < len(sub_queries) else query
            nodes = self._scoped_retrieve(sub_q, doc, faiss_k=top_k + 3, bm25_k=top_k)
            # Giảm từ 2 hops → 1 hop (FAISS đã có article vectors trực tiếp)
            expanded = self._expand_graph(set(nodes), hops=1)
            per_doc_nodes[doc] = expanded
        
        # Balanced merge: đảm bảo mỗi doc có ít nhất min_per_doc slots
        min_per_doc = max(3, top_k // len(unique_docs))
        selected = self._balanced_rerank(query, per_doc_nodes, min_per_doc, total_k=top_k * 2, boost_keywords=intent.keywords)
        
        return self._format_context(selected)

    # ── Strategy: listing ─────────────────────────────────────────────────

    def _retrieve_listing(self, query: str, intent: QueryIntent, top_k: int) -> str:
        """Liệt kê đối tượng/điều kiện. Multi-query + MMR diversity."""
        
        effective_k = top_k * 3 if intent.expand_search else top_k * 2
        
        # Multi-query: tận dụng sub_queries để cover nhiều khía cạnh
        all_nodes = self._multi_query_retrieve(query, intent, faiss_k=effective_k, bm25_k=effective_k, scoped=bool(intent.documents))
        
        expanded = self._expand_graph(set(all_nodes), hops=1)
        # Rerank + MMR cho diversity
        selected = self._rerank_select(query, expanded, effective_k, boost_keywords=intent.keywords, use_mmr=True)
        return self._format_context(selected)

    # ── Strategy: eligibility_check ───────────────────────────────────────

    def _retrieve_eligibility(self, query: str, intent: QueryIntent, top_k: int) -> str:
        """Kiểm tra điều kiện. Multi-query + check_aspects."""
        
        # Multi-query với sub_queries (bao gồm câu hỏi gốc)
        all_nodes = set(self._multi_query_retrieve(query, intent, faiss_k=top_k, bm25_k=top_k, scoped=bool(intent.documents)))
        
        # Search bổ sung cho từng check_aspect
        if intent.check_aspects:
            for aspect in intent.check_aspects[:3]:  # Max 3 aspects
                aspect_query = f"{intent.topic} {aspect}" if intent.topic else f"{query} {aspect}"
                if intent.documents:
                    for doc in intent.documents:
                        nodes = self._scoped_retrieve(aspect_query, doc, faiss_k=3, bm25_k=3)
                        all_nodes.update(nodes)
                else:
                    nodes = self._global_retrieve(aspect_query, faiss_k=3, bm25_k=3)
                    all_nodes.update(nodes)
        
        expanded = self._expand_graph(all_nodes, hops=1)
        selected = self._rerank_select(query, expanded, top_k * 2, boost_keywords=intent.keywords)
        return self._format_context(selected)

    # ── Strategy: procedure ───────────────────────────────────────────────

    def _retrieve_procedure(self, query: str, intent: QueryIntent, top_k: int) -> str:
        """Thủ tục/quy trình. Multi-query + BM25 boost procedure_keywords."""
        
        # Multi-query với sub_queries, BM25 trọng số cao hơn cho thuật ngữ thủ tục
        all_nodes = self._multi_query_retrieve(query, intent, faiss_k=top_k, bm25_k=top_k * 3, scoped=bool(intent.documents))
        
        expanded = self._expand_graph(set(all_nodes), hops=1)
        boost_kw = (intent.procedure_keywords or []) + intent.keywords
        selected = self._rerank_select(query, expanded, top_k * 2, boost_keywords=boost_kw)
        return self._format_context(selected)

    # ── Strategy: cross_document ──────────────────────────────────────────

    def _retrieve_cross_document(self, query: str, intent: QueryIntent, top_k: int) -> str:
        """Tìm kiếm xuyên văn bản. Multi-query để cover nhiều khía cạnh."""
        
        # Multi-query: mỗi sub_query tìm riêng rồi merge
        all_nodes = self._multi_query_retrieve(
            query, intent,
            faiss_k=top_k * 3, bm25_k=top_k * 2,
            scoped=bool(intent.documents)
        )
        
        if all_nodes:
            expanded = self._expand_graph(set(all_nodes), hops=1)
            selected = self._rerank_select(query, expanded, top_k * 2, boost_keywords=intent.keywords, use_mmr=True)
            result = self._format_context(selected)
            if result.strip():
                return result
        
        # Fallback: Global search không dùng sub_queries
        all_nodes = self._global_retrieve(query, faiss_k=top_k * 3, bm25_k=top_k * 2)
        expanded = self._expand_graph(set(all_nodes), hops=1)
        selected = self._rerank_select(query, expanded, top_k * 2, boost_keywords=intent.keywords)
        return self._format_context(selected)

    # ── Strategy: definition ──────────────────────────────────────────────

    def _retrieve_definition(self, query: str, intent: QueryIntent, top_k: int) -> str:
        """Tra cứu định nghĩa. Multi-query + BM25 boost cho term (exact match)."""
        
        boost_kw = intent.keywords.copy()
        if intent.term:
            boost_kw.insert(0, intent.term)
        
        # Multi-query retrieval với sub_queries
        all_nodes = self._multi_query_retrieve(
            query, intent,
            faiss_k=top_k, bm25_k=top_k * 2,
            scoped=bool(intent.documents)
        )
        
        if all_nodes:
            expanded = self._expand_graph(set(all_nodes), hops=1)
            selected = self._rerank_select(query, expanded, top_k * 2, boost_keywords=boost_kw)
            result = self._format_context(selected)
            if result.strip():
                return result
        
        # Fallback: Global search
        all_nodes = self._global_retrieve(query, faiss_k=top_k, bm25_k=top_k * 2)
        expanded = self._expand_graph(set(all_nodes), hops=1)
        selected = self._rerank_select(query, expanded, top_k * 2, boost_keywords=boost_kw)
        return self._format_context(selected)

    # ── Strategy: default fallback ────────────────────────────────────────

    def _retrieve_default(self, query: str, intent: QueryIntent, top_k: int) -> str:
        """Fallback strategy: scoped nếu có docs, global nếu không."""
        if intent.documents:
            # Có docs (từ regex hoặc Qwen) → scoped search
            all_nodes = []
            for doc in intent.documents:
                nodes = self._scoped_retrieve(query, doc, faiss_k=top_k * 2, bm25_k=top_k * 2)
                all_nodes.extend(nodes)
            if all_nodes:
                expanded = self._expand_graph(set(all_nodes), hops=1)
                selected = self._rerank_select(query, expanded, top_k * 2, boost_keywords=intent.keywords)
                result = self._format_context(selected)
                if result.strip():
                    return result
        
        # Fallback global
        all_nodes = self._global_retrieve(query, faiss_k=top_k * 2, bm25_k=top_k)
        expanded = self._expand_graph(set(all_nodes), hops=1)
        selected = self._rerank_select(query, expanded, top_k * 2, boost_keywords=intent.keywords if intent else [])
        return self._format_context(selected)
    


    # ─── Core Retrieval Building Blocks ───────────────────────────────────

    def _global_retrieve(self, query: str, faiss_k: int = 10, bm25_k: int = 10) -> List[str]:
        """FAISS + BM25 search trên toàn corpus (không filter)."""
        # FAISS
        query_emb = self.index.encoder.encode([query], normalize_embeddings=True)
        query_emb = np.array(query_emb).astype('float32')
        _, indices = self.index.faiss_index.search(query_emb, faiss_k)
        faiss_nodes = [self.index.node_mapping[idx] for idx in indices[0] if idx != -1 and idx in self.index.node_mapping]
        
        # BM25
        tokenized_query = self.index._tokenize(query)
        bm25_nodes = []
        if self.index.bm25_index is not None:
            bm25_scores = self.index.bm25_index.get_scores(tokenized_query)
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:bm25_k]
            bm25_nodes = [self.index.bm25_nodes[idx] for idx in top_bm25_indices if bm25_scores[idx] > 0]
        
        return list(set(faiss_nodes + bm25_nodes))

    def _scoped_retrieve(self, query: str, doc_so_hieu: str, faiss_k: int = 10, bm25_k: int = 10) -> List[str]:
        """FAISS + BM25 search chỉ trong phạm vi 1 document.
        
        FAISS: dùng scoped_vector_search (trực tiếp trên subset vectors của doc).
        BM25: vẫn dùng global search + filter (BM25 đã nhanh).
        """
        # Lấy set node IDs thuộc document này
        doc_node_ids = self.index.get_doc_node_ids(doc_so_hieu)
        
        if not doc_node_ids:
            # Nếu không tìm thấy doc → fallback global
            return self._global_retrieve(query, faiss_k, bm25_k)
        
        # FAISS: search TRỰC TIẾP trong document (không quét global)
        faiss_nodes = self.index.scoped_vector_search(query, doc_so_hieu, top_k=faiss_k)
        
        # BM25: search rộng rồi filter (BM25 rất nhẹ, giữ nguyên)
        tokenized_query = self.index._tokenize(query)
        bm25_nodes = []
        if self.index.bm25_index is not None:
            bm25_scores = self.index.bm25_index.get_scores(tokenized_query)
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:bm25_k * 5]
            for idx in top_bm25_indices:
                if bm25_scores[idx] <= 0:
                    break
                node_id = self.index.bm25_nodes[idx]
                if node_id in doc_node_ids:
                    bm25_nodes.append(node_id)
                    if len(bm25_nodes) >= bm25_k:
                        break
        
        return list(set(faiss_nodes + bm25_nodes))

    def _article_direct_lookup(self, doc_so_hieu: str, article_num: str) -> Optional[str]:
        """Direct lookup: tìm node Điều X trong document, bỏ qua vector search."""
        doc_node_ids = self.index.get_doc_node_ids(doc_so_hieu)
        
        # Tìm node có article_number matching
        target_suffix = f"_D{article_num}"
        for node_id in doc_node_ids:
            if node_id.upper().endswith(target_suffix.upper()):
                node_data = self.index.graph.nodes.get(node_id, {})
                content = node_data.get("full_text") or node_data.get("text", "")
                if content:
                    return self._format_context({node_id})
        
        # Thử tìm theo pattern d{num}
        for node_id in doc_node_ids:
            if str(node_id).endswith(f":d{article_num}") or str(node_id).endswith(f"_D{article_num}"):
                node_data = self.index.graph.nodes.get(node_id, {})
                content = node_data.get("full_text") or node_data.get("text", "")
                if content:
                    return self._format_context({node_id})
        
        return None  # Không tìm thấy → fallback

    # ─── Graph Expansion ──────────────────────────────────────────────────

    def _expand_graph(self, entry_nodes: Set[str], hops: int = 1) -> Set[str]:
        """Mở rộng nodes qua KG graph."""
        expanded = set(entry_nodes)
        current_layer = set(entry_nodes)
        
        for _ in range(hops):
            next_layer = set()
            for node in current_layer:
                if node in self.index.graph:
                    neighbors = list(self.index.graph.neighbors(node))
                    for neighbor in neighbors:
                        if neighbor not in expanded:
                            next_layer.add(neighbor)
                            expanded.add(neighbor)
            current_layer = next_layer
        
        return expanded

    # ─── Reranking ────────────────────────────────────────────────────────

    def _rerank_select(self, query: str, candidates: Set[str], top_n: int,
                       boost_keywords: List[str] = None, use_mmr: bool = False) -> Set[str]:
        """CrossEncoder reranking + keyword boost + optional MMR."""
        articles_and_clauses = [n for n in candidates if str(n).startswith("node:")]
        candidate_pairs = []
        node_contexts = []
        
        for node in articles_and_clauses:
            text = self.index.graph.nodes.get(node, {}).get("full_text") or \
                   self.index.graph.nodes.get(node, {}).get("text", "")
            if text:
                # Truncate để tránh OOM
                candidate_pairs.append([query, text[:1024]])
                node_contexts.append((node, text))
        
        if not candidate_pairs:
            return set()
        
        # CrossEncoder scores
        scores = self.reranker.predict(candidate_pairs)
        
        # Keyword boost
        if boost_keywords:
            for i, (node, text) in enumerate(node_contexts):
                text_lower = text.lower()
                for kw in boost_keywords:
                    if kw.lower() in text_lower:
                        scores[i] *= 1.2
                        break
        
        if use_mmr:
            selected = self._mmr_select(node_contexts, scores, top_n)
        else:
            ranked_indices = np.argsort(scores)[::-1]
            final_top = min(top_n, len(ranked_indices))
            selected = {node_contexts[i][0] for i in ranked_indices[:final_top]}
        
        return selected

    def _balanced_rerank(self, query: str, per_doc_nodes: Dict[str, Set[str]],
                         min_per_doc: int, total_k: int,
                         boost_keywords: List[str] = None) -> Set[str]:
        """Balanced reranking: đảm bảo mỗi doc có ít nhất min_per_doc slots."""
        selected = set()
        overflow = []
        
        for doc, candidates in per_doc_nodes.items():
            articles = [n for n in candidates if str(n).startswith("node:")]
            pairs = []
            nodes = []
            for node in articles:
                text = self.index.graph.nodes.get(node, {}).get("full_text") or \
                       self.index.graph.nodes.get(node, {}).get("text", "")
                if text:
                    pairs.append([query, text[:1024]])
                    nodes.append(node)
            
            if not pairs:
                continue
            
            scores = self.reranker.predict(pairs)
            
            # Keyword boost
            if boost_keywords:
                for i, node in enumerate(nodes):
                    text = (self.index.graph.nodes.get(node, {}).get("full_text") or "").lower()
                    for kw in boost_keywords:
                        if kw.lower() in text:
                            scores[i] *= 1.3
                            break
            
            ranked = np.argsort(scores)[::-1]
            
            # Guaranteed slots
            for idx in ranked[:min_per_doc]:
                selected.add(nodes[idx])
            
            # Overflow cho remaining
            for idx in ranked[min_per_doc:]:
                overflow.append((nodes[idx], scores[idx]))
        
        # Fill remaining slots from overflow (sorted by score)
        overflow.sort(key=lambda x: x[1], reverse=True)
        remaining = total_k - len(selected)
        for node, _ in overflow[:remaining]:
            selected.add(node)
        
        return selected

    def _mmr_select(self, node_contexts: list, scores: np.ndarray,
                    top_n: int, lambda_param: float = 0.7) -> Set[str]:
        """Maximal Marginal Relevance selection cho diversity."""
        if len(node_contexts) == 0:
            return set()
        
        selected_indices = []
        remaining = list(range(len(scores)))
        
        # Greedy MMR
        for _ in range(min(top_n, len(remaining))):
            if not remaining:
                break
            
            if not selected_indices:
                # Đầu tiên chọn score cao nhất
                best = max(remaining, key=lambda i: scores[i])
            else:
                # MMR: balance relevance vs diversity (đơn giản dùng text overlap)
                best = None
                best_mmr = float('-inf')
                for i in remaining:
                    relevance = scores[i]
                    # Diversity: check text similarity (simple word overlap)
                    max_sim = 0
                    text_i = node_contexts[i][1].lower()
                    words_i = set(text_i.split())
                    for j in selected_indices:
                        text_j = node_contexts[j][1].lower()
                        words_j = set(text_j.split())
                        overlap = len(words_i & words_j) / max(len(words_i | words_j), 1)
                        max_sim = max(max_sim, overlap)
                    
                    mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                    if mmr > best_mmr:
                        best_mmr = mmr
                        best = i
            
            selected_indices.append(best)
            remaining.remove(best)
        
        return {node_contexts[i][0] for i in selected_indices}

    # ─── Context Formatting ───────────────────────────────────────────────

    def _format_context(self, nodes: Set[str]) -> str:
        """Construct hierarchical context from selected nodes."""
        articles_and_clauses = [n for n in nodes if str(n).startswith("node:")]
        
        structured_contexts = []
        
        for node in articles_and_clauses:
            if node not in self.index.graph.nodes:
                continue
            node_data = self.index.graph.nodes[node]
            
            # Find ancestors (Document, Chapter, Section) structurally
            hierarchy = []
            
            # Lấy Văn bản chứa node này
            docs = [n for n in self.index.graph.neighbors(node) if self.index.graph.edges[node, n].get("type") == "THUOC_VAN_BAN"]
            if docs:
                doc_node_data = self.index.graph.nodes[docs[0]]
                doc_name = doc_node_data.get("name") or doc_node_data.get("search_text", docs[0])
                hierarchy.append(doc_name)
            else:
                hierarchy.append("Văn bản không xác định")
                
            # Lấy Chương
            chaps = [n for n in self.index.graph.neighbors(node) if self.index.graph.edges[node, n].get("type") == "THUOC_CHUONG"]
            if chaps:
                hierarchy.append(self.index.graph.nodes[chaps[0]].get("name", chaps[0]))
                
            # Lấy Mục (nếu có)
            secs = [n for n in self.index.graph.neighbors(node) if self.index.graph.edges[node, n].get("type") == "THUOC_MUC"]
            if secs:
                hierarchy.append(self.index.graph.nodes[secs[0]].get("name", secs[0]))
                    
            content = node_data.get("full_text") or node_data.get("text", "")
            if not content:
                continue
                
            path_str = " -> ".join(hierarchy)
            title = node_data.get("name") or node_data.get("search_text", node)
            
            structured_contexts.append(f"[{path_str} -> {title}]\n{content}\n")
            
        return "\n".join(set(structured_contexts))  # deduplicate

    # ─── Legacy API (backward compatibility) ──────────────────────────────

    def _retrieve_legacy(self, query: str, query_type: str = "tong_hop", top_k: int = 5, max_hops: int = 1) -> str:
        """Legacy retrieve — giữ nguyên logic cũ cho backward compatibility."""
        # Tùy chỉnh tham số Hybrid Search theo Intent (chiến lược Adaptive)
        if query_type == "tra_cuu":
            faiss_k = top_k
            bm25_k = top_k * 2  # Trọng số lexical cao hơn vì tra cứu cần chuẩn xác từ khóa
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
        
        # Giữ lại Top N có điểm cao nhất để gửi LLM (tránh nhiễu context)
        final_top = min(top_k * 2, len(ranked_indices))
        selected_nodes = [node_contexts[i] for i in ranked_indices[:final_top]]

        return self._format_context(set(selected_nodes))
