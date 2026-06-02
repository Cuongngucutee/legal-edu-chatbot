"""
LawEdu AI — Specialized Handlers.
Handle specific intent types: Summary, Listing, Statistical, Comparison.
All use Pro LLM API instead of local models.
"""
import json
import re
import logging

from app.rag.context_builder import build_context, postprocess_citations, fmt_source
from app.llm.prompts import (
    SUMMARY_PROMPT, LISTING_PROMPT, STATISTICAL_PROMPT,
    COMPARISON_PROMPT, COMPARISON_DECOMPOSE_PROMPT,
    GENERATION_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

from app.query.query_expander import EducationQueryExpander

class SummaryHandler:
    """Xử lý câu hỏi tóm tắt văn bản."""

    def handle(self, query, retriever, llm):
        expander = EducationQueryExpander()
        explicit_targets = expander.get_target_docs(query)
        
        docs = retriever.retrieve_as_docs(query, top_k=8)
        if not docs and not explicit_targets:
            return {"answer": "Không tìm thấy văn bản để tóm tắt.", "sources": [], "intent": "SUMMARY"}

        dominant = None
        doc_ids = {}
        if explicit_targets:
            dominant = explicit_targets[0]
            # Strip trailing slash if any, although get_target_docs returns exact names like '45/2021/TT-BGDĐT'
        else:
            # Focus on dominant document from retrieved chunks
            for d in docs:
                did = d.get("metadata", {}).get("doc_id", "")
                doc_ids[did] = doc_ids.get(did, 0) + 1
            dominant = max(doc_ids, key=doc_ids.get) if doc_ids else None
            
        # Check if we should override docs with full document chunks
        if dominant and hasattr(retriever, 'retrieve_by_doc_id'):
            if explicit_targets or (doc_ids and doc_ids.get(dominant, 0) >= 3):
                all_chunks = retriever.retrieve_by_doc_id(dominant)
                if all_chunks:
                    # Nhồi toàn bộ các điều luật của văn bản vào Context
                    docs = all_chunks

        # Mở khóa giới hạn ngữ cảnh cho GeneralHandler
        context = build_context(docs, max_chars=100000)
        answer = llm.generate(
            SUMMARY_PROMPT.format(query=query, context=context),
            system_prompt=GENERATION_SYSTEM_PROMPT,
        )
        answer = postprocess_citations(answer, docs)

        return {
            "answer": answer,
            "sources": [fmt_source(d) for d in docs[:5]],
            "context": context,
            "intent": "SUMMARY",
            "skill_log": [{"skill": "SUMMARY_HANDLER"}],
            "iterations": 1,
            "final_query": query,
        }


class StatisticalHandler:
    """Xử lý câu hỏi thống kê/đếm bằng metadata search."""

    def handle(self, query, retriever, llm):
        if hasattr(retriever, 'index') and hasattr(retriever.index, 'query_catalog'):
            q = query.lower()
            year = re.search(r'(20\d{2})', q)
            doc_type = None
            if "nghị định" in q:
                doc_type = "Nghị định"
            elif "thông tư" in q:
                doc_type = "Thông tư"
            elif "luật" in q:
                doc_type = "Luật"

            stats = retriever.index.query_catalog(
                query, doc_type=doc_type,
                year=year.group(1) if year else None,
            )
            if stats:
                answer = llm.generate(STATISTICAL_PROMPT.format(query=query, stats=stats))
                return {
                    "answer": answer, "sources": [], "context": str(stats), "intent": "STATISTICAL",
                    "skill_log": [{"skill": "STATISTICAL_HANDLER"}],
                    "iterations": 1, "final_query": query,
                }

        return {
            "answer": "Không tìm thấy dữ liệu thống kê.",
            "sources": [], "intent": "STATISTICAL",
            "skill_log": [], "iterations": 1, "final_query": query,
        }


class ListingHandler:
    """Xử lý câu hỏi liệt kê toàn bộ bằng iterative retrieval."""

    def handle(self, query, retriever, llm):
        all_docs = []
        seen = set()

        docs = retriever.retrieve_as_docs(query, top_k=10)
        for d in docs:
            cid = d.get("chunk_id", "")
            if cid not in seen:
                all_docs.append(d)
                seen.add(cid)

        logger.info(f"  📋 LISTING: {len(all_docs)} chunks")
        # Mở khóa giới hạn ngữ cảnh cho ComparisonHandler
        context = build_context(all_docs, max_chars=100000)
        answer = llm.generate(
            LISTING_PROMPT.format(query=query, context=context),
            system_prompt=GENERATION_SYSTEM_PROMPT,
        )
        answer = postprocess_citations(answer, all_docs)

        return {
            "answer": answer,
            "sources": [fmt_source(d) for d in all_docs[:8]],
            "context": context,
            "intent": "LISTING",
            "skill_log": [{"skill": "LISTING_HANDLER"}],
            "iterations": 1,
            "final_query": query,
        }


class ComparisonHandler:
    """Xử lý câu hỏi so sánh — retrieve riêng cho từng bên."""

    def handle(self, query, retriever, llm):
        all_docs = []
        seen = set()

        # Step 1: Decompose comparison into HyDE doc and separate queries
        result = llm.generate(COMPARISON_DECOMPOSE_PROMPT.format(query=query))
        sub_queries = [query]
        hyde_doc = ""
        try:
            match = re.search(r'\{.*\}', result, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                hyde_doc = parsed.get("hyde_document", "")
                subs = parsed.get("queries", [])
                if len(subs) >= 2:
                    sub_queries = subs
        except (json.JSONDecodeError, KeyError):
            pass

        logger.info(f"  ⚖️  COMPARISON: {len(sub_queries)} sub-queries, HyDE: {bool(hyde_doc)}")

        # Step 2: Retrieve using HyDE document
        if hyde_doc:
            for d in retriever.retrieve_as_docs(hyde_doc, top_k=6):
                cid = d.get("chunk_id", "")
                if cid not in seen:
                    all_docs.append(d)
                    seen.add(cid)

        # Step 3: Retrieve for each sub-query
        for sq in sub_queries[:4]:
            for d in retriever.retrieve_as_docs(sq, top_k=5):
                cid = d.get("chunk_id", "")
                if cid not in seen:
                    all_docs.append(d)
                    seen.add(cid)

        # Step 4: Also try original query
        for d in retriever.retrieve_as_docs(query, top_k=5):
            cid = d.get("chunk_id", "")
            if cid not in seen:
                all_docs.append(d)
                seen.add(cid)

        logger.info(f"  ⚖️  Total unique chunks: {len(all_docs)}")

        # Mở khóa giới hạn ngữ cảnh cho ProcessHandler
        context = build_context(all_docs, max_chars=100000)
        answer = llm.generate(
            COMPARISON_PROMPT.format(query=query, context=context),
            system_prompt=GENERATION_SYSTEM_PROMPT,
        )
        answer = postprocess_citations(answer, all_docs)

        return {
            "answer": answer,
            "sources": [fmt_source(d) for d in all_docs[:8]],
            "context": context,
            "intent": "COMPARISON",
            "skill_log": [{"skill": "COMPARISON_HANDLER"}],
            "iterations": 1,
            "final_query": query,
        }
