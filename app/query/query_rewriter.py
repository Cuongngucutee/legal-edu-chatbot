"""
LawEdu AI — Query Rewriter.
Uses 320B LLM to rewrite queries into legal terminology.
"""
import json
import re
import logging

logger = logging.getLogger(__name__)


class QueryRewriteSkill:
    """Viết lại câu hỏi bằng thuật ngữ pháp lý chuẩn (320B API)."""

    def apply(self, query, docs, llm, retriever=None):
        from app.llm.prompts import QUERY_REWRITE_PROMPT
        new_q = llm.generate(QUERY_REWRITE_PROMPT.format(query=query)).strip()
        new_q = new_q.strip('"').strip("'").strip()
        if len(new_q) < 5 or new_q == query:
            return query, docs
        logger.info(f"  🔄 REWRITE: '{query}' → '{new_q}'")
        if retriever:
            new_docs = retriever.retrieve_as_docs(new_q, top_k=5)
            if new_docs:
                seen = set(d.get("chunk_id", "") for d in docs)
                for d in new_docs:
                    if d.get("chunk_id", "") not in seen:
                        docs.append(d)
                        seen.add(d.get("chunk_id", ""))
        return new_q, docs


class DecomposeSkill:
    """Tách câu hỏi phức tạp thành nhiều câu hỏi con (320B API)."""

    def apply(self, query, docs, llm, retriever=None):
        from app.llm.prompts import DECOMPOSE_PROMPT
        result = llm.generate(DECOMPOSE_PROMPT.format(query=query))
        try:
            match = re.search(r'\{.*\}', result, re.DOTALL)
            if match:
                subs = json.loads(match.group()).get("sub_questions", [])
            else:
                return query, docs
        except (json.JSONDecodeError, KeyError):
            return query, docs
        if not subs or len(subs) < 2:
            return query, docs
        logger.info(f"  🔀 DECOMPOSE into {len(subs)} sub-questions")
        if retriever:
            seen = set(d.get("chunk_id", "") for d in docs)
            for sq in subs[:4]:
                for d in retriever.retrieve_as_docs(sq, top_k=3):
                    cid = d.get("chunk_id", "")
                    if cid not in seen:
                        docs.append(d)
                        seen.add(cid)
        super_q = f"{query}\nCác khía cạnh:\n" + "\n".join(f"- {s}" for s in subs)
        return super_q, docs


class EvidenceFocusSkill:
    """Thu hẹp phạm vi tìm kiếm (320B API)."""

    def apply(self, query, docs, llm, retriever=None):
        from app.llm.prompts import EVIDENCE_FOCUS_PROMPT
        result = llm.generate(EVIDENCE_FOCUS_PROMPT.format(query=query))
        try:
            match = re.search(r'\{.*\}', result, re.DOTALL)
            if match:
                fq = json.loads(match.group()).get("focused_query", query)
            else:
                return query, docs
        except (json.JSONDecodeError, KeyError):
            return query, docs
        if retriever and fq:
            new_docs = retriever.retrieve_as_docs(fq, top_k=5)
            if new_docs:
                docs = new_docs
        logger.info(f"  🎯 FOCUS: '{query}' → '{fq}'")
        return fq if fq else query, docs
