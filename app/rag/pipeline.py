"""
LawEdu AI — RAG Pipeline.
Main agentic RAG orchestrator: Intent → Route → Retrieve → Skill → Generate.
Uses 320B LLM API for all generation tasks.
"""
import re
import time
import logging

from app.query.intent_classifier import classify_intent
from app.query.query_expander import EducationQueryExpander
from app.query.query_rewriter import QueryRewriteSkill, DecomposeSkill, EvidenceFocusSkill
from app.rag.context_builder import build_context, postprocess_citations, fmt_source
from app.rag.handlers.handlers import (
    SummaryHandler, StatisticalHandler, ListingHandler, ComparisonHandler,
)
from app.llm.prompts import GENERATION_PROMPT, GENERATION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Failure State Prober
# ══════════════════════════════════════════════════════════════════

class FailureStateProber:
    """Phát hiện failure state sau retrieval bằng heuristic nhanh."""

    def probe(self, query, docs, llm=None):
        if not docs:
            return "NO_COVERAGE"

        doc_text = " ".join(d.get("content", "")[:300] for d in docs[:3]).lower()
        q_terms = set(re.findall(r'\b\w{3,}\b', query.lower()))
        d_terms = set(re.findall(r'\b\w{3,}\b', doc_text))

        stop = {
            "như", "thế", "nào", "gì", "có", "được", "không", "của", "cho", "khi",
            "thì", "phải", "theo", "là", "bao", "nhiêu", "những", "các", "trong",
            "về", "tại", "hay", "cần", "làm", "đối", "với", "này", "và", "hoặc",
            "đã", "sẽ", "đang", "một", "mà", "để", "từ", "quy", "định", "hãy",
            "liệt", "tất", "tóm", "tắt",
        }
        q_terms -= stop
        if not q_terms:
            return None

        overlap = len(q_terms & d_terms) / len(q_terms)

        if overlap < 0.15:
            return "SURFACE_MISMATCH"
        if overlap < 0.25:
            return "LOW_RELEVANCE"

        cond_markers = ["nếu", "khi", "trong trường hợp", "đối với", "điều kiện"]
        if sum(1 for m in cond_markers if m in query.lower()) > 1:
            return "ENTANGLED_PREMISES"

        return None


# ══════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════

class LawEduPipeline:
    """
    Agentic RAG Pipeline:
    1. Intent Classification (heuristics + 320B)
    2. Route to specialized handler OR standard retrieval
    3. Failure detection + Skill routing loop (max 3 iterations)
    4. Generation with 320B LLM + citation postprocessing
    """
    MAX_ITER = 3

    def __init__(self, retriever, agentic_llm, generator_llm):
        self.retriever = retriever
        self.llm = agentic_llm  # Used for intent, coref, rewrite
        self.generator_llm = generator_llm  # Used for final generation
        self.prober = FailureStateProber()
        self.query_expander = EducationQueryExpander()
        self.skills = {
            "QUERY_REWRITE": QueryRewriteSkill(),
            "DECOMPOSE": DecomposeSkill(),
            "EVIDENCE_FOCUS": EvidenceFocusSkill(),
        }
        # Specialized handlers
        self.summary_handler = SummaryHandler()
        self.statistical_handler = StatisticalHandler()
        self.listing_handler = ListingHandler()
        self.comparison_handler = ComparisonHandler()

    def run(self, query: str) -> dict:
        start = time.time()
        logger.info(f"\n{'=' * 50}\n📨 Query: {query}")

        # Step 1: Intent Classification
        intent = classify_intent(query, self.llm)
        logger.info(f"🏷️  Intent: {intent}")

        # Step 2: Instant responses (no retrieval needed)
        instant = {
            "GREETING": "Xin chào! 👋 Tôi là trợ lý AI chuyên về pháp luật giáo dục Việt Nam.\nBạn có thể hỏi tôi về Luật, Nghị định, Thông tư liên quan đến giáo dục.\nHãy đặt câu hỏi để tôi giúp bạn!",
            "THANKS": "Không có gì! 😊 Nếu bạn có thêm câu hỏi, đừng ngần ngại hỏi nhé.",
            "UNANSWERABLE": "Câu hỏi này nằm ngoài phạm vi cơ sở dữ liệu pháp luật giáo dục VN.",
        }
        if intent in instant:
            return {
                "answer": instant[intent], "sources": [], "skill_log": [],
                "iterations": 0, "intent": intent, "final_query": query,
                "latency_ms": (time.time() - start) * 1000,
            }

        # Step 3: Route to specialized handlers
        handlers = {
            "SUMMARY": self.summary_handler,
            "STATISTICAL": self.statistical_handler,
            "LISTING": self.listing_handler,
            "COMPARISON": self.comparison_handler,
        }
        if intent in handlers:
            result = handlers[intent].handle(query, self.retriever, self.generator_llm)
            result["latency_ms"] = (time.time() - start) * 1000
            return result

        # Step 4: Standard LOOKUP flow with failure detection
        return self._lookup_flow(query, start, intent)

    def _lookup_flow(self, query, start, intent):
        """Standard retrieval with query expansion + failure detection + skill routing."""
        current_q = query
        iteration = 0
        skill_log = []

        # Expand + retrieve
        expanded = self.query_expander.expand(query)
        all_docs = []
        seen = set()

        for d in self.retriever.retrieve_as_docs(query, top_k=5):
            cid = d.get("chunk_id", "")
            if cid not in seen:
                all_docs.append(d)
                seen.add(cid)

        for eq in expanded[:3]:
            for d in self.retriever.retrieve_as_docs(eq, top_k=5):
                cid = d.get("chunk_id", "")
                if cid not in seen:
                    all_docs.append(d)
                    seen.add(cid)

        docs = all_docs[:8] if all_docs else []

        # Failure detection loop
        while iteration < self.MAX_ITER:
            failure = self.prober.probe(current_q, docs)
            logger.info(f"  Iter {iteration}: failure={failure}, docs={len(docs)}")

            if failure is None or iteration == self.MAX_ITER - 1:
                break
            if failure == "NO_COVERAGE":
                return self._no_answer(query, start, intent)

            routing = {
                "SURFACE_MISMATCH": "QUERY_REWRITE",
                "BROAD_QUERY": "EVIDENCE_FOCUS",
                "ENTANGLED_PREMISES": "DECOMPOSE",
                "LOW_RELEVANCE": "QUERY_REWRITE",
            }
            skill_name = routing.get(failure, "EXIT")
            skill_log.append({"iter": iteration, "failure": failure, "skill": skill_name})

            if skill_name == "EXIT":
                return self._no_answer(query, start, intent)

            skill = self.skills.get(skill_name)
            if not skill:
                break

            current_q, docs = skill.apply(current_q, docs, self.llm, self.retriever)
            iteration += 1

        if not docs:
            return self._no_answer(query, start, intent)

        # Generate answer with 7B
        context = build_context(docs)
        answer = self.generator_llm.generate(
            GENERATION_PROMPT.format(query=query, context=context),
            system_prompt=GENERATION_SYSTEM_PROMPT,
        )
        answer = postprocess_citations(answer, docs)

        return {
            "answer": answer,
            "sources": [fmt_source(d) for d in docs],
            "skill_log": skill_log,
            "iterations": iteration + 1,
            "intent": intent,
            "final_query": current_q,
            "latency_ms": (time.time() - start) * 1000,
        }

    def _no_answer(self, query, start, intent="LOOKUP"):
        return {
            "answer": "Không tìm thấy quy định pháp luật giáo dục cụ thể liên quan đến câu hỏi này.",
            "sources": [], "skill_log": [], "iterations": 0,
            "intent": intent, "final_query": query,
            "latency_ms": (time.time() - start) * 1000,
        }


# ══════════════════════════════════════════════════════════════════
# Conversation Manager
# ══════════════════════════════════════════════════════════════════

class ConversationManager:
    """Manages conversation history + coreference resolution."""

    def __init__(self, pipeline: LawEduPipeline):
        self.pipeline = pipeline
        self.history = []

    def chat(self, user_query: str) -> dict:
        resolved = user_query

        # Resolve coreferences using 320B
        if self.history:
            from app.llm.prompts import CONVERSATION_RESOLVE_PROMPT
            hist = "\n".join(
                f"User: {h['q']}\nBot: {h['a'][:200]}..."
                for h in self.history[-3:]
            )
            resolved = self.pipeline.llm.generate(
                CONVERSATION_RESOLVE_PROMPT.format(history=hist, new_query=user_query)
            ).strip()
            if len(resolved) < 3:
                resolved = user_query
            logger.info(f"🔗 Resolved: '{user_query}' → '{resolved}'")

        result = self.pipeline.run(resolved)
        self.history.append({"q": user_query, "resolved": resolved, "a": result["answer"]})
        return result
