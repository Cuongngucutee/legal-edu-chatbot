"""
Answer Generation Layer

Generates final answers using LLM with:
1. Context-aware prompting for legal QA
2. Automatic citation extraction
3. Confidence scoring
"""
from __future__ import annotations

import re
import logging
from typing import Optional

import google.generativeai as genai

from src.config import GEMINI_API_KEY, GEMINI_MODEL
from src.models import (
    ParsedQuery, RetrievedChunk, ReasoningResult,
    PipelineAnswer, Citation, ReasoningStep,
)

logger = logging.getLogger(__name__)


_GENERATION_SYSTEM_PROMPT = """Bạn là trợ lý pháp luật giáo dục Việt Nam, chuyên trả lời câu hỏi về luật giáo dục.

## Nguyên tắc trả lời:
1. **Chính xác**: Chỉ trả lời dựa trên nội dung luật được cung cấp, KHÔNG bịa đặt
2. **Trích dẫn**: Luôn trích dẫn rõ nguồn (Luật số, Điều, Khoản, Điểm)
3. **Rõ ràng**: Giải thích bằng ngôn ngữ dễ hiểu, có cấu trúc
4. **Đầy đủ**: Nếu có nhiều điều khoản liên quan, nêu hết
5. **Trung thực**: Nếu không đủ thông tin, nói rõ là không chắc chắn

## Format trả lời:
- Sử dụng Markdown
- Trích dẫn inline: **(Luật X, Điều Y, Khoản Z)**
- Liệt kê các điểm quan trọng bằng bullet points
- Kết thúc với phần "📌 Căn cứ pháp lý" liệt kê tất cả nguồn đã dùng

## Lưu ý:
- Nếu luật mới (Luật sửa đổi) thay thế nội dung luật cũ, ưu tiên nội dung mới
- Phân biệt rõ quy định hiện hành và quy định đã bị sửa đổi"""


_GENERATION_USER_TEMPLATE = """## Câu hỏi
{question}

## Phân tích suy luận
{reasoning}

## Nội dung luật liên quan
{context}

---
Hãy trả lời câu hỏi trên dựa trên nội dung luật đã cung cấp. Nhớ trích dẫn nguồn cụ thể."""


class AnswerGenerator:
    """Generates final answers with citations using LLM."""

    def __init__(self, api_key: str = GEMINI_API_KEY, model: str = GEMINI_MODEL):
        self.model_name = model
        self.api_key = api_key
        if api_key:
            genai.configure(api_key=api_key)

    def generate(
        self,
        parsed_query: ParsedQuery,
        retrieved_chunks: list[RetrievedChunk],
        reasoning: ReasoningResult,
    ) -> PipelineAnswer:
        """Generate the final answer."""

        # Build context from retrieved chunks
        context = self._build_context(retrieved_chunks)

        # Build reasoning summary
        reasoning_text = self._format_reasoning(reasoning)

        if not self.api_key:
            return self._fallback_answer(parsed_query, retrieved_chunks, reasoning)

        # Build prompt
        user_prompt = _GENERATION_USER_TEMPLATE.format(
            question=parsed_query.original,
            reasoning=reasoning_text,
            context=context,
        )

        try:
            model = genai.GenerativeModel(self.model_name, system_instruction=_GENERATION_SYSTEM_PROMPT)
            response = model.generate_content(
                user_prompt,
                generation_config=genai.GenerationConfig(temperature=0.15, max_output_tokens=3000),
            )
            answer_text = response.text or ""
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._fallback_answer(parsed_query, retrieved_chunks, reasoning)

        # Extract citations from the answer and from chunks
        citations = self._extract_citations(answer_text, retrieved_chunks)

        # Estimate confidence
        confidence = self._estimate_confidence(
            answer_text, retrieved_chunks, reasoning
        )

        # Generate related questions
        related = self._suggest_related(parsed_query, retrieved_chunks)

        # Warning if low confidence
        warning = ""
        if confidence < 0.5:
            warning = "⚠️ Câu trả lời này có độ tin cậy thấp. Vui lòng tham khảo thêm văn bản luật gốc."
        elif reasoning.needs_more_info:
            warning = "⚠️ Có thể thiếu một số thông tin liên quan. Câu trả lời dựa trên dữ liệu hiện có."

        return PipelineAnswer(
            question=parsed_query.original,
            answer=answer_text,
            citations=citations,
            confidence=confidence,
            reasoning_trace=[
                ReasoningStep(step_name="query_analysis", content=reasoning.query_analysis),
                ReasoningStep(step_name="plan", content=reasoning.plan),
                ReasoningStep(step_name="evidence_alignment", content=reasoning.evidence_alignment),
                ReasoningStep(step_name="judgment", content=reasoning.judgment),
            ],
            related_questions=related,
            warning=warning,
        )

    # ── Context Building ───────────────────────────────────────────

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        """Build context string from retrieved chunks, grouped by article."""
        # Group by article
        articles: dict[str, list[RetrievedChunk]] = {}
        for rc in chunks:
            key = f"{rc.chunk.source}|{rc.chunk.article_title}"
            if key not in articles:
                articles[key] = []
            articles[key].append(rc)

        blocks = []
        for key, rcs in articles.items():
            source, title = key.split("|", 1) if "|" in key else (key, "")
            header = f"### {source} – {title}"

            # Use article_full_text if available (more complete), otherwise chunks
            first_chunk = rcs[0].chunk
            if first_chunk.article_full_text:
                blocks.append(f"{header}\n{first_chunk.article_full_text}")
            else:
                texts = [rc.chunk.text for rc in rcs]
                blocks.append(f"{header}\n" + "\n".join(texts))

        return "\n\n".join(blocks)

    def _format_reasoning(self, reasoning: ReasoningResult) -> str:
        """Format reasoning result for the prompt."""
        parts = []
        if reasoning.query_analysis:
            parts.append(f"**Phân tích:** {reasoning.query_analysis[:300]}")
        if reasoning.plan:
            parts.append(f"**Kế hoạch:** {reasoning.plan[:300]}")
        if reasoning.evidence_alignment:
            parts.append(f"**Đối chiếu:** {reasoning.evidence_alignment[:300]}")
        if reasoning.judgment:
            parts.append(f"**Phán đoán:** {reasoning.judgment[:300]}")
        return "\n".join(parts) if parts else "Chưa có phân tích."

    # ── Citation Extraction ────────────────────────────────────────

    def _extract_citations(
        self, answer_text: str, chunks: list[RetrievedChunk]
    ) -> list[Citation]:
        """Extract citations from the answer text and matched chunks."""
        citations = []
        seen = set()

        for rc in chunks:
            c = rc.chunk
            citation_key = (c.source, c.article_number, c.clause_number, c.point_label)
            if citation_key in seen:
                continue
            seen.add(citation_key)

            citation = Citation(
                source=c.source,
                article=f"Điều {c.article_number}" if c.article_number else "",
                clause=f"Khoản {c.clause_number}" if c.clause_number else "",
                point=f"Điểm {c.point_label}" if c.point_label else "",
                text_excerpt=c.text[:150] + "..." if len(c.text) > 150 else c.text,
            )
            citations.append(citation)

        return citations[:10]  # Limit to 10 citations

    # ── Confidence Estimation ──────────────────────────────────────

    def _estimate_confidence(
        self,
        answer_text: str,
        chunks: list[RetrievedChunk],
        reasoning: ReasoningResult,
    ) -> float:
        """Heuristic confidence estimation."""
        score = 0.5  # baseline

        # More retrieved evidence = more confidence
        if len(chunks) >= 3:
            score += 0.1
        if len(chunks) >= 5:
            score += 0.1

        # High retrieval scores = better match
        if chunks:
            avg_score = sum(r.score for r in chunks) / len(chunks)
            if avg_score > 0.02:  # RRF scores are small
                score += 0.1

        # Multiple sources agree
        sources = set(r.chunk.source for r in chunks)
        if len(sources) >= 2:
            score += 0.05

        # Reasoning indicates enough info
        if not reasoning.needs_more_info:
            score += 0.1

        # Cap at 0.95
        return min(score, 0.95)

    # ── Related Questions ──────────────────────────────────────────

    def _suggest_related(
        self, parsed_query: ParsedQuery, chunks: list[RetrievedChunk]
    ) -> list[str]:
        """Suggest related follow-up questions."""
        suggestions = []

        # Based on concepts in retrieved chunks but NOT in query
        query_concepts = set(parsed_query.key_concepts)
        related_concepts = set()
        for rc in chunks:
            text_lower = rc.chunk.text.lower()
            for concept in [
                "tự chủ", "kiểm định", "tuyển sinh", "giảng viên",
                "người học", "học phí", "văn bằng", "hợp tác quốc tế",
                "nghiên cứu khoa học", "chương trình đào tạo",
            ]:
                if concept in text_lower and concept not in query_concepts:
                    related_concepts.add(concept)

        for concept in list(related_concepts)[:3]:
            suggestions.append(f"Quy định về {concept} trong giáo dục đại học?")

        # Based on other articles in same chapter
        if chunks:
            first = chunks[0].chunk
            if first.source and first.article_number:
                suggestions.append(
                    f"Các quy định khác trong {first.chapter} của {first.source}?"
                )

        return suggestions[:3]

    # ── Fallback ───────────────────────────────────────────────────

    def _fallback_answer(
        self,
        parsed_query: ParsedQuery,
        chunks: list[RetrievedChunk],
        reasoning: ReasoningResult,
    ) -> PipelineAnswer:
        """Generate answer without LLM using retrieved chunks directly."""
        if not chunks:
            return PipelineAnswer(
                question=parsed_query.original,
                answer="Không tìm thấy thông tin liên quan trong cơ sở dữ liệu luật giáo dục.",
                confidence=0.1,
                warning="⚠️ Không có API key LLM. Kết quả trả về là nội dung luật thô.",
            )

        # Build a basic answer from top chunks
        parts = ["Dựa trên các điều khoản luật liên quan:\n"]

        for i, rc in enumerate(chunks[:5], 1):
            c = rc.chunk
            ref = f"**{c.source}, {c.article_title}"
            if c.clause_number:
                ref += f", Khoản {c.clause_number}"
            if c.point_label:
                ref += f", Điểm {c.point_label}"
            ref += "**"

            # Use chunked text (cleaner)
            original_text = c.text
            # Remove the enriched header to show original content
            if original_text.startswith("["):
                # Find the last ] before actual content
                last_bracket = original_text.rfind("]")
                if last_bracket > 0:
                    original_text = original_text[last_bracket + 1:].strip()

            parts.append(f"{i}. {ref}\n   {original_text}\n")

        citations = self._extract_citations("", chunks)

        parts.append("\n📌 **Căn cứ pháp lý:**")
        for cit in citations[:5]:
            parts.append(f"- {cit.format()}")

        return PipelineAnswer(
            question=parsed_query.original,
            answer="\n".join(parts),
            citations=citations,
            confidence=0.5,
            warning="⚠️ Không có API key LLM. Kết quả trả về là trích dẫn trực tiếp từ luật.",
            reasoning_trace=[
                ReasoningStep(step_name="fallback", content="No LLM available"),
            ],
        )
