"""
Reasoning Layer – Chain-of-Thought reasoning engine.

Implements 4-step reasoning from the pipeline diagram:
1. Phân tích câu hỏi (Query Analysis)
2. Lập kế hoạch (Planning)
3. Đối chiếu bằng chứng (Evidence Alignment)
4. Phán đoán & kiểm chứng (Judgment & Verification)
"""
from __future__ import annotations

import logging
from typing import Optional

import google.generativeai as genai

from src.config import GEMINI_API_KEY, GEMINI_MODEL
from src.models import (
    ParsedQuery, RetrievedChunk, ReasoningResult, ReasoningStep,
)

logger = logging.getLogger(__name__)


_REASONING_SYSTEM_PROMPT = """Bạn là chuyên gia pháp luật giáo dục Việt Nam. Nhiệm vụ của bạn là phân tích câu hỏi pháp luật và lập luận dựa trên bằng chứng được cung cấp.

Bạn phải thực hiện 4 bước suy luận:

**Bước 1 – Phân tích câu hỏi:**
- Xác định loại câu hỏi (định nghĩa / thủ tục / điều kiện / so sánh / tổng hợp)
- Xác định phạm vi (luật nào, điều nào, khoản nào)
- Liệt kê các khái niệm chính cần tra cứu

**Bước 2 – Lập kế hoạch:**
- Xác định bước lập luận cần thiết
- Xác định thứ tự xử lý thông tin
- Đánh giá liệu bằng chứng có đủ hay không

**Bước 3 – Đối chiếu bằng chứng:**
- Trích xuất các điểm chính từ mỗi bằng chứng
- Kiểm tra tính nhất quán giữa các nguồn
- Xác định mối liên hệ giữa các điều khoản

**Bước 4 – Phán đoán & kiểm chứng:**
- Đưa ra kết luận từ bằng chứng
- Phát hiện mâu thuẫn hoặc thiếu sót
- Đánh giá độ tin cậy của câu trả lời
- Ghi chú nếu cần thêm thông tin

Trả lời BẰNG TIẾNG VIỆT. Format mỗi bước rõ ràng."""


_REASONING_USER_TEMPLATE = """## Câu hỏi
{question}

## Loại câu hỏi đã phân tích
Intent: {intent}
Luật tham chiếu: {law_refs}
Điều tham chiếu: {article_refs}
Khái niệm chính: {concepts}

## Bằng chứng thu thập được
{evidence}

---
Hãy thực hiện 4 bước suy luận. Ở Bước 4, nếu bạn cần thêm thông tin thì ghi rõ cần tìm gì."""


class ReasoningEngine:
    """Chain-of-Thought reasoning engine using LLM."""

    def __init__(self, api_key: str = GEMINI_API_KEY, model: str = GEMINI_MODEL):
        self.model_name = model
        self.api_key = api_key
        if api_key:
            genai.configure(api_key=api_key)

    def reason(
        self,
        parsed_query: ParsedQuery,
        retrieved_chunks: list[RetrievedChunk],
    ) -> ReasoningResult:
        """Run chain-of-thought reasoning over retrieved evidence."""

        # Build evidence text
        evidence = self._format_evidence(retrieved_chunks)

        if not self.api_key:
            # Fallback: no LLM available, return basic analysis
            return self._fallback_reasoning(parsed_query, retrieved_chunks)

        # Build prompt
        user_prompt = _REASONING_USER_TEMPLATE.format(
            question=parsed_query.original,
            intent=parsed_query.intent.value,
            law_refs=", ".join(parsed_query.law_references) or "không xác định",
            article_refs=", ".join(parsed_query.article_references) or "không xác định",
            concepts=", ".join(parsed_query.key_concepts) or "chung",
            evidence=evidence,
        )

        try:
            model = genai.GenerativeModel(self.model_name, system_instruction=_REASONING_SYSTEM_PROMPT)
            response = model.generate_content(
                user_prompt,
                generation_config=genai.GenerationConfig(temperature=0.1, max_output_tokens=2000),
            )
            reasoning_text = response.text or ""
        except Exception as e:
            logger.error(f"LLM reasoning failed: {e}")
            return self._fallback_reasoning(parsed_query, retrieved_chunks)

        # Parse the 4 steps from the response
        return self._parse_reasoning(reasoning_text)

    # ── Evidence Formatting ────────────────────────────────────────

    def _format_evidence(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks as numbered evidence blocks."""
        blocks = []
        for i, rc in enumerate(chunks, 1):
            c = rc.chunk
            header = f"[{c.source}] {c.article_title}"
            if c.clause_number:
                header += f", Khoản {c.clause_number}"
            if c.point_label:
                header += f", Điểm {c.point_label}"

            blocks.append(
                f"### Bằng chứng {i} ({rc.source_method}, score={rc.score:.3f})\n"
                f"**{header}**\n"
                f"{c.text}\n"
            )

        return "\n".join(blocks)

    # ── Parse Reasoning Output ─────────────────────────────────────

    def _parse_reasoning(self, text: str) -> ReasoningResult:
        """Parse the 4-step reasoning output from LLM."""
        result = ReasoningResult()
        steps = []

        # Try to split by step markers
        sections = {
            "query_analysis": ["bước 1", "phân tích câu hỏi"],
            "plan": ["bước 2", "lập kế hoạch"],
            "evidence_alignment": ["bước 3", "đối chiếu bằng chứng"],
            "judgment": ["bước 4", "phán đoán", "kiểm chứng"],
        }

        text_lower = text.lower()
        current_section = ""
        current_content = []

        for line in text.split("\n"):
            line_lower = line.lower().strip()

            matched_section = None
            for section_name, markers in sections.items():
                if any(marker in line_lower for marker in markers):
                    matched_section = section_name
                    break

            if matched_section:
                # Save previous section
                if current_section and current_content:
                    content = "\n".join(current_content).strip()
                    setattr(result, current_section, content)
                    steps.append(ReasoningStep(
                        step_name=current_section,
                        content=content,
                    ))

                current_section = matched_section
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_section and current_content:
            content = "\n".join(current_content).strip()
            setattr(result, current_section, content)
            steps.append(ReasoningStep(
                step_name=current_section,
                content=content,
            ))

        # If parsing failed, put everything in judgment
        if not steps:
            result.judgment = text
            steps.append(ReasoningStep(step_name="judgment", content=text))

        # Check if more info is needed
        judgment_lower = result.judgment.lower()
        if any(
            phrase in judgment_lower
            for phrase in [
                "cần thêm thông tin",
                "chưa đủ bằng chứng",
                "không tìm thấy",
                "thiếu thông tin",
            ]
        ):
            result.needs_more_info = True

        return result

    # ── Fallback ───────────────────────────────────────────────────

    def _fallback_reasoning(
        self, parsed_query: ParsedQuery, chunks: list[RetrievedChunk]
    ) -> ReasoningResult:
        """Simple reasoning without LLM."""
        evidence_summary = []
        for rc in chunks[:5]:
            c = rc.chunk
            ref = f"{c.source}, {c.article_title}"
            if c.clause_number:
                ref += f", Khoản {c.clause_number}"
            evidence_summary.append(f"- {ref}")

        return ReasoningResult(
            query_analysis=f"Câu hỏi: {parsed_query.original}\n"
                          f"Loại: {parsed_query.intent.value}\n"
                          f"Khái niệm: {', '.join(parsed_query.key_concepts)}",
            plan="Tìm kiếm trong các điều khoản liên quan và tổng hợp câu trả lời.",
            evidence_alignment="Bằng chứng thu thập:\n" + "\n".join(evidence_summary),
            judgment="Trả lời dựa trên các bằng chứng thu thập được.",
            needs_more_info=False,
        )
