"""
Post-processing & Formatting

Formats the final pipeline answer into Markdown with:
- Structured answer
- Citation block
- Confidence indicator
- Warning messages
- Related questions
"""
from __future__ import annotations

from src.models import PipelineAnswer


class Formatter:
    """Format PipelineAnswer into user-friendly Markdown."""

    def format(self, answer: PipelineAnswer, show_reasoning: bool = False) -> str:
        """Format the answer as Markdown."""
        parts = []

        # ── Header ─────────────────────────────────────────────────
        parts.append(f"## 💬 Câu hỏi\n{answer.question}\n")

        # ── Warning (if any) ──────────────────────────────────────
        if answer.warning:
            parts.append(f"> {answer.warning}\n")

        # ── Confidence bar ─────────────────────────────────────────
        conf_pct = int(answer.confidence * 100)
        conf_bar = self._confidence_bar(answer.confidence)
        parts.append(f"**Độ tin cậy:** {conf_bar} {conf_pct}%\n")

        # ── Main answer ────────────────────────────────────────────
        parts.append(f"## 📝 Trả lời\n{answer.answer}\n")

        # ── Citations ──────────────────────────────────────────────
        if answer.citations:
            parts.append("## 📌 Căn cứ pháp lý")
            for i, cit in enumerate(answer.citations, 1):
                ref = cit.format()
                parts.append(f"{i}. **{ref}**")
                if cit.text_excerpt:
                    # Truncate excerpt
                    excerpt = cit.text_excerpt
                    if len(excerpt) > 120:
                        excerpt = excerpt[:120] + "..."
                    parts.append(f"   > {excerpt}")
            parts.append("")

        # ── Related questions ──────────────────────────────────────
        if answer.related_questions:
            parts.append("## 💡 Câu hỏi liên quan")
            for q in answer.related_questions:
                parts.append(f"- {q}")
            parts.append("")

        # ── Reasoning trace (debug) ────────────────────────────────
        if show_reasoning and answer.reasoning_trace:
            parts.append("---\n## 🔍 Chi tiết suy luận (debug)")
            for step in answer.reasoning_trace:
                if step.content:
                    parts.append(f"### {step.step_name}")
                    parts.append(step.content)
                    parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _confidence_bar(confidence: float) -> str:
        """Generate a visual confidence bar."""
        filled = int(confidence * 10)
        empty = 10 - filled
        if confidence >= 0.7:
            color = "🟢"
        elif confidence >= 0.4:
            color = "🟡"
        else:
            color = "🔴"
        return f"{color} {'█' * filled}{'░' * empty}"

    def format_compact(self, answer: PipelineAnswer) -> str:
        """Compact format for chat-like interfaces."""
        parts = [answer.answer]

        if answer.citations:
            refs = [cit.format() for cit in answer.citations[:5]]
            parts.append(f"\n📌 *Căn cứ: {'; '.join(refs)}*")

        if answer.warning:
            parts.append(f"\n{answer.warning}")

        return "\n".join(parts)
