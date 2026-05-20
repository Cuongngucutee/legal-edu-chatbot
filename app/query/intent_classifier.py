"""
LawEdu AI — Intent Classifier.
Replaces the QwenIntentClassifier (1.5B local model) with 320B API calls.
Keeps the fast heuristic rules + uses 320B for ambiguous cases.
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Valid intent types
VALID_INTENTS = {
    "LOOKUP", "SUMMARY", "LISTING", "STATISTICAL",
    "COMPARISON", "UNANSWERABLE", "GREETING", "THANKS",
}


def classify_intent(query: str, llm) -> str:
    """
    Classify query intent. Priority: fast heuristics → 320B LLM fallback.

    Args:
        query: User question
        llm: LLMClient instance (320B API)

    Returns:
        Intent string (LOOKUP, SUMMARY, LISTING, etc.)
    """
    q = query.lower().strip()

    # ── P0: Greeting / Chitchat detection (instant, no API call) ──
    greetings = ["xin chào", "chào bạn", "hello", "hi ", "hey", "chào"]
    thanks = ["cảm ơn", "thank", "cám ơn", "tks"]
    if len(q) < 30 and any(q.startswith(g) or q == g for g in greetings):
        return "GREETING"
    if len(q) < 40 and any(kw in q for kw in thanks):
        return "THANKS"

    # ── Heuristic rules (fast, no API call needed) ──
    if any(kw in q for kw in ["tóm tắt", "khái quát", "nội dung chính", "quy định những gì"]):
        return "SUMMARY"

    if any(kw in q for kw in ["bao nhiêu", "số lượng", "đếm", "thống kê"]):
        # Context-aware: "bao nhiêu" in lookup context → LOOKUP, not STATISTICAL
        lookup_context = [
            "bồi hoàn", "mức", "phí", "lương", "trợ cấp", "hỗ trợ",
            "phạt", "tiền", "chi phí", "học phí", "tuổi", "học sinh",
            "lớp", "lâu", "năm", "tháng", "ngày", "tối đa", "tối thiểu",
            "giờ", "tiết", "điểm", "hạng",
        ]
        if any(kw in q for kw in lookup_context):
            return "LOOKUP"
        if any(kw in q for kw in ["top 100", "thế giới", "xếp hạng quốc tế"]):
            return "UNANSWERABLE"
        return "STATISTICAL"

    if any(kw in q for kw in [
        "liệt kê", "tất cả", "những trường hợp nào", "trường hợp nào",
        "những đối tượng nào", "đối tượng nào", "bao gồm những gì",
        "các loại", "kể tên",
    ]):
        return "LISTING"

    if "so sánh" in q or " vs " in q or "đối chiếu" in q:
        return "COMPARISON"

    # ── 320B LLM classification for non-obvious cases ──
    from app.llm.prompts import INTENT_PROMPT
    try:
        result = llm.generate(INTENT_PROMPT.format(query=query), max_tokens=10)
        intent = result.strip().upper().split('\n')[0].strip()
        if intent in VALID_INTENTS:
            return intent
    except Exception as e:
        logger.warning(f"LLM intent classification failed: {e}")

    return "LOOKUP"  # Safe default
