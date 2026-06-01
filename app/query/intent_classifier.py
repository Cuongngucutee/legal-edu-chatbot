"""
LawEdu AI — Intent Classifier.
Replaces the QwenIntentClassifier (1.5B local model) with Pro LLM API calls.
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


def get_fast_intent(query: str) -> Optional[str]:
    """Fast check for greetings and thanks without LLM."""
    q = query.lower().strip()
    greetings = ["xin chào", "chào bạn", "hello", "hi ", "hey", "chào"]
    thanks = ["cảm ơn", "thank", "cám ơn", "tks"]
    if len(q) < 30 and any(q.startswith(g) or q == g for g in greetings):
        return "GREETING"
    if len(q) < 40 and any(kw in q for kw in thanks):
        return "THANKS"
    return None

def classify_intent(query: str, llm) -> str:
    """
    Classify query intent. Priority: fast heuristics → Pro LLM fallback.

    Args:
        query: User question
        llm: LLMClient instance (Pro LLM API)

    Returns:
        Intent string (LOOKUP, SUMMARY, LISTING, etc.)
    """
    # ── P0: Greeting / Chitchat detection (instant, no API call) ──
    fast_intent = get_fast_intent(query)
    if fast_intent:
        return fast_intent

    q = query.lower().strip()

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

    # ── P2: Legal education keyword heuristic (no API call) ──
    # If query contains ≥2 legal/education domain keywords → LOOKUP immediately
    LEGAL_EDU_KEYWORDS = {
        # Pháp luật chung
        "điều", "khoản", "điểm", "luật", "nghị định", "thông tư", "quy định",
        "quyết định", "văn bản", "pháp luật", "hiến pháp", "quyền", "nghĩa vụ",
        "điều kiện", "tiêu chuẩn", "trách nhiệm", "xử phạt", "vi phạm",
        "hướng dẫn", "thi hành", "hiệu lực", "bãi bỏ", "sửa đổi", "bổ sung",
        "ban hành", "áp dụng", "chuyển đổi", "quyết nghị",
        # Giáo dục
        "giáo viên", "nhà giáo", "giảng viên", "học sinh", "sinh viên",
        "trường", "đại học", "mầm non", "tiểu học", "trung học", "thcs", "thpt",
        "giáo dục", "đào tạo", "tuyển sinh", "học phí", "học bổng",
        "sư phạm", "chức danh", "nghề nghiệp", "thăng hạng", "bổ nhiệm",
        "hội đồng", "hiệu trưởng", "cơ sở giáo dục", "chương trình",
        "bằng cấp", "tốt nghiệp", "kỷ luật", "đánh giá", "kiểm tra",
        "nâng chuẩn", "du học", "cử tuyển", "bồi hoàn", "thỉnh giảng",
        "lợi nhuận", "tư thục", "công lập", "hỗ trợ", "chính sách",
        "khu công nghiệp", "dân tộc", "nội trú",
    }
    match_count = sum(1 for kw in LEGAL_EDU_KEYWORDS if kw in q)
    if match_count >= 2 and len(q) > 20:
        logger.info(f"🏷️  Intent heuristic: LOOKUP (matched {match_count} legal keywords)")
        return "LOOKUP"

    # ── Pro LLM classification for truly ambiguous cases (< 5%) ──
    from app.llm.prompts import INTENT_PROMPT
    try:
        result = llm.generate(INTENT_PROMPT.format(query=query), max_tokens=10)
        intent = result.strip().upper().split('\n')[0].strip()
        if intent in VALID_INTENTS:
            return intent
    except Exception as e:
        logger.warning(f"LLM intent classification failed: {e}")

    return "LOOKUP"  # Safe default
