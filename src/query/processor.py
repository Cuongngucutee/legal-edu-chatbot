"""
Query Processing Layer

Handles:
1. Intent Classification – categorize user questions
2. Query Normalization – fix abbreviations, typos
3. Named Entity Recognition – extract law/article/clause references
"""
from __future__ import annotations

import re
import logging

from src.models import ParsedQuery, QueryIntent

logger = logging.getLogger(__name__)

# ── Abbreviation map ───────────────────────────────────────────────
_ABBREVIATIONS = {
    r"\bĐH\b": "đại học",
    r"\bGD\b": "giáo dục",
    r"\bGD&ĐT\b": "Giáo dục và Đào tạo",
    r"\bGDĐT\b": "Giáo dục và Đào tạo",
    r"\bSV\b": "sinh viên",
    r"\bGV\b": "giảng viên",
    r"\bNCS\b": "nghiên cứu sinh",
    r"\bCĐ\b": "cao đẳng",
    r"\bThS\b": "thạc sĩ",
    r"\bTS\b": "tiến sĩ",
    r"\bCSGDĐH\b": "cơ sở giáo dục đại học",
    r"\bBGD\b": "Bộ Giáo dục",
    r"\bCP\b": "Chính phủ",
    r"\bQH\b": "Quốc hội",
    r"\bHĐT\b": "hội đồng trường",
}

# ── Intent patterns ────────────────────────────────────────────────
_INTENT_PATTERNS: dict[QueryIntent, list[str]] = {
    QueryIntent.DEFINITION: [
        r"là gì", r"được hiểu là", r"nghĩa là", r"định nghĩa",
        r"giải thích", r"khái niệm", r"thế nào là",
    ],
    QueryIntent.PROCEDURE: [
        r"thủ tục", r"quy trình", r"trình tự", r"hồ sơ",
        r"các bước", r"làm thế nào", r"như thế nào",
        r"cách thức", r"phải làm gì",
    ],
    QueryIntent.CONDITION: [
        r"điều kiện", r"tiêu chuẩn", r"yêu cầu",
        r"cần phải", r"đủ điều kiện", r"đáp ứng",
        r"khi nào", r"trường hợp nào",
    ],
    QueryIntent.COMPARISON: [
        r"so sánh", r"khác nhau", r"giống nhau",
        r"khác biệt", r"phân biệt", r"điểm khác",
        r"sự khác", r"so với",
    ],
}


class QueryProcessor:
    """Process user queries: normalize, classify intent, extract entities."""

    def process(self, raw_query: str) -> ParsedQuery:
        """Full query processing pipeline."""
        normalized = self._normalize(raw_query)
        intent = self._classify_intent(normalized)
        law_refs = self._extract_law_refs(normalized)
        article_refs = self._extract_article_refs(normalized)
        clause_refs = self._extract_clause_refs(normalized)
        point_refs = self._extract_point_refs(normalized)
        concepts = self._extract_concepts(normalized)

        parsed = ParsedQuery(
            original=raw_query,
            normalized=normalized,
            intent=intent,
            law_references=law_refs,
            article_references=article_refs,
            clause_references=clause_refs,
            point_references=point_refs,
            key_concepts=concepts,
        )

        logger.info(
            f"Query processed: intent={intent.value}, "
            f"laws={law_refs}, articles={article_refs}, "
            f"concepts={concepts[:3]}"
        )
        return parsed

    # ── Normalization ──────────────────────────────────────────────

    def _normalize(self, query: str) -> str:
        """Normalize query: expand abbreviations, clean whitespace."""
        text = query.strip()
        for pattern, replacement in _ABBREVIATIONS.items():
            text = re.sub(pattern, replacement, text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        return text

    # ── Intent Classification ──────────────────────────────────────

    def _classify_intent(self, query: str) -> QueryIntent:
        """Rule-based intent classification."""
        query_lower = query.lower()

        scores: dict[QueryIntent, int] = {}
        for intent, patterns in _INTENT_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, query_lower))
            if score > 0:
                scores[intent] = score

        if not scores:
            return QueryIntent.GENERAL

        return max(scores, key=scores.get)

    # ── Entity Extraction ──────────────────────────────────────────

    def _extract_law_refs(self, query: str) -> list[str]:
        """Extract law references like 'Luật 08/2012/QH13' or 'Luật Giáo dục đại học'."""
        refs = []

        # Pattern: Luật XX/YYYY/QHXX
        pattern1 = r"[Ll]uật\s+(\d+[/\-]\d{4}[/\-]\w+)"
        for match in re.finditer(pattern1, query):
            ref = f"Luật {match.group(1)}"
            ref = ref.replace("-", "/")
            refs.append(ref)

        # Pattern: Luật + name
        pattern2 = r"[Ll]uật\s+((?:Giáo dục|giáo dục)(?:\s+đại học)?)"
        for match in re.finditer(pattern2, query):
            refs.append(f"Luật {match.group(1)}")

        # Pattern: Just the number "Luật 08" or "Luật 34"
        pattern3 = r"[Ll]uật\s+(\d+)\b"
        for match in re.finditer(pattern3, query):
            if not any(match.group(1) in r for r in refs):
                refs.append(f"Luật {match.group(1)}")

        return list(set(refs))

    def _extract_article_refs(self, query: str) -> list[str]:
        """Extract article references: 'Điều 4', 'điều 32'."""
        pattern = r"[Đđ]iều\s+(\d+[a-z]?)"
        matches = re.findall(pattern, query)
        return [f"Điều {m}" for m in set(matches)]

    def _extract_clause_refs(self, query: str) -> list[str]:
        """Extract clause references: 'Khoản 2', 'khoản 1'."""
        pattern = r"[Kk]hoản\s+(\d+)"
        matches = re.findall(pattern, query)
        return [f"Khoản {m}" for m in set(matches)]

    def _extract_point_refs(self, query: str) -> list[str]:
        """Extract point references: 'Điểm a', 'điểm b'."""
        pattern = r"[Đđ]iểm\s+([a-zđ])"
        matches = re.findall(pattern, query)
        return [f"Điểm {m}" for m in set(matches)]

    def _extract_concepts(self, query: str) -> list[str]:
        """Extract key legal concepts from the query."""
        query_lower = query.lower()
        concepts = []

        known_concepts = [
            "giáo dục đại học", "giáo dục chính quy", "giáo dục thường xuyên",
            "tự chủ", "kiểm định chất lượng", "tuyển sinh",
            "đào tạo từ xa", "liên thông", "chương trình đào tạo",
            "nghiên cứu khoa học", "hợp tác quốc tế",
            "cơ sở giáo dục đại học", "trường đại học", "đại học",
            "giảng viên", "người học", "sinh viên", "nghiên cứu sinh",
            "hội đồng trường", "hiệu trưởng",
            "văn bằng", "chứng chỉ", "học phí",
            "công lập", "tư thục", "không vì lợi nhuận",
            "ngành đào tạo", "chuyên ngành", "trách nhiệm giải trình",
            "nhà giáo", "cán bộ quản lý", "quyền tự chủ",
            "giáo dục mầm non", "giáo dục phổ thông",
            "giáo dục nghề nghiệp",
        ]

        for concept in known_concepts:
            if concept in query_lower:
                concepts.append(concept)

        return concepts
