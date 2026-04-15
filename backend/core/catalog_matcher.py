"""
Catalog-based Document Matcher (v2 — Semantic-Aware)
=====================================================
Sử dụng document_catalog_final.json để pre-filter documents dựa trên ngữ nghĩa.
Khi câu hỏi không chứa tên văn bản, module này quét catalog để tìm ra
các doc_id phù hợp nhất dựa trên main_topics, keywords, usage_context.

v2 Changes:
- N-gram matching (2-gram, 3-gram) thay vì chỉ unigram
- Synonym/alias expansion cho thuật ngữ pháp luật giáo dục
- IDF-aware weighting
- Education level filtering
- Confidence tier (high/medium/low)
"""

import json
import math
import os
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional
from pathlib import Path


# ─── Synonym Groups ─────────────────────────────────────────────────────────
# Mỗi group: nếu query chứa bất kỳ term nào → match catalog chứa bất kỳ term nào
SYNONYM_GROUPS = [
    # Người dạy
    {"giáo viên", "nhà giáo", "giảng viên", "thầy cô", "người dạy", "cán bộ giảng dạy"},
    # Người học
    {"học sinh", "sinh viên", "người học", "học viên", "trẻ em", "trẻ mầm non"},
    # Học phí
    {"học phí", "chi phí đào tạo", "kinh phí học tập", "chi phí học tập", "mức thu"},
    # Miễn giảm
    {"miễn học phí", "giảm học phí", "miễn giảm", "hỗ trợ chi phí học tập", "miễn", "giảm"},
    # Trường học
    {"trường học", "cơ sở giáo dục", "nhà trường", "trường"},
    # Lương / chế độ
    {"lương", "tiền lương", "phụ cấp", "chế độ đãi ngộ", "đãi ngộ", "thù lao"},
    # Tuyển sinh
    {"tuyển sinh", "xét tuyển", "thi tuyển", "nhập học", "trúng tuyển"},
    # Bằng cấp / trình độ
    {"bằng cấp", "văn bằng", "chứng chỉ", "trình độ", "học vị"},
    # Kiểm định
    {"kiểm định", "đánh giá chất lượng", "kiểm tra chất lượng", "công nhận chất lượng"},
    # Tự chủ
    {"tự chủ", "tự chủ đại học", "tự chủ tài chính", "quyền tự chủ"},
    # Chương trình
    {"chương trình", "chương trình đào tạo", "chương trình giáo dục", "khung chương trình"},
    # Kỷ luật
    {"kỷ luật", "xử lý kỷ luật", "vi phạm", "xử phạt", "hành vi vi phạm"},
    # Thành lập trường
    {"thành lập", "thành lập trường", "cho phép thành lập", "điều kiện thành lập"},
    # Đại học
    {"đại học", "giáo dục đại học", "trường đại học", "viện đại học", "học viện"},
    # Phổ thông
    {"phổ thông", "cấp 1", "cấp 2", "cấp 3", "tiểu học", "trung học"},
    # Mầm non
    {"mầm non", "nhà trẻ", "mẫu giáo", "trẻ mầm non"},
    # Nghề nghiệp
    {"nghề nghiệp", "giáo dục nghề nghiệp", "dạy nghề", "trung cấp", "cao đẳng nghề"},
    # Hồ sơ / thủ tục
    {"hồ sơ", "thủ tục", "giấy tờ", "đơn xin", "quy trình"},
    # Mở ngành
    {"mở ngành", "đăng ký ngành", "ngành đào tạo"},
    # Bồi dưỡng
    {"bồi dưỡng", "tập huấn", "đào tạo lại", "nâng cao năng lực"},
    # Chức danh
    {"chức danh", "hạng chức danh", "nghề nghiệp", "xếp hạng", "thăng hạng"},
    # Du học
    {"du học", "tư vấn du học", "du học nước ngoài", "học nước ngoài"},
    # Sách giáo khoa
    {"sách giáo khoa", "sgk", "giáo trình", "tài liệu giảng dạy"},
]

# ─── Education Level Detection ──────────────────────────────────────────────

EDUCATION_LEVEL_MAP = {
    "mầm non": ["Mầm non", "mầm non", "nhà trẻ", "mẫu giáo"],
    "tiểu học": ["Tiểu học", "tiểu học", "cấp 1", "lớp 1", "lớp 2", "lớp 3", "lớp 4", "lớp 5"],
    "thcs": ["THCS", "thcs", "trung học cơ sở", "cấp 2", "lớp 6", "lớp 7", "lớp 8", "lớp 9"],
    "thpt": ["THPT", "thpt", "trung học phổ thông", "cấp 3", "lớp 10", "lớp 11", "lớp 12"],
    "phổ thông": ["Phổ thông", "phổ thông"],
    "cao đẳng": ["Cao đẳng", "cao đẳng"],
    "đại học": ["Đại học", "đại học", "Đại học/Sau đại học"],
    "thạc sĩ": ["Thạc sĩ", "thạc sĩ", "sau đại học"],
    "tiến sĩ": ["Tiến sĩ", "tiến sĩ"],
    "giáo dục thường xuyên": ["Giáo dục thường xuyên", "GDTX", "giáo dục thường xuyên"],
    "giáo dục nghề nghiệp": ["Giáo dục nghề nghiệp", "nghề nghiệp", "dạy nghề", "GDNN"],
}


class CatalogMatcher:
    """
    Quét catalog để mapping: câu hỏi → danh sách doc_id liên quan.
    
    v2: N-gram matching + synonym expansion + IDF weighting + confidence tiers.
    
    Scoring weights:
      - main_topics (weight: 3.0, ngram bonus: ×2)
      - key_entities (weight: 2.0)
      - usage_context (weight: 1.5)
      - target_subjects (weight: 1.0)
      - official_title (weight: 2.5)
      - education_levels (weight: 1.5)
    """
    
    def __init__(self, catalog_path: str = None):
        self.catalog: List[dict] = []
        self._doc_index: Dict[str, dict] = {}      # doc_id → catalog entry
        self._topic_index: Dict[str, List[str]] = {}  # keyword → [doc_ids]
        self._idf: Dict[str, float] = {}            # token → IDF score
        
        if catalog_path is None:
            # Auto-detect path
            catalog_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "outputs", "document_catalog_final.json"
            )
        
        self._load_catalog(catalog_path)
    
    def _load_catalog(self, path: str):
        """Load và index catalog."""
        path = os.path.abspath(path)
        if not os.path.exists(path):
            print(f"[CatalogMatcher] ⚠️ Không tìm thấy catalog: {path}")
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            self.catalog = json.load(f)
        
        # Build indexes
        self._doc_index = {}
        self._topic_index = {}
        doc_freq = Counter()  # For IDF calculation
        
        for item in self.catalog:
            doc_id = item["doc_id"]
            self._doc_index[doc_id] = item
            
            # Collect all searchable text for this entry
            all_tokens = set()
            
            # Index main_topics (term → doc_ids)
            for topic in item.get("main_topics", []):
                tokens = self._tokenize(topic)
                all_tokens.update(tokens)
                for word in tokens:
                    if len(word) > 1:
                        if word not in self._topic_index:
                            self._topic_index[word] = []
                        self._topic_index[word].append(doc_id)
                
                # Index n-grams from topics
                ngrams = self._extract_ngrams(topic)
                for ng in ngrams:
                    if ng not in self._topic_index:
                        self._topic_index[ng] = []
                    self._topic_index[ng].append(doc_id)
            
            # Index key_entities
            for entity in item.get("key_entities", []):
                tokens = self._tokenize(entity)
                all_tokens.update(tokens)
                for word in tokens:
                    if len(word) > 1:
                        if word not in self._topic_index:
                            self._topic_index[word] = []
                        self._topic_index[word].append(doc_id)
            
            # Index official_title
            title_tokens = self._tokenize(item.get("official_title", ""))
            all_tokens.update(title_tokens)
            for word in title_tokens:
                if len(word) > 1:
                    if word not in self._topic_index:
                        self._topic_index[word] = []
                    self._topic_index[word].append(doc_id)
            
            # Update document frequency for IDF
            for token in all_tokens:
                doc_freq[token] += 1
        
        # Compute IDF: log(N / df)
        N = len(self.catalog)
        self._idf = {}
        for token, df in doc_freq.items():
            self._idf[token] = math.log((N + 1) / (df + 1)) + 1  # Smoothed IDF
        
        print(f"[CatalogMatcher] Loaded {len(self.catalog)} entries, "
              f"{len(self._topic_index)} index keys, "
              f"{len(self._idf)} IDF terms.")
    
    # ─── Public API ──────────────────────────────────────────────────────────
    
    def match(self, query: str, top_k: int = 3, min_score: float = 2.0) -> List[str]:
        """
        Tìm các doc_id phù hợp nhất với câu hỏi.
        
        Returns: danh sách doc_id (đã xếp theo relevance giảm dần)
        """
        if not self.catalog:
            return []
        
        scored = self._score_all(query, min_score)
        return [doc_id for doc_id, _ in scored[:top_k]]
    
    def match_with_scores(self, query: str, top_k: int = 5, min_score: float = 1.0) -> List[Tuple[str, float]]:
        """Giống match() nhưng trả về cả điểm số."""
        if not self.catalog:
            return []
        
        scored = self._score_all(query, min_score)
        return scored[:top_k]
    
    def match_with_confidence(self, query: str, top_k: int = 5) -> Tuple[List[Tuple[str, float]], str]:
        """
        Match + trả về confidence tier.
        
        Returns:
            (results, confidence): results = [(doc_id, score)], confidence = "high"|"medium"|"low"|"none"
        """
        scored = self._score_all(query, min_score=1.0)
        
        if not scored:
            return [], "none"
        
        top_score = scored[0][1]
        if top_score >= 8.0:
            confidence = "high"
        elif top_score >= 4.0:
            confidence = "medium"
        else:
            confidence = "low"
        
        return scored[:top_k], confidence
    
    def get_entry(self, doc_id: str) -> Optional[dict]:
        """Lấy catalog entry theo doc_id."""
        return self._doc_index.get(doc_id)
    
    def get_related_docs(self, doc_id: str, max_related: int = 3) -> List[str]:
        """Tìm docs liên quan cùng topic với doc_id đã cho."""
        entry = self.get_entry(doc_id)
        if not entry:
            return []
        
        # Dùng main_topics của entry này để tìm docs khác
        topics = " ".join(entry.get("main_topics", []))
        results = self.match(topics, top_k=max_related + 1, min_score=3.0)
        return [r for r in results if r != doc_id][:max_related]
    
    # ─── Core Scoring ────────────────────────────────────────────────────────
    
    def _score_all(self, query: str, min_score: float) -> List[Tuple[str, float]]:
        """Score all catalog entries against query."""
        query_tokens = set(self._tokenize(query))
        query_ngrams = set(self._extract_ngrams(query))
        query_text_lower = query.lower()
        
        if not query_tokens:
            return []
        
        # Expand query with synonyms
        expanded_tokens = self._expand_synonyms(query_text_lower)
        
        # Detect education levels in query
        query_edu_levels = self._detect_education_levels(query_text_lower)
        
        # Phase 1: Quick candidate scan via topic_index
        candidate_scores: Dict[str, float] = {}
        
        # Scan with original tokens
        for token in query_tokens:
            if token in self._topic_index:
                idf = self._idf.get(token, 1.0)
                for doc_id in self._topic_index[token]:
                    candidate_scores[doc_id] = candidate_scores.get(doc_id, 0) + idf * 0.5
        
        # Scan with n-grams (higher weight)
        for ng in query_ngrams:
            if ng in self._topic_index:
                for doc_id in self._topic_index[ng]:
                    candidate_scores[doc_id] = candidate_scores.get(doc_id, 0) + 2.0
        
        # Phase 2: Deep scoring for top candidates (or all if few)
        if len(candidate_scores) > 40:
            top_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:40]
            candidates_to_score = [doc_id for doc_id, _ in top_candidates]
        else:
            candidates_to_score = list(candidate_scores.keys()) if candidate_scores else [
                item["doc_id"] for item in self.catalog
            ]
        
        # Deep scoring
        scored: List[Tuple[str, float]] = []
        for doc_id in candidates_to_score:
            item = self._doc_index.get(doc_id)
            if not item:
                continue
            score = self._score_item(query_tokens, query_ngrams, query_text_lower, 
                                     expanded_tokens, query_edu_levels, item)
            if score >= min_score:
                scored.append((doc_id, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def _score_item(self, query_tokens: set, query_ngrams: set, query_text: str,
                    expanded_tokens: set, query_edu_levels: set, item: dict) -> float:
        """Tính điểm phù hợp giữa câu hỏi và 1 mục catalog."""
        score = 0.0
        
        # ── main_topics (trọng số cao nhất) ──
        for topic in item.get("main_topics", []):
            topic_lower = topic.lower()
            topic_tokens = set(self._tokenize(topic))
            topic_ngrams = set(self._extract_ngrams(topic))
            
            # N-gram matching (cụm từ) — weight rất cao
            ngram_overlap = query_ngrams & topic_ngrams
            for ng in ngram_overlap:
                n_words = len(ng.split())
                score += n_words * 4.0  # 2-gram → 8.0, 3-gram → 12.0
            
            # Token matching with IDF
            token_overlap = query_tokens & topic_tokens
            for token in token_overlap:
                idf = self._idf.get(token, 1.0)
                score += 3.0 * min(idf, 3.0)  # Cap IDF weight
            
            # Synonym matching
            syn_match = expanded_tokens & topic_tokens
            non_direct = syn_match - query_tokens  # Only count new matches from synonyms
            if non_direct:
                score += len(non_direct) * 2.0
        
        # ── official_title ──
        title_tokens = set(self._tokenize(item.get("official_title", "")))
        title_ngrams = set(self._extract_ngrams(item.get("official_title", "")))
        
        # N-gram match on title
        title_ng_overlap = query_ngrams & title_ngrams
        for ng in title_ng_overlap:
            score += len(ng.split()) * 3.0
        
        # Token match on title
        title_overlap = query_tokens & title_tokens
        for token in title_overlap:
            idf = self._idf.get(token, 1.0)
            score += 2.5 * min(idf, 3.0)
        
        # ── key_entities ──
        for entity in item.get("key_entities", []):
            entity_tokens = set(self._tokenize(entity))
            overlap = query_tokens & entity_tokens
            if overlap:
                score += len(overlap) * 2.0
            # Synonym match on entities
            syn_overlap = expanded_tokens & entity_tokens
            non_direct = syn_overlap - query_tokens
            if non_direct:
                score += len(non_direct) * 1.5
        
        # ── usage_context ──
        context_text = item.get("usage_context", "")
        context_tokens = set(self._tokenize(context_text))
        context_ngrams = set(self._extract_ngrams(context_text))
        
        # N-gram match on context
        ctx_ng_overlap = query_ngrams & context_ngrams
        for ng in ctx_ng_overlap:
            score += len(ng.split()) * 2.0
        
        # Token match on context
        context_overlap = query_tokens & context_tokens
        if context_overlap:
            score += len(context_overlap) * 1.5
        
        # ── target_subjects ──
        for subject in item.get("target_subjects", []):
            subject_tokens = set(self._tokenize(subject))
            overlap = query_tokens & subject_tokens
            if overlap:
                score += len(overlap) * 1.0
            # Synonym
            syn_overlap = expanded_tokens & subject_tokens
            non_direct = syn_overlap - query_tokens
            if non_direct:
                score += len(non_direct) * 0.8
        
        # ── education_levels (bonus cho level match) ──
        if query_edu_levels:
            item_levels = set()
            for level in item.get("education_levels", []):
                item_levels.add(level.lower())
                # Also match against level map values
                for key, aliases in EDUCATION_LEVEL_MAP.items():
                    if level in aliases or level.lower() == key:
                        item_levels.add(key)
            
            level_overlap = query_edu_levels & item_levels
            if level_overlap:
                score += len(level_overlap) * 2.0
        
        return score
    
    # ─── Synonym Expansion ───────────────────────────────────────────────────
    
    def _expand_synonyms(self, query_text: str) -> set:
        """Expand query tokens with synonym groups."""
        query_tokens = set(self._tokenize(query_text))
        expanded = set(query_tokens)
        
        for syn_group in SYNONYM_GROUPS:
            # Check if any synonym in this group appears in query text
            matched = False
            for term in syn_group:
                if term in query_text:
                    matched = True
                    break
            
            if matched:
                # Add all tokenized forms of all synonyms in the group
                for term in syn_group:
                    expanded.update(self._tokenize(term))
        
        return expanded
    
    # ─── Education Level Detection ───────────────────────────────────────────
    
    def _detect_education_levels(self, query_text: str) -> set:
        """Detect education levels mentioned in query."""
        levels = set()
        for level_key, aliases in EDUCATION_LEVEL_MAP.items():
            for alias in aliases:
                if alias.lower() in query_text:
                    levels.add(level_key)
                    break
        return levels
    
    # ─── N-gram Extraction ───────────────────────────────────────────────────
    
    @staticmethod
    def _extract_ngrams(text: str, min_n: int = 2, max_n: int = 3) -> List[str]:
        """Extract 2-grams and 3-grams from text."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
        words = text.split()
        
        # Filter stopwords
        stopwords = {
            'là', 'của', 'và', 'các', 'có', 'trong', 'được', 'một', 'này',
            'cho', 'đến', 'theo', 'về', 'với', 'từ', 'tại', 'đã', 'khi',
            'không', 'những', 'như', 'nhưng', 'cũng', 'để', 'hay', 'hoặc',
            'nếu', 'thì', 'bởi', 'do', 'nên', 'sẽ', 'đang', 'vào', 'ra',
            'trên', 'dưới', 'số', 'ngày', 'bị', 'đó', 'hơn', 'nào',
            'mà', 'sau', 'trước',
        }
        # Keep words but filter single-char
        words = [w for w in words if len(w) > 1]
        
        ngrams = []
        for n in range(min_n, max_n + 1):
            for i in range(len(words) - n + 1):
                gram = words[i:i+n]
                # Skip n-grams that are entirely stopwords
                if all(w in stopwords for w in gram):
                    continue
                ngrams.append(" ".join(gram))
        
        return ngrams
    
    # ─── Tokenization ────────────────────────────────────────────────────────
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize đơn giản cho tiếng Việt."""
        text = text.lower().strip()
        # Bỏ dấu câu nhưng giữ dấu tiếng Việt
        text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
        tokens = text.split()
        # Filter stopwords tiếng Việt cơ bản
        stopwords = {
            'là', 'của', 'và', 'các', 'có', 'trong', 'được', 'một', 'này',
            'cho', 'đến', 'theo', 'về', 'với', 'từ', 'tại', 'đã', 'khi',
            'không', 'những', 'như', 'nhưng', 'cũng', 'để', 'hay', 'hoặc',
            'nếu', 'thì', 'bởi', 'do', 'nên', 'sẽ', 'đang', 'vào', 'ra',
            'trên', 'dưới', 'quy', 'định', 'quy định', 'số', 'ngày',
        }
        return [t for t in tokens if t not in stopwords and len(t) > 1]


# ─── Standalone test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    matcher = CatalogMatcher()
    
    test_queries = [
        "Hội phụ huynh có được thu tiền mua điều hòa cho lớp không?",
        "Giáo viên tiểu học cần bằng cấp gì?",
        "So sánh mức học phí đại học công lập giữa NĐ 238 và NĐ 84",
        "Điều kiện miễn học phí cho hộ nghèo",
        "Thủ tục thành lập trường đại học tư thục",
        "Tất cả quy định về dạy thêm học thêm",
        "Sinh viên sư phạm có phải bồi hoàn kinh phí không?",
        # Thêm benchmark queries
        "Liệt kê các đối tượng được miễn hoặc giảm học phí theo quy định pháp luật hiện hành",
        "Hệ thống giáo dục quốc dân gồm những cấp học và trình độ đào tạo nào?",
        "Con tôi thuộc hộ nghèo đang học tiểu học, có được miễn học phí không?",
        "Tiêu chuẩn chức danh nghề nghiệp giảng viên đại học hạng I được quy định như thế nào?",
        "Hồ sơ đăng ký mở ngành đào tạo trình độ đại học gồm những thành phần gì?",
        "Quy trình kiểm định chất lượng cơ sở giáo dục đại học được thực hiện như thế nào?",
        "Giáo dục thường xuyên là gì theo quy định của Luật Giáo dục 2019?",
    ]
    
    for q in test_queries:
        results, confidence = matcher.match_with_confidence(q, top_k=3)
        print(f"\n❓ {q}")
        print(f"   Confidence: {confidence}")
        for doc_id, score in results:
            entry = matcher.get_entry(doc_id)
            title = entry.get("official_title", "?") if entry else "?"
            print(f"   📄 {doc_id} (score: {score:.1f}) — {title[:60]}")
