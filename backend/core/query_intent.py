"""
Query Intent Classifier — 2-Step Qwen2.5 Pipeline
===================================================
Bước 1: Classify query → 1 trong 7 loại (max_new_tokens=5)
Bước 2: Extract JSON schema tương ứng (max_new_tokens=200)

Tận dụng Qwen2.5-3B-Instruct chạy local trên M1.
Fallback: Regex-based classification nếu model fail.
"""

import json
import re
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# ─── QueryIntent Data Class ──────────────────────────────────────────────────

VALID_TYPES = [
    "single_lookup",
    "comparison",
    "listing",
    "eligibility_check",
    "procedure",
    "cross_document",
    "definition",
]


@dataclass
class QueryIntent:
    """Cấu trúc dữ liệu kết quả phân tích câu hỏi."""
    type: str = "single_lookup"
    documents: List[str] = field(default_factory=list)
    topic: str = ""
    keywords: List[str] = field(default_factory=list)
    # Optional fields tùy type
    article_hint: Optional[str] = None          # single_lookup
    sub_queries: Optional[List[str]] = None     # comparison
    criteria: Optional[List[str]] = None        # comparison
    expand_search: bool = False                 # listing
    subject: Optional[str] = None               # eligibility_check
    check_aspects: Optional[List[str]] = None   # eligibility_check
    procedure_keywords: Optional[List[str]] = None  # procedure
    search_scope: str = "scoped"                # cross_document: "global" | "scoped"
    term: Optional[str] = None                  # definition
    # Catalog-driven fields (v2)
    catalog_confidence: str = "none"            # "high" | "medium" | "low" | "none"
    catalog_metadata: dict = field(default_factory=dict)  # doc_id → {main_topics, education_levels, score}
    # Metadata
    _classify_time: float = 0.0
    _extract_time: float = 0.0
    _fallback_used: bool = False

    def to_dict(self) -> dict:
        """Serialize for UI display / logging ( bỏ private fields)."""
        d = asdict(self)
        return {k: v for k, v in d.items() if not k.startswith("_") and v is not None}

    @property
    def total_time_ms(self) -> float:
        return (self._classify_time + self._extract_time) * 1000


# ─── Regex Pre-Extraction ────────────────────────────────────────────────────

# Pattern nhận diện tên văn bản pháp luật
DOC_PATTERNS = [
    # "Nghị định 81/2021/NĐ-CP", "NĐ 81/2021/NĐ-CP"
    r'(?:Nghị\s*định|NĐ|Thông\s*tư|TT|Luật|Quyết\s*định|QĐ)\s*(?:số\s*)?(\d+[\/-]\d{4}[\/-]?[A-ZĐa-zđ\-]*)',
    # "Nghị định 81", "NĐ 81", "Luật 43"
    r'(?:Nghị\s*định|NĐ|Thông\s*tư|TT|Luật)\s*(?:số\s*)?(\d{2,3})\b',
]

# Pattern nhận diện số điều
ARTICLE_PATTERN = r'[Đđ]iều\s+(\d+)'

# Regex-based intent classification patterns
COMPARISON_PATTERNS = [
    r'so\s*sánh', r'khác\s*nhau', r'giống\s*nhau', r'sự\s*khác\s*biệt',
    r'phân\s*biệt', r'giữa\s*.*\s*và', r'khác\s*gì',
]
LISTING_PATTERNS = [
    r'liệt\s*kê', r'các\s+(?:điều\s*kiện|trường\s*hợp|đối\s*tượng|loại|hình\s*thức)',
    r'toàn\s*bộ', r'tất\s*cả', r'những\s+(?:ai|gì|nào)',
    r'gồm\s+(?:những|các)',
]
ELIGIBILITY_PATTERNS = [
    r'có\s+được\s+(?:miễn|giảm|hưởng|nhận)',
    r'(?:con\s+tôi|tôi)\s+.*\s+(?:được|có\s+thể|thuộc)',
    r'có\s+đủ\s+điều\s+kiện', r'thuộc\s+diện',
]
PROCEDURE_PATTERNS = [
    r'thủ\s*tục', r'hồ\s*sơ', r'quy\s*trình', r'cần\s+(?:giấy\s+tờ|những\s+gì)',
    r'làm\s+(?:thế\s+nào|sao)\s+để', r'đăng\s*ký\s+(?:như|thế)',
    r'nộp\s+(?:ở\s+đâu|cho\s+ai)',
]
DEFINITION_PATTERNS = [
    r'là\s+gì', r'định\s*nghĩa', r'(?:khái\s*niệm|nghĩa\s*là)',
    r'được\s+hiểu\s+(?:là|như)',
]


def pre_extract_docs(query: str) -> List[str]:
    """Regex trích xuất sơ bộ tên văn bản từ query (không cần LLM)."""
    found = []
    for pattern in DOC_PATTERNS:
        matches = re.findall(pattern, query, re.IGNORECASE)
        found.extend(matches)
    # Deduplicate giữ thứ tự
    seen = set()
    result = []
    for doc in found:
        doc_clean = doc.strip().rstrip('/')
        if doc_clean not in seen:
            seen.add(doc_clean)
            result.append(doc_clean)
    return result


def pre_extract_article(query: str) -> Optional[str]:
    """Regex trích xuất số Điều từ query."""
    match = re.search(ARTICLE_PATTERN, query)
    return match.group(1) if match else None


def regex_classify(query: str) -> str:
    """Fallback: Phân loại câu hỏi bằng regex khi model fail."""
    q = query.lower()
    
    # Thứ tự ưu tiên: comparison > eligibility > procedure > listing > definition > cross_doc > single
    for p in COMPARISON_PATTERNS:
        if re.search(p, q):
            return "comparison"
    for p in ELIGIBILITY_PATTERNS:
        if re.search(p, q):
            return "eligibility_check"
    for p in PROCEDURE_PATTERNS:
        if re.search(p, q):
            return "procedure"
    for p in LISTING_PATTERNS:
        if re.search(p, q):
            return "listing"
    for p in DEFINITION_PATTERNS:
        if re.search(p, q):
            return "definition"
    
    # Nếu không nhận diện doc nào → cross_document
    docs = pre_extract_docs(query)
    if not docs:
        return "cross_document"
    
    return "single_lookup"


# ─── Qwen Prompts ────────────────────────────────────────────────────────────

CLASSIFY_PROMPT = """Phân loại câu hỏi pháp luật giáo dục vào 1 trong 7 loại:
single_lookup | comparison | listing | eligibility_check | procedure | cross_document | definition

Ví dụ:
"So sánh mức học phí NĐ 81 và NĐ 97" → comparison
"Điều 15 NĐ 81 quy định gì" → single_lookup
"Các đối tượng được miễn học phí" → listing
"Con tôi học lớp 1 có được miễn học phí không" → eligibility_check
"Thủ tục xin miễn giảm học phí cần gì" → procedure
"Tất cả quy định về học phí mầm non" → cross_document
"Trường chuẩn quốc gia là gì" → definition

Câu hỏi: {query}
Loại:"""

# Schema templates cho Bước 2 — mỗi type 1 template riêng
# QUAN TRỌNG: Qwen CHỈ được trích xuất số hiệu dạng chuẩn (VD: "238/2025/NĐ-CP").
# KHÔNG được tự viết tên tự nhiên (VD: "Luật Giáo dục 2019"). Nếu không rõ số hiệu → để [].
_DOC_RULE = 'documents: mảng số hiệu văn bản dạng "XX/YYYY/LOẠI" (VD: "238/2025/NĐ-CP", "43/2019/QH14"). KHÔNG viết tên như "Luật Giáo dục 2019". Nếu không rõ số hiệu thì để [].'

EXTRACT_SCHEMAS = {
    "single_lookup": {
        "instruction": f'Trích xuất: {_DOC_RULE} topic, article_hint (số điều nếu có), keywords.',
        "example_q": "Điều 15 Nghị định 81/2021 quy định gì",
        "example_a": '{"documents":["81/2021/NĐ-CP"],"topic":"nội dung điều 15","article_hint":"15","keywords":["điều 15"]}',
    },
    "comparison": {
        "instruction": f'Trích xuất: {_DOC_RULE} topic, sub_queries (1 query cho mỗi văn bản), criteria (tiêu chí so sánh), keywords.',
        "example_q": "So sánh mức miễn giảm học phí giữa Nghị định 81 và Nghị định 97",
        "example_a": '{"documents":["81/2021/NĐ-CP","97/2023/NĐ-CP"],"topic":"mức miễn giảm học phí","sub_queries":["mức miễn giảm học phí NĐ 81/2021","mức miễn giảm học phí NĐ 97/2023"],"criteria":["đối tượng","mức miễn giảm","điều kiện"],"keywords":["miễn giảm","học phí"]}',
    },
    "listing": {
        "instruction": f'Trích xuất: {_DOC_RULE} topic, keywords, expand_search (true/false).',
        "example_q": "Các đối tượng được miễn học phí theo Nghị định 81",
        "example_a": '{"documents":["81/2021/NĐ-CP"],"topic":"đối tượng được miễn học phí","keywords":["miễn học phí","đối tượng","trường hợp"],"expand_search":true}',
    },
    "eligibility_check": {
        "instruction": f'Trích xuất: {_DOC_RULE} topic, subject (ai/cái gì cần kiểm tra), check_aspects (các khía cạnh cần kiểm), keywords.',
        "example_q": "Con tôi học lớp 1 trường công có được miễn học phí không",
        "example_a": '{"documents":["81/2021/NĐ-CP"],"topic":"miễn học phí học sinh lớp 1","subject":"học sinh lớp 1 trường công lập","check_aspects":["đối tượng áp dụng","điều kiện","giấy tờ"],"keywords":["miễn học phí","tiểu học","công lập"]}',
    },
    "procedure": {
        "instruction": f'Trích xuất: {_DOC_RULE} topic, procedure_keywords (từ liên quan thủ tục), keywords.',
        "example_q": "Thủ tục xin miễn giảm học phí cần giấy tờ gì",
        "example_a": '{"documents":["81/2021/NĐ-CP"],"topic":"thủ tục miễn giảm học phí","procedure_keywords":["hồ sơ","thủ tục","thời hạn","nộp"],"keywords":["miễn giảm học phí","hồ sơ","thủ tục"]}',
    },
    "cross_document": {
        "instruction": f'Trích xuất: {_DOC_RULE} topic, keywords, search_scope ("global").',
        "example_q": "Tất cả quy định về học phí mầm non hiện hành",
        "example_a": '{"documents":[],"topic":"học phí mầm non","keywords":["học phí","mầm non"],"search_scope":"global"}',
    },
    "definition": {
        "instruction": f'Trích xuất: {_DOC_RULE} topic, term (thuật ngữ), keywords.',
        "example_q": "Trường chuẩn quốc gia là gì",
        "example_a": '{"documents":[],"topic":"định nghĩa trường chuẩn quốc gia","term":"trường chuẩn quốc gia","keywords":["trường chuẩn quốc gia","tiêu chí","định nghĩa"]}',
    },
}

EXTRACT_PROMPT_TEMPLATE = """Trích xuất thông tin từ câu hỏi pháp luật giáo dục. Trả về JSON.
{instruction}

Ví dụ:
Q: {example_q}
A: {example_a}

Văn bản nhận diện: {pre_docs}
Câu hỏi: {query}
JSON:"""


# ─── QwenIntentClassifier ────────────────────────────────────────────────────

class QwenIntentClassifier:
    """
    2-Step Query Intent Classifier sử dụng Qwen2.5 local.
    
    Bước 1: Classify → 1 label (max_new_tokens=5)
    Bước 2: Extract → JSON schema (max_new_tokens=200)
    
    Fallback: Regex-based nếu model fail.
    
    Args:
        use_regex_only: Nếu True, bỏ qua Qwen hoàn toàn, chỉ dùng regex (90% accuracy).
                        Dùng cho Streamlit vì PyTorch model loading crash trong forked process.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        doc_registry: dict = None,
        device: str = "auto",
        use_regex_only: bool = False,
        catalog_matcher=None,
    ):
        self.model_name = model_name
        self.doc_registry = doc_registry or {}
        self.device = device
        self.use_regex_only = use_regex_only
        self.catalog_matcher = catalog_matcher
        self._model = None
        self._tokenizer = None
        self._load_error = "Regex-only mode (Qwen disabled)" if use_regex_only else None
    
    def _load_model(self):
        """Lazy load model + tokenizer (chỉ load 1 lần)."""
        if self._model is not None or self._load_error is not None:
            return
        
        try:
            import os
            # CRITICAL: Tắt MPS memory watermark limit.
            # Khi CrossEncoder (sentence-transformers) đã load trên MPS trước,
            # Qwen sẽ segfault nếu watermark chặn allocation. 
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"[QueryIntent] Loading model: {self.model_name}...")
            t0 = time.time()
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            
            # Detect target device
            if self.device == "auto":
                if torch.cuda.is_available():
                    target_device = "cuda"
                elif torch.backends.mps.is_available():
                    target_device = "mps"
                else:
                    target_device = "cpu"
            else:
                target_device = self.device
            
            # Load model lên CPU trước (an toàn nhất), rồi move sang MPS nếu cần
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            # Move sang MPS/CUDA nếu có
            if target_device != "cpu":
                try:
                    self._model = self._model.to(target_device)
                    print(f"[QueryIntent] Model moved to {target_device}")
                except Exception as move_err:
                    print(f"[QueryIntent] ⚠️ Không move được sang {target_device}: {move_err}")
                    print(f"[QueryIntent] Giữ model trên CPU.")
                    target_device = "cpu"
            
            self._model.eval()
            print(f"[QueryIntent] Model loaded in {time.time() - t0:.1f}s on {target_device}")
            
        except Exception as e:
            self._load_error = str(e)
            print(f"[QueryIntent] ⚠️ Failed to load model: {e}")
            print(f"[QueryIntent] Sẽ dùng Regex fallback cho mọi query.")
    
    def _generate(self, prompt: str, max_new_tokens: int = 10) -> str:
        """Gọi Qwen generate. Trả về text output."""
        import torch
        
        messages = [
            {"role": "system", "content": "Bạn là trợ lý phân tích câu hỏi pháp luật. Trả lời ngắn gọn, chính xác."},
            {"role": "user", "content": prompt},
        ]
        
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self._tokenizer([text], return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # Greedy decoding cho deterministic
                temperature=1.0,
                top_p=1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        
        # Chỉ lấy phần generated (bỏ prompt)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        result = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        return result.strip()

    # ── Bước 1: Classify ──────────────────────────────────────────────────
    
    def _classify(self, query: str) -> str:
        """Bước 1: Phân loại query thành 1 trong 7 types."""
        prompt = CLASSIFY_PROMPT.format(query=query)
        
        try:
            raw = self._generate(prompt, max_new_tokens=10)
            # Tìm type trong output
            raw_lower = raw.lower().strip()
            for t in VALID_TYPES:
                if t in raw_lower:
                    return t
            # Không match → fallback regex
            print(f"[QueryIntent] Classify output không match: '{raw}' → regex fallback")
            return regex_classify(query)
        except Exception as e:
            print(f"[QueryIntent] Classify error: {e} → regex fallback")
            return regex_classify(query)
    
    # ── Bước 2: Extract ───────────────────────────────────────────────────
    
    def _extract(self, query: str, intent_type: str, pre_docs: List[str]) -> dict:
        """Bước 2: Trích xuất JSON schema theo type đã classify."""
        schema = EXTRACT_SCHEMAS.get(intent_type, EXTRACT_SCHEMAS["single_lookup"])
        
        prompt = EXTRACT_PROMPT_TEMPLATE.format(
            instruction=schema["instruction"],
            example_q=schema["example_q"],
            example_a=schema["example_a"],
            pre_docs=", ".join(pre_docs) if pre_docs else "không rõ",
            query=query,
        )
        
        try:
            raw = self._generate(prompt, max_new_tokens=200)
            parsed = self._safe_parse_json(raw)
            if parsed:
                return parsed
            print(f"[QueryIntent] Extract JSON parse fail: '{raw[:100]}...'")
            return {}
        except Exception as e:
            print(f"[QueryIntent] Extract error: {e}")
            return {}
    
    # ── Hàm Public chính ──────────────────────────────────────────────────
    
    def analyze(self, query: str) -> QueryIntent:
        """
        Phân tích câu hỏi → QueryIntent.
        Pipeline: pre_extract_docs → ALWAYS Catalog Matching → Bước 1 (Classify) → Bước 2 (Extract) → QueryIntent
        """
        # Step 0: Regex pre-extraction (luôn chạy, không cần model)
        pre_docs = pre_extract_docs(query)
        article_hint = pre_extract_article(query)
        
        # Resolve docs through registry
        regex_resolved = []
        for doc in pre_docs:
            resolved = self._resolve_doc(doc)
            if resolved:
                regex_resolved.append(resolved)
        
        # Step 0.5: ALWAYS run Catalog Matching (không chỉ fallback)
        catalog_confidence = "none"
        catalog_metadata = {}
        catalog_resolved = []
        
        if self.catalog_matcher is not None:
            try:
                catalog_results, catalog_confidence = self.catalog_matcher.match_with_confidence(
                    query, top_k=5
                )
                if catalog_results:
                    for doc_id, score in catalog_results:
                        # Resolve qua doc_registry
                        resolved = self._resolve_doc(doc_id)
                        if resolved and resolved not in catalog_resolved:
                            catalog_resolved.append(resolved)
                        
                        # Lưu metadata cho retriever
                        entry = self.catalog_matcher.get_entry(doc_id)
                        if entry:
                            catalog_metadata[doc_id] = {
                                "main_topics": entry.get("main_topics", []),
                                "education_levels": entry.get("education_levels", []),
                                "score": score,
                                "resolved": resolved or doc_id,
                            }
                    
                    print(f"[QueryIntent] Catalog matched ({catalog_confidence}): "
                          f"{[(d, f'{s:.1f}') for d, s in catalog_results[:3]]}")
            except Exception as e:
                print(f"[QueryIntent] Catalog matching error: {e}")
        
        # Merge: regex docs + catalog docs
        resolved_docs = self._merge_doc_sources(regex_resolved, catalog_resolved, catalog_confidence)
        
        # Kiểm tra model có sẵn không
        self._load_model()
        
        if self._model is None:
            # Full regex fallback
            intent_type = regex_classify(query)
            intent = self._build_intent_from_regex(
                query, intent_type, resolved_docs, article_hint
            )
            intent.catalog_confidence = catalog_confidence
            intent.catalog_metadata = catalog_metadata
            return intent
        
        # Step 1: Classify
        t0 = time.time()
        intent_type = self._classify(query)
        classify_time = time.time() - t0
        
        # Step 2: Extract
        t1 = time.time()
        extracted = self._extract(query, intent_type, pre_docs)
        extract_time = time.time() - t1
        
        # Build QueryIntent từ extracted data
        intent = self._build_intent(
            query, intent_type, extracted, resolved_docs, article_hint
        )
        intent._classify_time = classify_time
        intent._extract_time = extract_time
        intent.catalog_confidence = catalog_confidence
        intent.catalog_metadata = catalog_metadata
        
        return intent
    
    def _merge_doc_sources(
        self, regex_docs: List[str], catalog_docs: List[str], catalog_confidence: str
    ) -> List[str]:
        """
        Merge doc sources từ regex và catalog.
        
        Logic:
        - Regex docs luôn được giữ (ưu tiên cao vì user ghi rõ)
        - Catalog docs bổ sung thêm, số lượng tùy confidence:
          - high: thêm tối đa 3 catalog docs
          - medium: thêm tối đa 2 catalog docs 
          - low: thêm tối đa 1 catalog doc
          - none: không thêm
        """
        merged = list(regex_docs)  # Regex docs đầu tiên
        
        max_catalog = {"high": 3, "medium": 2, "low": 1, "none": 0}
        limit = max_catalog.get(catalog_confidence, 0)
        
        # Nếu không có regex docs → cho phép catalog nhiều hơn
        if not regex_docs and catalog_confidence in ("high", "medium"):
            limit = 5  # Catalog là nguồn chính
        
        added = 0
        for doc in catalog_docs:
            if doc not in merged and added < limit:
                merged.append(doc)
                added += 1
        
        return merged
    
    # ── Helpers ───────────────────────────────────────────────────────────
    
    def _resolve_doc(self, raw_doc: str) -> Optional[str]:
        """Dùng doc_registry để resolve tên văn bản → so_hieu chuẩn."""
        if not self.doc_registry:
            return raw_doc
        
        # Thử exact match trước
        if raw_doc in self.doc_registry:
            return self.doc_registry[raw_doc]
        
        # Thử bỏ hậu tố loại văn bản
        clean = re.sub(r'[\/-]?(NĐ|TT|QĐ)[\/-]?(CP|BGDĐT)?$', '', raw_doc, flags=re.IGNORECASE).strip('/-')
        if clean in self.doc_registry:
            return self.doc_registry[clean]
        
        # Thử chỉ lấy số đầu
        num_match = re.match(r'(\d+)', raw_doc)
        if num_match:
            num = num_match.group(1)
            if num in self.doc_registry:
                return self.doc_registry[num]
        
        return raw_doc  # Trả nguyên nếu không resolve được
    
    def _build_intent(
        self, query: str, intent_type: str, extracted: dict,
        resolved_docs: List[str], article_hint: Optional[str]
    ) -> QueryIntent:
        """Xây dựng QueryIntent từ dữ liệu extracted + pre-extracted."""
        # Documents: Qwen docs chỉ được chấp nhận nếu đúng format số hiệu
        qwen_docs = extracted.get("documents", []) or []
        
        # Validate: chỉ giữ doc có dấu "/" (format chuẩn như 238/2025/NĐ-CP)
        # Loại bỏ tên tự nhiên như "Luật Giáo dục 2019", "Luật Nhà giáo 2025"
        valid_qwen_docs = []
        for d in qwen_docs:
            d_str = str(d).strip()
            if '/' in d_str:  # Format chuẩn: XX/YYYY/TYPE
                resolved = self._resolve_doc(d_str)
                if resolved:
                    valid_qwen_docs.append(resolved)
        
        # Merge: resolved_docs (catalog+regex) là nền tảng, Qwen bổ sung thêm
        seen = set(resolved_docs)
        final_docs = list(resolved_docs)
        for d in valid_qwen_docs:
            if d not in seen:
                final_docs.append(d)
                seen.add(d)
        
        if not final_docs:
            final_docs = resolved_docs
        
        intent = QueryIntent(
            type=intent_type,
            documents=final_docs,
            topic=extracted.get("topic", ""),
            keywords=extracted.get("keywords", []),
        )
        
        # Type-specific fields
        if intent_type == "single_lookup":
            intent.article_hint = extracted.get("article_hint") or article_hint
        
        elif intent_type == "comparison":
            intent.sub_queries = extracted.get("sub_queries", [])
            intent.criteria = extracted.get("criteria", [])
            # Nếu LLM không sinh sub_queries, tạo tự động
            if not intent.sub_queries and len(final_docs) >= 2:
                topic = intent.topic or query
                intent.sub_queries = [f"{topic} {doc}" for doc in final_docs]
        
        elif intent_type == "listing":
            intent.expand_search = extracted.get("expand_search", True)
        
        elif intent_type == "eligibility_check":
            intent.subject = extracted.get("subject", "")
            intent.check_aspects = extracted.get("check_aspects", [])
        
        elif intent_type == "procedure":
            intent.procedure_keywords = extracted.get(
                "procedure_keywords",
                ["hồ sơ", "thủ tục", "thời hạn", "nộp", "cơ quan"]
            )
        
        elif intent_type == "cross_document":
            intent.search_scope = extracted.get("search_scope", "global")
        
        elif intent_type == "definition":
            intent.term = extracted.get("term", "")
        
        return intent
    
    def _build_intent_from_regex(
        self, query: str, intent_type: str,
        resolved_docs: List[str], article_hint: Optional[str]
    ) -> QueryIntent:
        """Xây dựng QueryIntent từ regex fallback (không có LLM extract)."""
        # Trích keywords đơn giản từ query
        q_lower = query.lower()
        keywords = [w for w in q_lower.split() if len(w) > 2][:5]
        
        intent = QueryIntent(
            type=intent_type,
            documents=resolved_docs,
            topic=query[:100],
            keywords=keywords,
            _fallback_used=True,
        )
        
        if intent_type == "single_lookup":
            intent.article_hint = article_hint
        elif intent_type == "comparison" and len(resolved_docs) >= 2:
            intent.sub_queries = [f"{query} {doc}" for doc in resolved_docs]
            intent.criteria = []
        elif intent_type == "listing":
            intent.expand_search = True
        elif intent_type == "eligibility_check":
            intent.subject = query[:80]
            intent.check_aspects = ["đối tượng áp dụng", "điều kiện"]
        elif intent_type == "procedure":
            intent.procedure_keywords = ["hồ sơ", "thủ tục", "thời hạn", "nộp"]
        elif intent_type == "cross_document":
            intent.search_scope = "global"
        elif intent_type == "definition":
            # Trích term
            match = re.search(r'(.+?)\s+là\s+gì', q_lower)
            intent.term = match.group(1).strip() if match else query[:50]
        
        return intent
    
    @staticmethod
    def _safe_parse_json(text: str) -> Optional[dict]:
        """Parse JSON an toàn từ output LLM."""
        if not text or not text.strip():
            return None
        text = text.strip()
        # Bỏ markdown wrapper
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        # Tìm JSON object
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Thử parse trực tiếp
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None


# ─── Mapping QueryIntent.type → legacy query_type (backward compat) ──────────

INTENT_TO_LEGACY = {
    "single_lookup": "tra_cuu",
    "comparison": "so_sanh",
    "listing": "tong_hop",
    "eligibility_check": "tra_cuu",
    "procedure": "thu_tuc",
    "cross_document": "tong_hop",
    "definition": "tra_cuu",
}


def intent_to_legacy_type(intent: QueryIntent) -> str:
    """Chuyển đổi QueryIntent.type → legacy query_type (4 loại cũ)."""
    return INTENT_TO_LEGACY.get(intent.type, "tong_hop")
