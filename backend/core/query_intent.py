"""
Query Intent Classifier — 1-Step Fine-tuned Qwen2.5-1.5B Pipeline
===================================================================
Model fine-tuned: manhcuong2005/qwen2.5-1.5b-legal-edu
Output: JSON 1 bước chứa {type, documents, topic, sub_queries, keywords}

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
    sub_queries: Optional[List[str]] = None
    # Optional fields tùy type
    article_hint: Optional[str] = None          # single_lookup
    criteria: Optional[List[str]] = None        # comparison
    expand_search: bool = False                 # listing
    subject: Optional[str] = None               # eligibility_check
    check_aspects: Optional[List[str]] = None   # eligibility_check
    procedure_keywords: Optional[List[str]] = None  # procedure
    search_scope: str = "scoped"                # cross_document: "global" | "scoped"
    term: Optional[str] = None                  # definition
    # Metadata
    _generation_time: float = 0.0
    _fallback_used: bool = False

    def to_dict(self) -> dict:
        """Serialize for UI display / logging ( bỏ private fields)."""
        d = asdict(self)
        return {k: v for k, v in d.items() if not k.startswith("_") and v is not None}

    @property
    def total_time_ms(self) -> float:
        return self._generation_time * 1000


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


# ─── System Prompt — Giống hệt lúc train ────────────────────────────────────
# ⚠️ QUAN TRỌNG: Prompt này PHẢI giống hệt trong format_for_qwen.py khi tạo data train.
# Nếu sửa ở đây → PHẢI sửa ở format_for_qwen.py → retrain model.

SYSTEM_PROMPT = "Bạn là trợ lý phân tích câu hỏi pháp luật. Nhận diện loại câu hỏi, trích xuất từ khóa, văn bản và PHẢI CHIA NHỎ câu hỏi phức tạp thành danh sách `sub_queries` đơn giản để dễ tìm kiếm. Trả về đúng định dạng JSON không chứa markdown."


# ─── QwenIntentClassifier ────────────────────────────────────────────────────

class QwenIntentClassifier:
    """
    1-Step Query Intent Classifier sử dụng Qwen2.5-1.5B fine-tuned.
    
    Model trả ra JSON 1 bước gồm: type, documents, topic, sub_queries, keywords.
    Không cần 2 bước classify + extract nữa → nhanh hơn, chính xác hơn.
    
    Fallback: Regex-based nếu model fail.
    """
    
    def __init__(
        self,
        model_name: str = "manhcuong2005/qwen2.5-1.5b-legal-edu-v5",
        doc_registry: dict = None,
        device: str = "auto",
        use_regex_only: bool = False,
    ):
        self.model_name = model_name
        self.doc_registry = doc_registry or {}
        self.device = device
        self.use_regex_only = use_regex_only
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
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"[QueryIntent] Loading fine-tuned model: {self.model_name}...")
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
            
            # FP16 cho MPS/CUDA (tiết kiệm ~50% RAM: 3GB thay vì 6GB)
            # FP32 cho CPU (MPS/CUDA hỗ trợ FP16 native)
            use_dtype = torch.float16 if target_device != "cpu" else torch.float32
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=use_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            # Move sang MPS/CUDA nếu có
            if target_device != "cpu":
                try:
                    self._model = self._model.to(target_device)
                    print(f"[QueryIntent] Model moved to {target_device} (dtype={use_dtype})")
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
    
    def _generate(self, query: str, max_new_tokens: int = 200, force_json: bool = False) -> str:
        """Gọi fine-tuned model: nhận query, trả JSON string.
        
        Args:
            force_json: Nếu True, prepend '{"' vào đầu output để ép model sinh JSON.
        """
        import torch
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # TRICK: Nếu force_json, thêm '{"' vào cuối prompt để ép model tiếp tục sinh JSON
        if force_json:
            text += '{"'
        
        inputs = self._tokenizer([text], return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # Greedy decoding
                num_beams=1,              # Không dùng beam search
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        
        # Chỉ lấy phần generated (bỏ prompt)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        result = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Nếu force_json, prepend '{"' vào result vì nó là phần model tiếp tục
        if force_json:
            result = '{"' + result
        
        return result.strip()

    # ── Hàm Public chính ──────────────────────────────────────────────────
    
    def analyze(self, query: str) -> QueryIntent:
        """
        Phân tích câu hỏi → QueryIntent.
        Pipeline 1-step: Regex pre-extract → Fine-tuned Qwen → Build QueryIntent
        Có retry với force_json nếu lần đầu model không sinh JSON.
        """
        # Step 0: Regex pre-extraction (luôn chạy, không cần model)
        pre_docs = pre_extract_docs(query)
        article_hint = pre_extract_article(query)
        
        # Resolve docs through registry
        resolved_docs = []
        for doc in pre_docs:
            resolved = self._resolve_doc(doc)
            if resolved:
                resolved_docs.append(resolved)
        
        # Kiểm tra model có sẵn không
        self._load_model()
        
        if self._model is None:
            # Full regex fallback
            intent_type = regex_classify(query)
            intent = self._build_intent_from_regex(
                query, intent_type, resolved_docs, article_hint
            )
            return intent
        
        # Step 1: 1-Step Generation (classify + extract trong 1 lần gọi)
        t0 = time.time()
        parsed = None
        
        try:
            # Lần 1: Gọi bình thường
            raw_output = self._generate(query)
            parsed = self._safe_parse_json(raw_output)
            
            # Lần 2: Nếu không phải JSON → retry với force_json=True
            if not parsed:
                print(f"[QueryIntent] ⚠️ Output không phải JSON → retry với force_json...")
                raw_output = self._generate(query, force_json=True)
                parsed = self._safe_parse_json(raw_output)
                
        except Exception as e:
            print(f"[QueryIntent] Generation error: {e} → regex fallback")
            parsed = None
        generation_time = time.time() - t0
        
        if not parsed:
            # JSON parse fail → regex fallback
            print(f"[QueryIntent] JSON parse fail → regex fallback")
            intent_type = regex_classify(query)
            intent = self._build_intent_from_regex(
                query, intent_type, resolved_docs, article_hint
            )
            intent._generation_time = generation_time
            return intent
        
        # Build QueryIntent từ parsed JSON
        intent = self._build_intent_from_model(
            query, parsed, resolved_docs, article_hint
        )
        intent._generation_time = generation_time
        
        return intent
    
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
    
    def _build_intent_from_model(
        self, query: str, parsed: dict,
        resolved_docs: List[str], article_hint: Optional[str]
    ) -> QueryIntent:
        """Xây dựng QueryIntent từ JSON output của model fine-tuned."""
        # Validate type
        model_type = parsed.get("type", "single_lookup")
        if model_type not in VALID_TYPES:
            model_type = regex_classify(query)
        
        # Documents: merge regex docs + model docs
        model_docs = parsed.get("documents", []) or []
        valid_model_docs = []
        for d in model_docs:
            d_str = str(d).strip()
            if '/' in d_str:  # Format chuẩn: XX/YYYY/TYPE
                resolved = self._resolve_doc(d_str)
                if resolved:
                    valid_model_docs.append(resolved)
        
        # Regex docs ưu tiên, model docs bổ sung
        seen = set(resolved_docs)
        final_docs = list(resolved_docs)
        for d in valid_model_docs:
            if d not in seen:
                final_docs.append(d)
                seen.add(d)
        
        # Sub-queries từ model
        sub_queries = parsed.get("sub_queries", []) or []
        if not sub_queries:
            sub_queries = [query]  # Fallback: dùng nguyên câu hỏi gốc
        # Topic fallback: nếu model không trả topic, tự sinh từ sub_queries hoặc query
        topic = parsed.get("topic", "") or ""
        if not topic.strip():
            if sub_queries and sub_queries[0] != query:
                topic = sub_queries[0][:80]
            else:
                # Trích topic từ query: bỏ các từ đệm đầu câu
                topic = re.sub(r'^(?:cho\s+tôi\s+biết|hãy\s+cho\s+biết|xin\s+hỏi|theo\s+quy\s+định)\s*', '', query, flags=re.IGNORECASE).strip()[:80]
        
        intent = QueryIntent(
            type=model_type,
            documents=final_docs,
            topic=topic,
            keywords=parsed.get("keywords", []),
            sub_queries=sub_queries,
        )
        
        # Bổ sung article_hint từ regex nếu model không trả
        if model_type == "single_lookup":
            intent.article_hint = article_hint
        
        # Type-specific defaults
        if model_type == "comparison":
            intent.criteria = parsed.get("criteria", [])
            if not intent.sub_queries or len(intent.sub_queries) < 2:
                if len(final_docs) >= 2:
                    topic = intent.topic or query
                    intent.sub_queries = [f"{topic} {doc}" for doc in final_docs]
        
        elif model_type == "listing":
            intent.expand_search = True
        
        elif model_type == "eligibility_check":
            intent.subject = parsed.get("subject", "")
            intent.check_aspects = parsed.get("check_aspects", ["đối tượng áp dụng", "điều kiện"])
        
        elif model_type == "procedure":
            intent.procedure_keywords = parsed.get(
                "procedure_keywords",
                [kw for kw in intent.keywords if kw] or ["hồ sơ", "thủ tục", "thời hạn", "nộp"]
            )
        
        elif model_type == "cross_document":
            intent.search_scope = "global"
        
        elif model_type == "definition":
            intent.term = parsed.get("term", intent.topic)
        
        return intent
    
    def _build_intent_from_regex(
        self, query: str, intent_type: str,
        resolved_docs: List[str], article_hint: Optional[str]
    ) -> QueryIntent:
        """Xây dựng QueryIntent từ regex fallback (không có LLM)."""
        # Trích keywords đơn giản từ query
        q_lower = query.lower()
        keywords = [w for w in q_lower.split() if len(w) > 2][:5]
        
        intent = QueryIntent(
            type=intent_type,
            documents=resolved_docs,
            topic=query[:100],
            keywords=keywords,
            sub_queries=[query],  # Regex fallback: giữ nguyên câu gốc
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
        """Parse JSON an toàn từ output LLM. Xử lý cả JSON bị malformed."""
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
        
        # Thử parse trực tiếp trước
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Tìm JSON object (nested-aware)
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Fallback: Trích xuất từng field bằng regex từ text bị malformed
        # VD: {"type": "definition", "documents": []}, "sub_queries": [], "keywords": [...]}
        result = {}
        
        # type
        m = re.search(r'"type"\s*:\s*"([^"]+)"', text)
        if m:
            result["type"] = m.group(1)
        
        # topic
        m = re.search(r'"topic"\s*:\s*"([^"]*)"', text)
        if m:
            result["topic"] = m.group(1)
        
        # sub_queries
        m = re.search(r'"sub_queries"\s*:\s*\[([^\]]*)\]', text)
        if m:
            try:
                result["sub_queries"] = json.loads(f"[{m.group(1)}]")
            except (ValueError, json.JSONDecodeError):
                result["sub_queries"] = []
        
        # keywords
        m = re.search(r'"keywords"\s*:\s*\[([^\]]*)\]', text)
        if m:
            try:
                result["keywords"] = json.loads(f"[{m.group(1)}]")
            except (ValueError, json.JSONDecodeError):
                result["keywords"] = []
        
        # documents
        m = re.search(r'"documents"\s*:\s*\[([^\]]*)\]', text)
        if m:
            try:
                result["documents"] = json.loads(f"[{m.group(1)}]")
            except:
                result["documents"] = []
        
        # Chỉ trả kết quả nếu có ít nhất "type"
        if "type" in result:
            return result
        
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
