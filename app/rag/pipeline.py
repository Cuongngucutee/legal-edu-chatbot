"""
LawEdu AI — RAG Pipeline.
Main agentic RAG orchestrator: Intent → Route → Retrieve → Skill → Generate.
Uses 320B LLM API for all generation tasks.
"""
import re
import time
import logging
import unicodedata
import ast
import math
import json

from app.query.intent_classifier import classify_intent
from app.query.query_expander import EducationQueryExpander
from app.query.query_rewriter import QueryRewriteSkill, DecomposeSkill, EvidenceFocusSkill, HyDESkill
from app.rag.context_builder import build_context, postprocess_citations, fmt_source
from app.rag.handlers.handlers import (
    SummaryHandler, StatisticalHandler, ListingHandler, ComparisonHandler,
)
from app.llm.prompts import GENERATION_PROMPT, GENERATION_SYSTEM_PROMPT

# Cross-reference: implementing docs → parent laws
CROSS_REFERENCE_MAP = {
    "71/2020/NĐ-CP": ["43/2019/QH14"],
    "71/2020/ND-CP": ["43/2019/QH14"],
    "84/2020/NĐ-CP": ["43/2019/QH14"],
    "84/2020/ND-CP": ["43/2019/QH14"],
    "105/2020/NĐ-CP": ["43/2019/QH14"],
    "105/2020/ND-CP": ["43/2019/QH14"],
    "116/2020/NĐ-CP": ["43/2019/QH14"],
    "116/2020/ND-CP": ["43/2019/QH14"],
    "238/2025/NĐ-CP": ["43/2019/QH14"],
    "238/2025/ND-CP": ["43/2019/QH14"],
    "99/2019/NĐ-CP": ["34/2018/QH14"],
    "99/2019/ND-CP": ["34/2018/QH14"],
    "08/2023/TT-BGDĐT": ["01/2021/TT-BGDĐT", "02/2021/TT-BGDĐT"],
    "08/2012/QH13": ["34/2018/QH14"],
}

def _normalize_text(text):
    if not text: return ""
    text = str(text).lower().strip()
    text = text.replace("đ", "d")
    nfd_form = unicodedata.normalize('NFD', text)
    return "".join([c for c in nfd_form if not unicodedata.combining(c)])

def _bare_sh(text):
    text_norm = _normalize_text(text)
    m = re.search(r'(\d+[\/-]\d{4}[\/-]?[\w\-]*)', text_norm)
    return m.group(1) if m else text_norm

def _resolve_sh(raw_sh, registry):
    if raw_sh in registry: 
        return registry[raw_sh]
    
    norm_raw = _bare_sh(raw_sh)
    for k in registry:
        if _bare_sh(k) == norm_raw:
            return registry[k]
            
    for k in registry:
        norm_k = _bare_sh(k)
        if len(norm_k) > 3 and len(norm_raw) > 3:
            if norm_k in norm_raw or norm_raw in norm_k:
                return registry[k]
                
    return raw_sh

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
            "HYDE": HyDESkill(),
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

    def _rrf_merge(self, ranked_lists, k=60):
        scores, chunk_map = {}, {}
        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked):
                cid = doc.get("chunk_id", "")
                if not cid: continue
                scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
                if cid not in chunk_map: chunk_map[cid] = doc
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [chunk_map[cid] for cid in sorted_ids], scores

    def _inject_cross_refs(self, top_docs, all_docs_map):
        injected = []
        seen = set()
        for sh in top_docs:
            bare = _bare_sh(sh)
            if bare not in seen:
                injected.append(sh)
                seen.add(bare)
            for key, parents in CROSS_REFERENCE_MAP.items():
                if _bare_sh(key) == bare or key == sh:
                    for parent in parents:
                        resolved = _resolve_sh(parent, self.retriever.index.doc_registry)
                        parent_bare = _bare_sh(resolved)
                        if parent_bare not in seen:
                            injected.append(resolved)
                            seen.add(parent_bare)
        return injected

    def _parse_320b(self, response_text, uniq_docs):
        nids = set()
        arts = []
        
        cleaned_text = response_text.strip()
        match_tag = re.search(r'<selected_clauses>(.*?)</selected_clauses>', cleaned_text, re.DOTALL)
        if match_tag:
            target_block = match_tag.group(1).strip()
        else:
            target_block = cleaned_text

        if target_block.startswith("```"):
            lines_block = target_block.splitlines()
            if lines_block[0].startswith("```"):
                lines_block = lines_block[1:]
            if lines_block and lines_block[-1].startswith("```"):
                lines_block = lines_block[:-1]
            target_block = "\n".join(lines_block).strip()
            
        m = re.search(r'\[\s*\{.*\}\s*\]', target_block, re.DOTALL)
        if not m:
            m = re.search(r'\[.*\]', target_block, re.DOTALL)
            
        items = None
        if m:
            json_str = m.group(0)
            try:
                items = json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    items = ast.literal_eval(json_str)
                except Exception:
                    pass
                    
        if not isinstance(items, list):
            items = []
            matches = re.findall(r'\{\s*["\']so_hieu["\']\s*:\s*["\']([^"\']+)["\']\s*,\s*["\']dieu["\']\s*:\s*(\d+)\s*\}', target_block)
            for raw_sh, dieu_str in matches:
                try:
                    items.append({"so_hieu": raw_sh, "dieu": int(dieu_str)})
                except ValueError:
                    pass
            if not items and not match_tag:
                matches_global = re.findall(r'\{\s*["\']so_hieu["\']\s*:\s*["\']([^"\']+)["\']\s*,\s*["\']dieu["\']\s*:\s*(\d+)\s*\}', cleaned_text)
                for raw_sh, dieu_str in matches_global:
                    try:
                        items.append({"so_hieu": raw_sh, "dieu": int(dieu_str)})
                    except ValueError:
                        pass

        for item in items:
            if not isinstance(item, dict):
                continue
            raw = _bare_sh(item.get("so_hieu", ""))
            dieu = item.get("dieu")
            if not raw or dieu is None:
                continue
            try:
                dieu_num = int(dieu)
            except (ValueError, TypeError):
                continue
                
            raw_upper = raw.upper()
            if "43/2019" in raw_upper:
                raw = "43/2019/QH14"
            elif "34/2018" in raw_upper:
                raw = "34/2018/QH14"
            elif "84/2020" in raw_upper:
                raw = "84/2020/NĐ-CP"
            elif "116/2020" in raw_upper:
                raw = "116/2020/NĐ-CP"
            elif "105/2020" in raw_upper:
                raw = "105/2020/NĐ-CP"
            elif "71/2020" in raw_upper:
                raw = "71/2020/NĐ-CP"
            elif "86/2021" in raw_upper:
                raw = "86/2021/NĐ-CP"
            elif "22/2021" in raw_upper:
                raw = "22/2021/TT-BGDĐT"
            elif "24/2024" in raw_upper:
                raw = "24/2024/TT-BGDĐT"
            elif "32/2018" in raw_upper:
                raw = "32/2018/TT-BGDĐT"
            elif "05/2023" in raw_upper:
                raw = "05/2023/TT-BGDĐT"
            elif "08/2023" in raw_upper:
                raw = "08/2023/TT-BGDĐT"
            elif "01/2021" in raw_upper:
                raw = "01/2021/TT-BGDĐT"
            elif "02/2021" in raw_upper:
                raw = "02/2021/TT-BGDĐT"
            elif "03/2021" in raw_upper:
                raw = "03/2021/TT-BGDĐT"
            elif "04/2021" in raw_upper:
                if "ND" in raw_upper or "NĐ" in raw_upper or "NGHỊ ĐỊNH" in raw_upper or "NGHI DINH" in raw_upper:
                    raw = "04/2021/NĐ-CP"
                else:
                    raw = "04/2021/TT-BGDĐT"
                
            resolved_key = _resolve_sh(raw, self.retriever.index.doc_registry)
            for nid in self.retriever.index.get_doc_node_ids(resolved_key):
                if nid not in self.retriever.index.graph.nodes:
                    continue
                nd = self.retriever.index.graph.nodes[nid]
                name = nd.get("name", "")
                if not name:
                    ft = nd.get("full_text", "") or nd.get("search_text", "")
                    if ft:
                        name = ft.split("\n")[0].strip()
                dm = re.search(r'Điều\s+(\d+)', name)
                if dm and int(dm.group(1)) == dieu_num:
                    nids.add(nid)
                    
            disp_name = uniq_docs.get(raw, raw)
            if not any(a == f"Điều {dieu_num} ({disp_name})" for a in arts):
                arts.append(f"Điều {dieu_num} ({disp_name})")
                
        return nids, arts

    def _lookup_flow(self, query, start, intent):
        """Unified 3-stage Agentic RAG flow: Stage 1 (Hybrid RRF), Stage 2 (CoT TOC), Stage 3 (Generation + Reflection)."""
        logger.info("🎬 [Stage 1] Khởi chạy Hybrid Search + RRF + Cross-Reference...")
        
        # 1. Expand query + Retrieve
        main_docs = self.retriever.retrieve_as_docs(query, top_k=20)
        all_ranked = [main_docs]
        
        expanded = self.query_expander.expand(query)
        for eq in expanded[:3]:
            eq_docs = self.retriever.retrieve_as_docs(eq, top_k=8)
            if eq_docs:
                all_ranked.append(eq_docs)
                
        # HyDE Search
        hyde_prompt = (
            "Bạn là chuyên gia pháp luật giáo dục Việt Nam.\n"
            "Hãy viết một đoạn văn ngắn (2-4 câu) mô tả quy định pháp luật giả định chính xác nhất để trả lời cho câu hỏi sau.\n"
            "Hãy sử dụng văn phong văn bản luật chính xác, trang trọng và khách quan.\n"
            "Không cần mở đầu bằng lời chào hay giải thích, hãy viết thẳng nội dung quy định giả định.\n\n"
            f"Câu hỏi: {query}\n\n"
            "Quy định pháp luật giả định:"
        )
        try:
            hyde_doc = self.generator_llm.generate(hyde_prompt, temperature=0.3).strip()
            hyde_docs = self.retriever.retrieve_as_docs(hyde_doc, top_k=8)
            if hyde_docs:
                all_ranked.append(hyde_docs)
        except Exception as e:
            logger.warning(f"⚠️ HyDE generation error: {e}")
            
        docs, rrf_scores = self._rrf_merge(all_ranked)
        if not docs:
            return self._no_answer(query, start, intent)
            
        # Doc-level Vote Aggregation
        doc_agg = {}
        for d in docs:
            meta = d.get("metadata", {})
            ten = meta.get("ten_van_ban") or meta.get("source") or ""
            bare = _bare_sh(ten)
            cid = d.get("chunk_id", "")
            rrf = rrf_scores.get(cid, 0)
            if bare:
                if bare not in doc_agg:
                    doc_agg[bare] = {"ten": ten, "count": 0, "total_rrf": 0, "max_rrf": 0}
                doc_agg[bare]["count"] += 1
                doc_agg[bare]["total_rrf"] += rrf
                doc_agg[bare]["max_rrf"] = max(doc_agg[bare]["max_rrf"], rrf)

        unique_docs, ranked_docs = {}, []
        if doc_agg:
            for bare, info in doc_agg.items():
                avg = info["total_rrf"] / info["count"] if info["count"] else 0
                info["rank_score"] = info["max_rrf"] + 0.3 * math.log(1 + info["count"]) * avg
                unique_docs[bare] = info["ten"]
            ranked_docs = sorted(doc_agg.items(), key=lambda x: x[1]["rank_score"], reverse=True)

        top_so_hieu = [sh for sh, _ in ranked_docs[:5]]
        
        # Explicit target forcing
        explicit_targets = self.query_expander.get_target_docs(query)
        for ext_doc in explicit_targets:
            ext_bare = _bare_sh(ext_doc)
            if ext_bare in top_so_hieu:
                top_so_hieu.remove(ext_bare)
            top_so_hieu.insert(0, ext_bare)
            unique_docs[ext_bare] = ext_doc
            
        # Cross-reference injection
        top_so_hieu = self._inject_cross_refs(top_so_hieu, unique_docs)
        logger.info(f"   [Stage 1] Top docs: {top_so_hieu[:5]}")
        
        # 2. Stage 2: CoT TOC Selection
        logger.info("🧠 [Stage 2] Khởi chạy CoT TOC Selection...")
        toc_parts = []
        for sh in top_so_hieu[:4]:
            canonical_sh = _resolve_sh(sh, self.retriever.index.doc_registry)
            toc = self.retriever.index.build_toc(canonical_sh)
            if toc:
                toc_parts.append(toc)
                
        final_node_ids, stage2_articles = set(), []
        if toc_parts:
            toc_text = "\n\n".join(toc_parts)
            context_text = "\n".join([
                f"[{d.get('metadata',{}).get('ten_van_ban','')}] Điều {d.get('metadata',{}).get('so_dieu','')}: {(d.get('content','') or d.get('text',''))[:150]}..."
                for d in docs[:7]
            ])
            
            prompt_320b = f"""Bạn là một chuyên gia cao cấp về pháp luật giáo dục Việt Nam. Hãy thực hiện phân tích chuỗi lập luận (Chain-of-Thought) CỰC KỲ NGẮN GỌN (tối đa 2-3 câu) trước khi lựa chọn các Điều khoản cần thiết để trả lời câu hỏi dưới đây.

Câu hỏi: \"{query}\"

Các trích đoạn liên quan (tham khảo):
{context_text}

Mục lục các văn bản pháp luật:
{toc_text}

HƯỚNG DẪN ĐỐI CHIẾU LUẬT HỌC DÀNH CHO CHUYÊN GIA:
1. TIÊU CHUẨN CHỨC DANH NGHỀ nghiệp GIÁO VIÊN (Thông tư 01, 02, 03, 04/2021 và sửa đổi 08/2023):
   - Giáo viên Mầm non: Chọn ĐỒNG THỜI Điều của TT 01/2021/TT-BGDĐT và Điều 1 của TT 08/2023/TT-BGDĐT (Chứa quy định sửa đổi bổ sung).
   - Giáo viên Tiểu học: Chọn Điều của TT 02/2021/TT-BGDĐT và Điều 2 của TT 08/2023/TT-BGDĐT.
   - Giáo viên THCS: Chọn Điều của TT 03/2021/TT-BGDĐT và Điều 3 của TT 08/2023/TT-BGDĐT.
   - Giáo viên THPT: Chọn Điều của TT 04/2021/TT-BGDĐT và Điều 4 của TT 08/2023/TT-BGDĐT.
2. HỖ TRỢ SINH VIÊN SƯ PHẠM (Nghị định 116/2020/NĐ-CP):
   - Mức hỗ trợ/Đối tượng: Chọn Điều 4.
   - Cơ chế đặt hàng, giao nhiệm vụ: Chọn Điều 3, Điều 5.
   - Bảo lưu học tập, nghỉ học tạm thời, ngừng học, bồi hoàn kinh phí: Phải chọn Điều 6.
   - Trách nhiệm bồi hoàn, thu hồi: Chọn Điều 8, Điều 9.
3. PHÁT TRIỂN & CHUYỂN ĐỔI TRƯỜNG ĐẠI HỌC (Luật 34/2018/QH14 và Nghị định 99/2019/NĐ-CP):
   - Luôn chọn ĐỒNG THỜI Điều 1 của Luật 34/2018/QH14 và các Điều tương ứng trong Nghị định 99/2019/NĐ-CP (Điều 3 hoặc Điều 4).
4. XÃ HỘI HÓA & MẦM NON KHU CÔNG NGHIỆP:
   - Chọn ĐỒNG THỜI Luật Giáo dục 43/2019/QH14 (Điều 17 hoặc Điều 26 hoặc Điều 102) và Nghị định 105/2020/NĐ-CP (Điều 5).
5. HỌC BỔNG & HỌC PHÍ (Nghị định 84/2020/NĐ-CP và Nghị định 81/2021/NĐ-CP):
   - Học bổng chính sách, học bổng cử tuyển: Chọn Nghị định 84/2020/NĐ-CP (Điều 8, Điều 9) hoặc Nghị định 81/2021/NĐ-CP.
   - Nếu có từ "cử tuyển" và "học bổng chính sách": Luôn chọn Điều 9 của Nghị định 84/2020/NĐ-CP.
6. THI TỐT NGHIỆP THPT (Thông tư 24/2024/TT-BGDĐT):
   - Lộ trình áp dụng quy chế thi mới, thí sinh tự do: Phải chọn Điều 2 hoặc Điều 3 của Thông tư 24/2024/TT-BGDĐT.
7. LỘ TRÌNH TRIỂN KHAI CTGDPT MỚI (Thông tư 32/2018/TT-BGDĐT):
   - Luôn chọn ĐỒNG THỜI cả Điều 2 và Điều 3 của Thông tư 32/2018/TT-BGDĐT.

Nhiệm vụ của bạn:
1. Lập luận CỰC KỲ TÓM TẮT (1-2 câu) dựa trên hướng dẫn đối chiếu luật học ở trên để giải thích lựa chọn của bạn.
2. BẮT BUỘC đặt mảng JSON kết quả trong cặp thẻ <selected_clauses>...</selected_clauses> ở cuối câu trả lời.

Ví dụ định dạng đầu ra bắt buộc ở cuối câu trả lời:
Lập luận: Theo hướng dẫn đối chiếu luật học, giáo viên mầm non hạng II thăng hạng lên hạng I cần áp dụng cả Điều 5 của Thông tư 01/2021 và Điều 1 của Thông tư 08/2023 sửa đổi.
<selected_clauses>
[
  {{"so_hieu": "01/2021/TT-BGDĐT", "dieu": 5}},
  {{"so_hieu": "08/2023/TT-BGDĐT", "dieu": 1}}
]
</selected_clauses>"""

            try:
                res_320b = self.generator_llm.generate(prompt_320b, temperature=0.1)
                final_node_ids, stage2_articles = self._parse_320b(res_320b, unique_docs)
            except Exception as e:
                logger.warning(f"⚠️ 320B Stage 2 Selection error: {e}")
                
            # Retry Stage 2
            if len(stage2_articles) < 2:
                retry_prompt = f"""Bạn là một chuyên gia pháp luật giáo dục Việt Nam. Hãy thực hiện lập luận (Chain-of-Thought) CỰC KỲ TÓM TẮT (tối đa 2 câu) để liệt kê TẤT CẢ các Điều khoản có thể liên quan đến câu hỏi dưới đây, kể cả liên quan gián tiếp dựa trên các nguyên tắc luật học.

Câu hỏi: \"{query}\"

Mục lục các văn bản:
{toc_text}

Nhiệm vụ của bạn:
1. Lập luận siêu ngắn gọn về việc liên kết và thay thế của các thông tư/luật cũ và mới.
2. Chọn ÍT NHẤT 2 điều khoản.
3. Đặt mảng JSON kết quả trong cặp thẻ <selected_clauses>...</selected_clauses> ở cuối câu trả lời.

Ví dụ định dạng đầu ra:
<selected_clauses>
[
  {{"so_hieu": "01/2021/TT-BGDĐT", "dieu": 4}},
  {{"so_hieu": "08/2023/TT-BGDĐT", "dieu": 1}}
]
</selected_clauses>"""
                try:
                    res_retry = self.generator_llm.generate(retry_prompt, temperature=0.2)
                    retry_nodes, retry_arts = self._parse_320b(res_retry, unique_docs)
                    if len(retry_arts) > len(stage2_articles):
                        final_node_ids = retry_nodes
                        stage2_articles = retry_arts
                except Exception as e:
                    logger.warning(f"⚠️ 320B Stage 2 Selection retry error: {e}")

        logger.info(f"   [Stage 2] Selected Articles: {stage2_articles}")
        
        # 3. Stage 3: Generation
        logger.info("✍️ [Stage 3] Khởi chạy Generation...")
        final_nodes = list(final_node_ids)
        final_docs = []
        for nid in final_nodes:
            d = self.retriever._node_to_doc(nid)
            if d:
                final_docs.append(d)
                
        if not final_docs:
            logger.info("   ⚠️ Không có Điều khoản nào được chọn từ Stage 2, dùng top 5 văn bản Stage 1 làm fallback.")
            final_docs = docs[:5]
        else:
            final_docs = final_docs[:5]
            
        context = build_context(final_docs, max_chars=8000)
        
        try:
            answer = self.generator_llm.generate(
                GENERATION_PROMPT.format(query=query, context=context),
                system_prompt=GENERATION_SYSTEM_PROMPT,
                temperature=0.1,
            ).strip()
        except Exception as e:
            logger.error(f"⚠️ Final generation error: {e}")
            answer = ""
            
        # Citation Reflection
        if stage2_articles and answer:
            # Check if any selected article numbers are missing in the generated response
            missing_any = False
            cited_dieu = set()
            for match in re.finditer(r'[Đđ]iều\s+([0-9\s,vàhoặc]+)', answer):
                nums = re.findall(r'\d+', match.group(0))
                for num in nums:
                    cited_dieu.add(int(num))
                    
            for item_str in stage2_articles:
                # e.g., "Điều 5 (01/2021/TT-BGDĐT)"
                m_num = re.search(r'Điều\s+(\d+)', item_str)
                if m_num:
                    num_val = int(m_num.group(1))
                    if num_val not in cited_dieu:
                        missing_any = True
                        break
                        
            if missing_any:
                logger.info("🔄 [Stage 3 Reflection] Phát hiện câu trả lời thiếu trích dẫn Điều khoản bắt buộc, tiến hành Reflection hiệu chỉnh...")
                reflection_prompt = f"""Bạn là một chuyên gia kiểm duyệt pháp lý tối cao. 
Câu hỏi của người dùng: \"{query}\"
Câu trả lời hiện tại:
\"{answer}\"

Yêu cầu bắt buộc: Hiệu chỉnh lại câu trả lời trên để tích hợp trực tiếp và tự nhiên các trích dẫn Điều khoản pháp lý cụ thể sau đây: {stage2_articles}.
Vui lòng viết lại câu trả lời, đảm bảo giữ nguyên tính chính xác, ngắn gọn, và chèn các số Điều đã chọn một cách chính xác nhất."""
                try:
                    answer = self.generator_llm.generate(
                        reflection_prompt,
                        system_prompt="Bạn là chuyên gia hiệu chỉnh pháp lý chính xác và chuyên nghiệp.",
                        temperature=0.1,
                    ).strip()
                except Exception as e:
                    logger.warning(f"⚠️ Reflection generation error: {e}")

        answer = postprocess_citations(answer, final_docs)
        
        return {
            "answer": answer,
            "sources": [fmt_source(d) for d in final_docs],
            "skill_log": [],
            "iterations": 1,
            "intent": intent,
            "final_query": query,
            "latency_ms": (time.time() - start) * 1000,
            "stage1_docs": top_so_hieu,
            "stage2_articles": list(stage2_articles),
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
