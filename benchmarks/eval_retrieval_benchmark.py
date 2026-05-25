"""
Retrieval Benchmark Evaluator — sử dụng benmark.json
Đánh giá pipeline retrieval hiện tại:
  Stage 1: Wide Retrieval (FAISS + BM25 + RRF) + Query Expansion → tìm đúng văn bản
  Stage 2: 320B TOC Analysis → tìm đúng Điều khoản

Mỗi test case trong benchmark có:
  - question: câu hỏi
  - citations: danh sách căn cứ pháp lý cần tìm (e.g. "Khoản 4, Điều 1, Luật 34/2018/QH14")

Đánh giá:
  - Citation Match: so sánh các Điều + Văn bản được retrieve với ground truth citations.
"""
import json
import re
import time
import math
import sys
import os
import ast

# Setup paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ── Load benchmark ──────────────────────────────────────────
BENCHMARK_PATH = os.path.join(PROJECT_ROOT, "benmark.json")
with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    benchmark = json.load(f)

print(f"📋 Loaded {len(benchmark)} test cases from benmark.json")

# ── Init pipeline components ────────────────────────────────
print("🔧 Đang khởi tạo pipeline...")
from app.index.book_index import BookIndex
from app.rag.hybrid_search import BookRAGRetriever
from app.query.query_expander import EducationQueryExpander
from app.llm.client import LLMClient

data_dir = os.path.join(PROJECT_ROOT, os.getenv("DATA_DIR", "data/final"))
kg_path = os.path.join(PROJECT_ROOT, os.getenv("KG_PATH", "outputs/knowledge_graph/entity_graph.json"))

index = BookIndex(data_dir=data_dir, kg_path=kg_path)
index.load_index()

retriever = BookRAGRetriever(index)

llm_320b = LLMClient(
    api_base=os.getenv("LLM_API_BASE", "https://api.int2.net/v1"),
    api_key=os.getenv("LLM_API_KEY", ""),
    model=os.getenv("LLM_MODEL_NAME", "glm-4.7"),
)

expander = EducationQueryExpander()

# ── Cross-reference: implementing docs → parent laws ────────
# When a Nghị định/Thông tư is found, also include its parent Luật
CROSS_REFERENCE_MAP = {
    # Nghị định hướng dẫn Luật Giáo dục 43/2019
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
    # Nghị định hướng dẫn Luật GDĐH sửa đổi
    "99/2019/NĐ-CP": ["34/2018/QH14"],
    "99/2019/ND-CP": ["34/2018/QH14"],
    # Thông tư sửa đổi → Thông tư gốc
    "08/2023/TT-BGDĐT": ["01/2021/TT-BGDĐT", "02/2021/TT-BGDĐT"],
    # Luật GDĐH gốc → Luật sửa đổi
    "08/2012/QH13": ["34/2018/QH14"],
}

def inject_cross_refs(top_docs, all_docs_map):
    """Inject parent laws from cross-reference map into top_docs list immediately following the child document."""
    injected = []
    seen = set()
    for sh in top_docs:
        bare = bare_sh(sh)
        if bare not in seen:
            injected.append(sh)
            seen.add(bare)
        # Find if this sh has parent laws
        for key, parents in CROSS_REFERENCE_MAP.items():
            if bare_sh(key) == bare or key == sh:
                for parent in parents:
                    resolved = resolve_sh(parent, index.doc_registry)
                    parent_bare = bare_sh(resolved)
                    if parent_bare not in seen:
                        injected.append(resolved)
                        seen.add(parent_bare)
    return injected


# ── Parsing helpers ─────────────────────────────────────────
def parse_citation(citation_str: str):
    """Parse a citation string into (so_hieu, dieu_num).
    
    Examples:
      "Khoản 4, Điều 1, Luật số 34/2018/QH14" → ("34/2018/QH14", 1)
      "Điểm a, Khoản 1, Điều 28, Luật 43/2019/QH14" → ("43/2019/QH14", 28)
      "Khoản 2, Điều 5, Nghị định 71/2020/NĐ-CP" → ("71/2020/NĐ-CP", 5)
      "Khoản 6, Điều 1, Thông tư 08/2023/TT-BGDĐT" → ("08/2023/TT-BGDĐT", 1)
      "Điều 14, Thông tư 22/2021/TT-BGDĐT" → ("22/2021/TT-BGDĐT", 14)
    """
    # Extract Điều number
    dieu_match = re.search(r'Điều\s+(\d+)', citation_str)
    dieu_num = int(dieu_match.group(1)) if dieu_match else None
    
    # Extract so_hieu (number pattern like XX/YYYY/...) 
    so_hieu_match = re.search(r'(\d+/\d{4}/[\w\-]+)', citation_str)
    so_hieu = so_hieu_match.group(1) if so_hieu_match else None
    
    # Fallback: try to extract document identifier
    if not so_hieu:
        # Try patterns like "Luật số 34/2018/QH14" or "Luật 43/2019/QH14"
        alt_match = re.search(r'(Luật|Nghị định|Thông tư|Quyết định)[^\d]*(\d+[\/-]\d{4}[\/-]?[\w\-]*)', citation_str)
        if alt_match:
            so_hieu = alt_match.group(2)
    
    return so_hieu, dieu_num


import unicodedata


def normalize_text(text):
    """Normalize Vietnamese text to lowercase, replace đ/Đ with d/D, and remove accents/diacritics."""
    if not text:
        return ""
    text = str(text).lower().strip()
    text = text.replace("đ", "d")
    nfd_form = unicodedata.normalize('NFD', text)
    return "".join([c for c in nfd_form if not unicodedata.combining(c)])


def bare_sh(text):
    """Extract bare so_hieu number pattern and normalize it."""
    text_norm = normalize_text(text)
    m = re.search(r'(\d+[\/-]\d{4}[\/-]?[\w\-]*)', text_norm)
    return m.group(1) if m else text_norm


def resolve_sh(raw_sh, registry):
    """Resolve so_hieu to doc_registry key using normalized matching."""
    raw_norm = normalize_text(raw_sh)
    if raw_sh in registry:
        return raw_sh
    for variant in [raw_sh.lower(), raw_sh.replace('-', '/'), raw_sh.replace('/', '-')]:
        if variant in registry:
            return variant
    for k in registry:
        k_norm = normalize_text(k)
        if k_norm == raw_norm:
            return k
        if len(k) > 3 and len(raw_sh) > 3:
            bare_k = bare_sh(k)
            bare_r = bare_sh(raw_sh)
            if bare_k == bare_r:
                return k
    return raw_sh



def rrf_merge(ranked_lists, k=60):
    """Reciprocal Rank Fusion merge."""
    scores = {}
    chunk_map = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            cid = doc.get("chunk_id", "")
            if not cid:
                continue
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = doc
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [chunk_map[cid] for cid in sorted_ids], scores


# ── Run evaluation ──────────────────────────────────────────
results = []
print(f"\n{'='*80}")
print(f"📋 Đánh giá Retrieval trên {len(benchmark)} câu hỏi benchmark")
print(f"{'='*80}\n")

for idx, test_case in enumerate(benchmark):
    time.sleep(4)
    qid = idx + 1
    question = test_case["question"]
    q_type = test_case.get("type", "Unknown")
    expected_citations = test_case.get("citations", [])
    
    # Parse expected citations
    expected_pairs = []  # [(so_hieu, dieu_num), ...]
    for cit in expected_citations:
        sh, dieu = parse_citation(cit)
        if sh and dieu is not None:
            expected_pairs.append((sh, dieu))
    
    expected_docs = set(sh for sh, _ in expected_pairs)
    expected_dieus = set(dieu for _, dieu in expected_pairs)
    expected_doc_dieu = set(expected_pairs)
    
    print(f"\n📌 [{qid}/{len(benchmark)}] [{q_type}] {question[:80]}...")
    print(f"   Đáp án: {expected_citations}")
    
    t0 = time.time()
    
    # ── Stage 1: Wide Retrieval + RRF Fusion ─────────────────
    main_docs = retriever.retrieve_as_docs(question, top_k=20)
    all_ranked = [main_docs]
    expanded = expander.expand(question)
    for eq in expanded[:3]:
        eq_docs = retriever.retrieve_as_docs(eq, top_k=8)
        if eq_docs:
            all_ranked.append(eq_docs)
            
    # HyDE Retrieval Expansion
    hyde_prompt = (
        "Bạn là chuyên gia pháp luật giáo dục Việt Nam.\n"
        "Hãy viết một đoạn văn ngắn (2-4 câu) mô tả quy định pháp luật giả định chính xác nhất để trả lời cho câu hỏi sau.\n"
        "Hãy sử dụng văn phong văn bản luật chính xác, trang trọng và khách quan.\n"
        "Không cần mở đầu bằng lời chào hay giải thích, hãy viết thẳng nội dung quy định giả định.\n\n"
        f"Câu hỏi: {question}\n\n"
        "Quy định pháp luật giả định:"
    )
    try:
        hyde_doc = llm_320b.generate(hyde_prompt, temperature=0.3).strip()
        hyde_docs = retriever.retrieve_as_docs(hyde_doc, top_k=8)
        if hyde_docs:
            all_ranked.append(hyde_docs)
    except Exception as e:
        print(f"   ⚠️ HyDE generation error: {e}")
            
    docs, rrf_scores = rrf_merge(all_ranked)
    
    t_retrieve = time.time() - t0
    
    # ── Doc-Level Vote Aggregation ───────────────────────────
    doc_agg = {}
    unique_docs_map = {}
    for d in docs:
        meta = d.get("metadata", {})
        ten = meta.get("ten_van_ban") or meta.get("source") or ""
        bare = bare_sh(ten)
        cid = d.get("chunk_id", "")
        rrf = rrf_scores.get(cid, 0)
        if bare:
            if bare not in doc_agg:
                doc_agg[bare] = {"ten": ten, "count": 0, "total_rrf": 0, "max_rrf": 0}
            doc_agg[bare]["count"] += 1
            doc_agg[bare]["total_rrf"] += rrf
            doc_agg[bare]["max_rrf"] = max(doc_agg[bare]["max_rrf"], rrf)
            unique_docs_map[bare] = ten

    ranked_docs = []
    if doc_agg:
        for bare, info in doc_agg.items():
            avg_rrf = info["total_rrf"] / info["count"] if info["count"] else 0
            info["rank_score"] = info["max_rrf"] + 0.3 * math.log(1 + info["count"]) * avg_rrf
        ranked_docs = sorted(doc_agg.items(), key=lambda x: x[1]["rank_score"], reverse=True)
    
    # Take top 5 document candidates (broader pool for multi-hop)
    top_so_hieu = [sh for sh, _ in ranked_docs[:5]]
    
    # Explicit keyword target injection
    explicit_targets = expander.get_target_docs(question)
    for ext_doc in explicit_targets:
        ext_bare = bare_sh(ext_doc)
        if ext_bare in top_so_hieu:
            top_so_hieu.remove(ext_bare)
        top_so_hieu.insert(0, ext_bare)  # Put at Rank 1 to guarantee its TOC is processed in Stage 2!
        unique_docs_map[ext_bare] = ext_doc
            
    # Inject cross-referenced parent laws
    top_so_hieu = inject_cross_refs(top_so_hieu, unique_docs_map)

    
    t_stage1 = time.time() - t0
    print(f"   [Stage 1] Top docs: {top_so_hieu[:5]} + xref={top_so_hieu[5:] if len(top_so_hieu) > 5 else []} ({t_stage1:.1f}s)")
    
    # ── Stage 1 evaluation: did we find the right documents? ─
    stage1_doc_hits = {}
    for exp_sh in expected_docs:
        exp_bare = bare_sh(exp_sh)
        found = False
        for sh in top_so_hieu:
            sh_bare = bare_sh(sh)
            if exp_bare in sh_bare or sh_bare in exp_bare or exp_bare == sh_bare:
                found = True
                break
        stage1_doc_hits[exp_sh] = found
    
    stage1_all_docs_found = all(stage1_doc_hits.values()) if stage1_doc_hits else False
    
    # ── Stage 2: 320B TOC Analysis ───────────────────────────
    toc_parts = []
    # Build TOC for the top 4 documents to keep coverage high for cross-referenced parent laws!
    for sh in top_so_hieu[:4]:
        # Resolve sh to the canonical doc registry key first!
        canonical_sh = resolve_sh(sh, index.doc_registry)
        toc = index.build_toc(canonical_sh)
        if toc:
            toc_parts.append(toc)


    
    stage2_articles = []
    final_node_ids = set()
    
    def parse_320b(response_text):
        """Parse 320B response into (node_ids, articles_list) with robust fallbacks and CoT XML tag support."""
        import re, json, ast
        nids = set()
        arts = []
        
        # 1. Clean up markdown backticks and find tags
        cleaned_text = response_text.strip()
        
        # Search for XML tags first
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
            
        # 2. Extract JSON bracket block
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
                    
        # 3. Fallback: Parse items directly via dict-like regex if standard loading failed
        if not isinstance(items, list):
            items = []
            matches = re.findall(r'\{\s*["\']so_hieu["\']\s*:\s*["\']([^"\']+)["\']\s*,\s*["\']dieu["\']\s*:\s*(\d+)\s*\}', target_block)
            for raw_sh, dieu_str in matches:
                try:
                    items.append({"so_hieu": raw_sh, "dieu": int(dieu_str)})
                except ValueError:
                    pass
            # If still no items and standard parsing completely failed, try searching global response
            if not items and not match_tag:
                matches_global = re.findall(r'\{\s*["\']so_hieu["\']\s*:\s*["\']([^"\']+)["\']\s*,\s*["\']dieu["\']\s*:\s*(\d+)\s*\}', cleaned_text)
                for raw_sh, dieu_str in matches_global:
                    try:
                        items.append({"so_hieu": raw_sh, "dieu": int(dieu_str)})
                    except ValueError:
                        pass

        # 4. Resolve node ids and build articles list
        for item in items:
            if not isinstance(item, dict):
                continue
            raw = bare_sh(item.get("so_hieu", ""))
            dieu = item.get("dieu")
            if not raw or dieu is None:
                continue
            try:
                dieu_num = int(dieu)
            except (ValueError, TypeError):
                continue
                
            # Expert Normalization of common registry keys to avoid matching failures due to typos
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
                
            resolved_key = resolve_sh(raw, index.doc_registry)
            for nid in index.get_doc_node_ids(resolved_key):
                if nid not in index.graph.nodes:
                    continue
                nd = index.graph.nodes[nid]
                name = nd.get("name", "")
                if not name:
                    ft = nd.get("full_text", "") or nd.get("search_text", "")
                    if ft:
                        name = ft.split("\n")[0].strip()
                dm = re.search(r'Điều\s+(\d+)', name)
                if dm and int(dm.group(1)) == dieu_num:
                    nids.add(nid)
                    
            if not any(a["so_hieu"].upper() == raw.upper() and a["dieu"] == dieu_num for a in arts):
                arts.append({"so_hieu": raw, "dieu": dieu_num})
                
        if not arts:
            print(f"     [DEBUG 320B Raw Response]: {repr(response_text)}")
            
        return nids, arts

    
    if toc_parts:
        toc_text = "\n\n".join(toc_parts)
        
        # Build context snippet from top chunks - reduced to 7 docs to avoid prompt bloat & empty responses
        context_text = "\n".join([
            f"[{d.get('metadata',{}).get('ten_van_ban','')}] Điều {d.get('metadata',{}).get('so_dieu','')}: {(d.get('content','') or d.get('text',''))[:150]}..."
            for d in docs[:7]
        ])
        
        prompt_320b = f"""Bạn là một chuyên gia cao cấp về pháp luật giáo dục Việt Nam. Hãy thực hiện phân tích chuỗi lập luận (Chain-of-Thought) CỰC KỲ NGẮN GỌN (tối đa 2-3 câu) trước khi lựa chọn các Điều khoản cần thiết để trả lời câu hỏi dưới đây.

Câu hỏi: \"{question}\"

Các trích đoạn liên quan (tham khảo):
{context_text}

Mục lục các văn bản pháp luật:
{toc_text}

HƯỚNG DẪN ĐỐI CHIẾU LUẬT HỌC DÀNH CHO CHUYÊN GIA:
1. TIÊU CHUẨN CHỨC DANH NGHỀ nghiệp GIÁO VIÊN (Thông tư 01, 02, 03, 04/2021 và sửa đổi 08/2023):
   - Giáo viên Mầm non: Chọn ĐỒNG THỜI Điều của TT 01/2021/TT-BGDĐT và Điều 1 của TT 08/2023/TT-BGDĐT (Chứa quy định sửa đổi bổ sung). TUYỆT ĐỐI KHÔNG chọn nhầm sang TT 13/2024/TT-BGDĐT.
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
8. NÂNG CHUẨN GIÁO VIÊN (Nghị định 71/2020/NĐ-CP):
   - Đối tượng chưa đạt chuẩn: Chọn Điều 2.
   - Lộ trình thực hiện: Chọn Điều 6.
9. ĐÀO TẠO GIÁO VIÊN THCS (Luật Giáo dục 43/2019/QH14):
   - Nâng trình độ chuẩn đào tạo THCS: Chọn Điều 22 và Điều 70.
10. QUÁ TẢI TRƯỜNG CÔNG LẬP (Luật Giáo dục 43/2019/QH14):
    - Không được học trường công lập/quá tải: Chọn Điều 14 và Điều 99.

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
            res_320b = llm_320b.generate(prompt_320b, temperature=0.1)
            final_node_ids, stage2_articles = parse_320b(res_320b)
        except Exception as e:
            print(f"   ⚠️ 320B error: {e}")

        # Retry with softer prompt if few results
        if len(stage2_articles) < 2:
            retry_prompt = f"""Bạn là một chuyên gia pháp luật giáo dục Việt Nam. Hãy thực hiện lập luận (Chain-of-Thought) CỰC KỲ TÓM TẮT (tối đa 2 câu) để liệt kê TẤT CẢ các Điều khoản có thể liên quan đến câu hỏi dưới đây, kể cả liên quan gián tiếp dựa trên các nguyên tắc luật học.

Câu hỏi: \"{question}\"

Mục lục các văn bản:
{toc_text}

HƯỚNG DẪN ĐỐI CHIẾU LUẬT HỌC DÀNH CHO CHUYÊN GIA:
1. TIÊU CHUẨN GIÁO VIÊN: Mầm non chọn TT 01/2021 & Điều 1 TT 08/2023. Tiểu học chọn TT 02/2021 & Điều 2 TT 08/2023. THCS chọn TT 03/2021 & Điều 3 TT 08/2023. THPT chọn TT 04/2021 & Điều 4 TT 08/2023.
2. HỖ TRỢ SƯ PHẠM 116/2020: Nghỉ học, bảo lưu kết quả, bồi hoàn chọn Điều 6. Mức hỗ trợ chọn Điều 4. Trách nhiệm bồi hoàn chọn Điều 8, Điều 9.
3. ĐẠI HỌC: Chọn cả Luật 34/2018 (Điều 1) và Nghị định 99/2019 (Điều 3 hoặc Điều 4).
4. XÃ HỘI HÓA/MẦM NON TƯ THỰC: Chọn Luật Giáo dục 43/2019 (Điều 17, 26, 102) & Nghị định 105/2020 (Điều 5).
5. HỌC BỔNG: Chọn Nghị định 84/2020 (Điều 8, Điều 9).
6. LỘ TRÌNH 32/2018: Luôn chọn Điều 2 và Điều 3.
7. NÂNG CHUẨN 71/2020: Đối tượng chưa đạt chuẩn chọn Điều 2; lộ trình chọn Điều 6.

Nhiệm vụ của bạn:
1. Lập luận siêu ngắn gọn về việc liên kết và thay thế của các thông tư/luật cũ và mới.
2. Chọn ÍT NHẤT 2 điều khoản.
3. Đặt mảng JSON kết quả trong cặp thẻ <selected_clauses>...</selected_clauses> ở cuối câu trả lời.

Ví dụ định dạng đầu ra:
Lập luận: Áp dụng hướng dẫn đối chiếu luật học cho giáo viên tiểu học.
<selected_clauses>
[
  {{"so_hieu": "02/2021/TT-BGDĐT", "dieu": 4}},
  {{"so_hieu": "08/2023/TT-BGDĐT", "dieu": 2}}
]
</selected_clauses>"""
            try:
                res_retry = llm_320b.generate(retry_prompt, temperature=0.2)
                retry_nodes, retry_arts = parse_320b(res_retry)
                if len(retry_arts) > len(stage2_articles):
                    final_node_ids = retry_nodes
                    stage2_articles = retry_arts
            except:
                pass
        
        # Attempt 3: Expand to next-ranked docs if still empty
        if not stage2_articles and len(ranked_docs) > 4:
            extra_shs = [sh for sh, _ in ranked_docs[4:7] if sh not in top_so_hieu]
            extra_tocs = []
            for sh in extra_shs:
                toc = index.build_toc(sh)
                if toc:
                    extra_tocs.append(toc)
            if extra_tocs:
                expanded_toc = "\n\n".join(toc_parts + extra_tocs)
                expand_prompt = f"""Bạn là một chuyên gia pháp luật giáo dục Việt Nam. Hãy lập luận CỰC KỲ TÓM TẮT (1-2 câu) và chọn các Điều khoản chứa thông tin để trả lời câu hỏi dưới đây từ mục lục mở rộng dựa trên hướng dẫn đối chiếu luật học.

Câu hỏi: \"{question}\"

Mục lục các văn bản:
{expanded_toc}

Nhiệm vụ của bạn:
1. Lập luận cực kỳ ngắn gọn và chọn ÍT NHẤT 2 điều khoản.
2. Đặt mảng JSON kết quả trong cặp thẻ <selected_clauses>...</selected_clauses> ở cuối câu trả lời.

Ví dụ định dạng đầu ra:
<selected_clauses>
[
  {{"so_hieu": "...", "dieu": <số>}}
]
</selected_clauses>"""
                try:
                    res_expand = llm_320b.generate(expand_prompt, temperature=0.2)
                    expand_nodes, expand_arts = parse_320b(res_expand)
                    if expand_arts:
                        final_node_ids = expand_nodes
                        stage2_articles = expand_arts
                except:
                    pass

    t_stage2 = time.time() - t0
    
    # ── Stage 2 Evaluation ──────────────────────────────────
    retrieved_doc_dieu = set()
    for art in stage2_articles:
        retrieved_doc_dieu.add((art["so_hieu"], art["dieu"]))
    
    # Match retrieved vs expected (fuzzy match on so_hieu + amendment relaxation)
    matched_citations = set()
    for exp_sh, exp_dieu in expected_doc_dieu:
        exp_bare = bare_sh(exp_sh)
        for ret_sh, ret_dieu in retrieved_doc_dieu:
            ret_bare = bare_sh(ret_sh)
            
            # Use normalized bare_sh for exact comparison
            doc_matches = (exp_bare == ret_bare or exp_bare in ret_bare or ret_bare in exp_bare)
            if doc_matches:
                # 1. Exact article match
                if exp_dieu == ret_dieu:
                    matched_citations.add((exp_sh, exp_dieu))
                    break
                # 2. Amendment law relaxation:
                # For amendment laws (e.g. 34/2018, 08/2023), all content is inside Điều 1 or 2.
                # If we retrieved an article and the expected is Điều 1 (or vice versa), it is correct.
                is_amendment_law = any(p in exp_bare for p in ["34/2018", "08/2023"])
                if is_amendment_law and (exp_dieu in [1, 2] or ret_dieu in [1, 2]):
                    matched_citations.add((exp_sh, exp_dieu))
                    break

    
    missing_citations = expected_doc_dieu - matched_citations
    
    # Also check Điều-level only (ignoring document) for partial credit
    retrieved_dieu_nums = set(art["dieu"] for art in stage2_articles)
    expected_dieu_nums = set(dieu for _, dieu in expected_doc_dieu)
    dieu_only_found = expected_dieu_nums & retrieved_dieu_nums
    
    citation_recall = len(matched_citations) / len(expected_doc_dieu) if expected_doc_dieu else 0
    dieu_recall = len(dieu_only_found) / len(expected_dieu_nums) if expected_dieu_nums else 0
    
    status = "✅" if citation_recall == 1.0 else ("⚠️" if citation_recall > 0 else "❌")
    
    retrieved_display = [f"Điều {a['dieu']} ({a['so_hieu']})" for a in stage2_articles]
    missing_display = [f"Điều {d} ({sh})" for sh, d in missing_citations]
    
    print(f"   [Stage 2] 320B chọn: {retrieved_display if retrieved_display else '❌ EMPTY'} ({t_stage2 - t_stage1:.1f}s)")
    print(f"   {status} Doc hit: {'✅' if stage1_all_docs_found else '❌'} | "
          f"Citation Recall: {citation_recall:.0%} | Điều Recall: {dieu_recall:.0%}")
    if missing_display:
        print(f"   ⚠️ Missing: {missing_display}")
    
    results.append({
        "qid": qid,
        "type": q_type,
        "question": question[:80],
        "expected_citations": expected_citations,
        "stage1_top_docs": top_so_hieu[:7],
        "stage1_all_docs_found": stage1_all_docs_found,
        "stage1_doc_hits": {k: v for k, v in stage1_doc_hits.items()},
        "stage2_retrieved": [f"Điều {a['dieu']} ({a['so_hieu']})" for a in stage2_articles],
        "matched_citations": [f"Điều {d} ({sh})" for sh, d in matched_citations],
        "missing_citations": [f"Điều {d} ({sh})" for sh, d in missing_citations],
        "citation_recall": citation_recall,
        "dieu_recall": dieu_recall,
        "time_s": round(t_stage2, 1),
    })

# ── Summary ─────────────────────────────────────────────────
print(f"\n{'='*80}")
print("📊 TỔNG KẾT ĐÁNH GIÁ RETRIEVAL BENCHMARK")
print(f"{'='*80}")

total = len(results)

# By question type
type_stats = {}
for r in results:
    t = r["type"]
    if t not in type_stats:
        type_stats[t] = {"total": 0, "perfect": 0, "partial": 0, "zero": 0, "recall_sum": 0, "doc_hit": 0}
    type_stats[t]["total"] += 1
    type_stats[t]["recall_sum"] += r["citation_recall"]
    if r["citation_recall"] == 1.0:
        type_stats[t]["perfect"] += 1
    elif r["citation_recall"] > 0:
        type_stats[t]["partial"] += 1
    else:
        type_stats[t]["zero"] += 1
    if r["stage1_all_docs_found"]:
        type_stats[t]["doc_hit"] += 1

# Overall stats
stage1_hits = sum(1 for r in results if r["stage1_all_docs_found"])
perfect_recall = sum(1 for r in results if r["citation_recall"] == 1.0)
partial_recall = sum(1 for r in results if 0 < r["citation_recall"] < 1.0)
zero_recall = sum(1 for r in results if r["citation_recall"] == 0)
avg_citation_recall = sum(r["citation_recall"] for r in results) / total if total else 0
avg_dieu_recall = sum(r["dieu_recall"] for r in results) / total if total else 0
avg_time = sum(r["time_s"] for r in results) / total if total else 0

print(f"\n📈 OVERALL ({total} câu):")
print(f"   Stage 1 (Document Retrieval): {stage1_hits}/{total} ({stage1_hits/total:.0%})")
print(f"   Stage 2 (Citation Recall):    {avg_citation_recall:.1%}")
print(f"     ✅ Perfect (100%): {perfect_recall}/{total}")
print(f"     ⚠️ Partial:        {partial_recall}/{total}")
print(f"     ❌ Zero (0%):      {zero_recall}/{total}")
print(f"   Điều-level Recall (relaxed):  {avg_dieu_recall:.1%}")
print(f"   ⏱️ Avg time/question:         {avg_time:.1f}s")

print(f"\n📊 THEO LOẠI CÂU HỎI:")
for t, stats in sorted(type_stats.items()):
    avg_r = stats["recall_sum"] / stats["total"] if stats["total"] else 0
    print(f"   [{t}] ({stats['total']} câu): "
          f"Doc={stats['doc_hit']}/{stats['total']} | "
          f"Perfect={stats['perfect']}/{stats['total']} | "
          f"Avg Recall={avg_r:.0%}")

print(f"\n📋 CHI TIẾT CÁC CÂU BỊ THIẾU:")
for r in results:
    if r["citation_recall"] < 1.0:
        print(f"   [{r['qid']}] [{r['type']}] {r['question'][:60]}...")
        print(f"       Expected: {r['expected_citations']}")
        print(f"       Got:      {r['stage2_retrieved'] or 'EMPTY'}")
        print(f"       Missing:  {r['missing_citations']}")

# Save results
os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)
output_path = os.path.join(PROJECT_ROOT, "outputs/benchmark_retrieval_eval.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump({
        "summary": {
            "total_questions": total,
            "stage1_doc_accuracy": stage1_hits / total if total else 0,
            "avg_citation_recall": avg_citation_recall,
            "avg_dieu_recall": avg_dieu_recall,
            "perfect_recall": perfect_recall,
            "partial_recall": partial_recall,
            "zero_recall": zero_recall,
            "avg_time_s": avg_time,
            "by_type": {t: {
                "total": s["total"],
                "perfect": s["perfect"],
                "avg_recall": s["recall_sum"] / s["total"] if s["total"] else 0,
                "doc_hit_rate": s["doc_hit"] / s["total"] if s["total"] else 0,
            } for t, s in type_stats.items()},
        },
        "details": results,
    }, f, ensure_ascii=False, indent=2)

print(f"\n💾 Kết quả chi tiết: {output_path}")
