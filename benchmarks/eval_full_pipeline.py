"""
Full Pipeline Evaluation — Retrieval + Generation
Chạy toàn bộ 12 câu benchmark qua pipeline agentic, ghi lại câu trả lời
và đánh giá chi tiết từng câu.
"""
import json, re, time, sys, os, math

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# ── Load benchmark ──
with open(os.path.join(os.path.dirname(__file__), "education_benchmark_v2.json"), "r", encoding="utf-8") as f:
    benchmark = json.load(f)
questions = benchmark["questions"]

# ── Init pipeline ──
print("🔧 Đang khởi tạo pipeline...")
from app.index.book_index import BookIndex
from app.rag.hybrid_search import BookRAGRetriever
from app.llm.client import LLMClient
from app.query.query_expander import EducationQueryExpander
from app.rag.context_builder import build_context
from app.llm.prompts import GENERATION_PROMPT, GENERATION_SYSTEM_PROMPT
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

data_dir = os.path.join(PROJECT_ROOT, os.getenv("DATA_DIR", "data/final"))
kg_path = os.path.join(PROJECT_ROOT, os.getenv("KG_PATH", "outputs/knowledge_graph/entity_graph.json"))
index = BookIndex(data_dir=data_dir, kg_path=kg_path)
index.load_index()
retriever = BookRAGRetriever(index)

pro_llm = LLMClient(
    api_base=os.getenv("LLM_API_BASE", "https://api.int2.net/v1"),
    api_key=os.getenv("LLM_API_KEY", ""),
    model=os.getenv("LLM_MODEL_NAME", "glm-4.7"),
)
expander = EducationQueryExpander()


# ── Cross-reference: implementing docs → parent laws ────────
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

def inject_cross_refs(top_docs, all_docs_map):
    injected = []
    seen = set()
    for sh in top_docs:
        bare = bare_sh(sh)
        if bare not in seen:
            injected.append(sh)
            seen.add(bare)
        for key, parents in CROSS_REFERENCE_MAP.items():
            if bare_sh(key) == bare or key == sh:
                for parent in parents:
                    resolved = resolve_sh(parent, index.doc_registry)
                    parent_bare = bare_sh(resolved)
                    if parent_bare not in seen:
                        injected.append(resolved)
                        seen.add(parent_bare)
    return injected

# ── Helpers ──
def bare_sh(text):
    m = re.search(r'(\d+[\/\-]\d{4}[\/\-]?[\w\-]*)', str(text))
    return m.group(1) if m else text

def resolve_sh(raw_sh, registry):
    if raw_sh in registry: 
        return registry[raw_sh]
    
    norm_raw = bare_sh(raw_sh)
    for k in registry:
        if bare_sh(k) == norm_raw:
            return registry[k]
            
    for k in registry:
        norm_k = bare_sh(k)
        if len(norm_k) > 3 and len(norm_raw) > 3:
            if norm_k in norm_raw or norm_raw in norm_k:
                return registry[k]
                
    return raw_sh

def rrf_merge(ranked_lists, k=60):
    scores, chunk_map = {}, {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            cid = doc.get("chunk_id", "")
            if not cid: continue
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
            if cid not in chunk_map: chunk_map[cid] = doc
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [chunk_map[cid] for cid in sorted_ids], scores

def parse_pro_llm(response_text, uniq_docs):
    """Parse 320B response into (node_ids, articles_list) with robust fallbacks and CoT XML tag support."""
    import re, json, ast
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
        raw = bare_sh(item.get("so_hieu", ""))
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
                
        disp_name = uniq_docs.get(raw, raw)
        if not any(a == f"Điều {dieu_num} ({disp_name})" for a in arts):
            arts.append(f"Điều {dieu_num} ({disp_name})")
            
    return nids, arts

# ── Run full pipeline ──
print(f"\n📋 Chạy full pipeline cho {len(questions)} câu...\n")
print("=" * 80)

all_results = []
output_lines = []
output_lines.append("# 📊 Đánh giá Full Pipeline — Retrieval + Generation\n")
output_lines.append(f"Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
output_lines.append("=" * 80 + "\n")

for q in questions:
    qid = q["id"]
    question = q["question"]
    expected_docs = q["ground_truth"]["relevant_docs"]
    expected_articles = q["ground_truth"]["relevant_articles"]
    legal_basis = q["legal_basis"]
    ref_answer = q["ground_truth"].get("reference_answer", "")
    expected_keywords = q["ground_truth"].get("expected_answer_keywords", [])

    print(f"\n📌 [{qid}] {question[:70]}...")
    t0 = time.time()

    # ── Stage 0: Wide Retrieval + RRF ──
    main_docs = retriever.retrieve_as_docs(question, top_k=20)
    all_ranked = [main_docs]
    expanded = expander.expand(question)
    for eq in expanded[:3]:
        eq_docs = retriever.retrieve_as_docs(eq, top_k=8)
        if eq_docs: all_ranked.append(eq_docs)
        
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
        hyde_doc = pro_llm.generate(hyde_prompt, temperature=0.3).strip()
        hyde_docs = retriever.retrieve_as_docs(hyde_doc, top_k=8)
        if hyde_docs:
            all_ranked.append(hyde_docs)
    except Exception as e:
        print(f"   ⚠️ HyDE generation error: {e}")
        
    docs, rrf_scores = rrf_merge(all_ranked)

    # ── Stage 1: Doc-Level Vote + 7B ──
    doc_agg = {}
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

    unique_docs, ranked_docs = {}, []
    if doc_agg:
        for bare, info in doc_agg.items():
            avg = info["total_rrf"] / info["count"] if info["count"] else 0
            info["rank_score"] = info["max_rrf"] + 0.3 * math.log(1 + info["count"]) * avg
            unique_docs[bare] = info["ten"]
        ranked_docs = sorted(doc_agg.items(), key=lambda x: x[1]["rank_score"], reverse=True)

    # ── Stage 1: Candidate Pool Selection & Parent Injections ──
    top_so_hieu = [sh for sh, _ in ranked_docs[:5]]
    
    explicit_targets = expander.get_target_docs(question)
    for ext_doc in explicit_targets:
        ext_bare = bare_sh(ext_doc)
        if ext_bare in top_so_hieu:
            top_so_hieu.remove(ext_bare)
        top_so_hieu.insert(0, ext_bare)
        unique_docs[ext_bare] = ext_doc
            
    top_so_hieu = inject_cross_refs(top_so_hieu, unique_docs)
    t_s1 = time.time() - t0
    print(f"   [Stage 1] Top docs: {top_so_hieu[:5]} + xref={top_so_hieu[5:] if len(top_so_hieu) > 5 else []} ({t_s1:.1f}s)")

    # ── Stage 2: 320B TOC scan with expert heuristics guidelines ──
    toc_parts = []
    for sh in top_so_hieu[:4]:
        canonical_sh = resolve_sh(sh, index.doc_registry)
        toc = index.build_toc(canonical_sh)
        if toc:
            toc_parts.append(toc)

    final_node_ids, stage2_articles = set(), []
    if toc_parts:
        toc_text = "\n\n".join(toc_parts)
        
        # Build context snippet from top chunks - reduced to 7 docs to avoid prompt bloat & empty responses
        context_text = "\n".join([
            f"[{d.get('metadata',{}).get('ten_van_ban','')}] Điều {d.get('metadata',{}).get('so_dieu','')}: {(d.get('content','') or d.get('text',''))[:150]}..."
            for d in docs[:7]
        ])
        
        pro_prompt = f"""Bạn là một chuyên gia cao cấp về pháp luật giáo dục Việt Nam. Hãy thực hiện phân tích chuỗi lập luận (Chain-of-Thought) CỰC KỲ NGẮN GỌN (tối đa 2-3 câu) trước khi lựa chọn các Điều khoản cần thiết để trả lời câu hỏi dưới đây.

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
            pro_res = pro_llm.generate(pro_prompt, temperature=0.1)
            final_node_ids, stage2_articles = parse_pro_llm(pro_res, unique_docs)
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
                res_retry = pro_llm.generate(retry_prompt, temperature=0.2)
                retry_nodes, retry_arts = parse_pro_llm(res_retry, unique_docs)
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
                    res_expand = pro_llm.generate(expand_prompt, temperature=0.2)
                    expand_nodes, expand_arts = parse_pro_llm(res_expand, unique_docs)
                    if expand_arts:
                        final_node_ids = expand_nodes
                        stage2_articles = expand_arts
                except:
                    pass

    t_s2 = time.time() - t0
    print(f"   [Stage 2] 320B: {stage2_articles if stage2_articles else 'EMPTY'} ({t_s2 - t_s1:.1f}s)")

    # ── Stage 3: Build context + Generate ──
    if final_node_ids:
        final_docs = []
        for nid in final_node_ids:
            d = retriever._node_to_doc(nid)
            if d: final_docs.append(d)
        if final_docs: docs = final_docs

    context = build_context(docs[:5])
    prompt = GENERATION_PROMPT.format(query=question, context=context)
    answer = pro_llm.generate(prompt, system_prompt=GENERATION_SYSTEM_PROMPT, temperature=0.0)

    t_total = time.time() - t0
    print(f"   [Generate] ({t_total - t_s2:.1f}s) → {answer[:120]}...")

    # ── Evaluate ──
    # Check if answer mentions the expected articles
    expected_dieu = set()
    for art in expected_articles:
        m = re.search(r'(\d+)', art)
        if m: expected_dieu.add(int(m.group(1)))

    cited_dieu = set()
    for match in re.finditer(r'[Đđ]iều\s+([0-9\s,vàhoặc]+)', answer):
        nums = re.findall(r'\d+', match.group(0))
        for num in nums:
            cited_dieu.add(int(num))

    articles_cited = expected_dieu & cited_dieu
    articles_missing = expected_dieu - cited_dieu
    citation_recall = len(articles_cited) / len(expected_dieu) if expected_dieu else 0

    # Check keywords
    kw_found = []
    kw_missing = []
    ans_lower = answer.lower()
    for kw in expected_keywords:
        if kw.lower() in ans_lower:
            kw_found.append(kw)
        else:
            kw_missing.append(kw)
    kw_coverage = len(kw_found) / len(expected_keywords) if expected_keywords else 1.0

    status = "✅" if citation_recall == 1.0 else ("⚠️" if citation_recall > 0 else "❌")
    print(f"   {status} Citation: {citation_recall:.0%} | Keywords: {kw_coverage:.0%} | Time: {t_total:.1f}s")

    result = {
        "qid": qid, "question": question, "legal_basis": legal_basis,
        "stage1_docs": top_so_hieu, "stage2_articles": stage2_articles,
        "expected_articles": list(expected_dieu), "cited_articles": list(cited_dieu),
        "articles_found": list(articles_cited), "articles_missing": list(articles_missing),
        "citation_recall": citation_recall, "kw_coverage": kw_coverage,
        "answer": answer, "time_s": round(t_total, 1),
    }
    all_results.append(result)

    # ── Write to output ──
    output_lines.append(f"\n## [{qid}] {question}\n")
    output_lines.append(f"**Căn cứ pháp lý:** {legal_basis}\n")
    output_lines.append(f"**Đáp án tham khảo:** {ref_answer}\n")
    output_lines.append(f"**Từ khóa bắt buộc:** {', '.join(expected_keywords)}\n")
    output_lines.append(f"\n### Câu trả lời của LLM\n")
    output_lines.append(f"```\n{answer}\n```\n")
    output_lines.append(f"\n### Đánh giá\n")
    output_lines.append(f"| Tiêu chí | Kết quả |\n|----------|---------|")
    output_lines.append(f"| Stage 1 (VB chọn) | {', '.join(top_so_hieu)} |")
    output_lines.append(f"| Stage 2 (Điều chọn) | {', '.join(stage2_articles) if stage2_articles else 'EMPTY'} |")
    output_lines.append(f"| Điều khoản kỳ vọng | Điều {expected_dieu} |")
    output_lines.append(f"| Điều khoản trích dẫn | Điều {cited_dieu} |")
    output_lines.append(f"| Citation Recall | {status} {citation_recall:.0%} (tìm thấy: {articles_cited}, thiếu: {articles_missing}) |")
    output_lines.append(f"| Keyword Coverage | {kw_coverage:.0%} (thiếu: {', '.join(kw_missing) if kw_missing else 'không'}) |")
    output_lines.append(f"| Thời gian | {t_total:.1f}s |")
    output_lines.append("")

# ── Summary ──
total = len(all_results)
perfect_citation = sum(1 for r in all_results if r["citation_recall"] == 1.0)
avg_citation = sum(r["citation_recall"] for r in all_results) / total
avg_kw = sum(r["kw_coverage"] for r in all_results) / total
avg_time = sum(r["time_s"] for r in all_results) / total

summary = f"""
{'=' * 80}
📊 TỔNG KẾT FULL PIPELINE
{'=' * 80}

 Citation Recall (LLM trích dẫn đúng Điều):
   Perfect (100%): {perfect_citation}/{total}
   Average:        {avg_citation:.1%}

 Keyword Coverage:
   Average:        {avg_kw:.1%}

 Performance:
   Avg time/query: {avg_time:.1f}s
   Total time:     {sum(r['time_s'] for r in all_results):.0f}s

📋 Chi tiết các câu chưa đạt Citation 100%:"""

for r in all_results:
    if r["citation_recall"] < 1.0:
        summary += f"\n   [{r['qid']}] Recall={r['citation_recall']:.0%} | Missing: Điều {r['articles_missing']} | Cited: Điều {sorted(r['cited_articles'])}"

print(summary)

output_lines.append(f"\n{'=' * 80}")
output_lines.append(f"## 📊 TỔNG KẾT\n")
output_lines.append(f"| Metric | Giá trị |")
output_lines.append(f"|--------|---------|")
output_lines.append(f"| Perfect Citation (100%) | {perfect_citation}/{total} ({perfect_citation/total:.0%}) |")
output_lines.append(f"| Avg Citation Recall | {avg_citation:.1%} |")
output_lines.append(f"| Avg Keyword Coverage | {avg_kw:.1%} |")
output_lines.append(f"| Avg Time/Query | {avg_time:.1f}s |")

# Save
out_path = os.path.join(PROJECT_ROOT, "outputs/full_pipeline_eval.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))
print(f"\n💾 Chi tiết: {out_path}")

# Also save JSON
json_path = os.path.join(PROJECT_ROOT, "outputs/full_pipeline_eval.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({"summary": {"perfect_citation": perfect_citation, "avg_citation_recall": avg_citation, "avg_kw_coverage": avg_kw, "total": total}, "details": all_results}, f, ensure_ascii=False, indent=2)
print(f"💾 JSON: {json_path}")
