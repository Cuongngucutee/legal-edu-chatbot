import json, re, time, os, sys, math, unicodedata

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from app.index.book_index import BookIndex
from app.rag.hybrid_search import BookRAGRetriever
from app.llm.client import LLMClient
from app.rag.pipeline import LawEduPipeline, FailureStateProber
from app.query.intent_classifier import classify_intent
from app.rag.context_builder import build_context, postprocess_citations, fmt_source
from app.llm.prompts import GENERATION_PROMPT, GENERATION_SYSTEM_PROMPT

# Cross-reference map
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

def normalize_text(text):
    if not text: return ""
    text = str(text).lower().strip()
    text = text.replace("đ", "d")
    nfd_form = unicodedata.normalize('NFD', text)
    return "".join([c for c in nfd_form if not unicodedata.combining(c)])

def bare_sh(text):
    text_norm = normalize_text(text)
    m = re.search(r'(\d+[\/-]\d{4}[\/-]?[\w\-]*)', text_norm)
    return m.group(1) if m else text_norm

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

print("🔧 Khởi tạo Vector Index và Pipeline...")
index = BookIndex(data_dir=os.path.join(PROJECT_ROOT, "data/final"), kg_path=os.path.join(PROJECT_ROOT, "outputs/knowledge_graph/entity_graph.json"))
index.load_index()
retriever = BookRAGRetriever(index)

llm_320b = LLMClient(
    api_base=os.getenv("LLM_API_BASE"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL_NAME"),
)
pipeline = LawEduPipeline(retriever=retriever, agentic_llm=llm_320b, generator_llm=llm_320b)

# Select Q1 as the perfect complex scenario to trace
BENCHMARK_PATH = os.path.join(PROJECT_ROOT, "benmark.json")
with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    benchmark = json.load(f)
test_case = benchmark[2]
query = test_case["question"]

print(f"\n🚀 Khởi chạy Tracing chi tiết cho câu hỏi:\n\"{query}\"\n")

trace_steps = []
t_total_start = time.time()

# ── 1. PHÂN LOẠI Ý ĐỊNH (INTENT CLASSIFICATION) ──
print("🏷️  Chặng 1: Phân loại ý định...")
t_start = time.time()
intent = classify_intent(query, llm_320b)
t_end = time.time()
latency = t_end - t_start
trace_steps.append({
    "stage": "Chặng 1: Phân loại ý định (Intent Classification)",
    "description": "Phân loại câu hỏi của người dùng để quyết định luồng xử lý hoặc handler chuyên biệt.",
    "latency": latency,
    "input": f"Query: \"{query}\"",
    "output": f"Intent: {intent}"
})

# ── 2. TRUY VẤN VÀ GỘP HYBRID SEARCH (STAGE 1 RETRIEVAL & RRF MERGE) ──
print("🎬 Chặng 2: Hybrid Retrieval & RRF Merge...")
t_start = time.time()
main_docs = retriever.retrieve_as_docs(query, top_k=20)
all_ranked = [main_docs]

# Expand query
expanded = pipeline.query_expander.expand(query)
for eq in expanded[:3]:
    eq_docs = retriever.retrieve_as_docs(eq, top_k=8)
    if eq_docs:
        all_ranked.append(eq_docs)
        
# HyDE Generation & Retrieval
hyde_prompt = (
    "Bạn là chuyên gia pháp luật giáo dục Việt Nam.\n"
    "Hãy viết một đoạn văn ngắn (2-4 câu) mô tả quy định pháp luật giả định chính xác nhất để trả lời cho câu hỏi sau.\n"
    "Hãy sử dụng văn phong văn bản luật chính xác, trang trọng và khách quan.\n"
    "Không cần mở đầu bằng lời chào hay giải thích, hãy viết thẳng nội dung quy định giả định.\n\n"
    f"Câu hỏi: {query}\n\n"
    "Quy định pháp luật giả định:"
)
hyde_doc = llm_320b.generate(hyde_prompt, temperature=0.3).strip()
hyde_docs = retriever.retrieve_as_docs(hyde_doc, top_k=8)
if hyde_docs:
    all_ranked.append(hyde_docs)

# RRF Merge
docs, rrf_scores = pipeline._rrf_merge(all_ranked)

t_end = time.time()
latency = t_end - t_start
trace_steps.append({
    "stage": "Chặng 2: Truy xuất hỗn hợp & Trộn điểm RRF (Stage 1 & RRF Merge)",
    "description": "Kích hoạt tìm kiếm Hybrid (BM25 + Vector), mở rộng câu hỏi với Query Expander và HyDE, sau đó hợp nhất kết quả bằng thuật toán Reciprocal Rank Fusion (RRF).",
    "latency": latency,
    "input": f"Query gốc: \"{query}\"\nQuery mở rộng: {expanded[:3]}\nĐoạn văn HyDE sinh ra: \"{hyde_doc}\"",
    "output": f"Tổng số chunks sau RRF: {len(docs)}\nTop 3 Chunk IDs: {[d.get('chunk_id') for d in docs[:3]]}\nTop 3 Chunks tóm tắt: {[d.get('content')[:100] + '...' for d in docs[:3]]}"
})

# ── 3. VOTE CHO CẤP ĐỘ VĂN BẢN (DOC-LEVEL VOTE AGGREGATION) ──
print("🗳️  Chặng 3: Doc-Level Vote Aggregation...")
t_start = time.time()
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

top_so_hieu = [sh for sh, _ in ranked_docs[:5]]

# Explicit target forcing
explicit_targets = pipeline.query_expander.get_target_docs(query)
for ext_doc in explicit_targets:
    ext_bare = bare_sh(ext_doc)
    if ext_bare in top_so_hieu:
        top_so_hieu.remove(ext_bare)
    top_so_hieu.insert(0, ext_bare)
    unique_docs[ext_bare] = ext_doc

t_end = time.time()
latency = t_end - t_start
trace_steps.append({
    "stage": "Chặng 3: Tính toán độ liên quan cấp Văn bản (Doc-Level Vote Aggregation)",
    "description": "Gộp điểm RRF của các chunks thành điểm độ liên quan cấp văn bản (Document-level) bằng công thức bỏ phiếu logarit, kết hợp ép buộc tài liệu trích dẫn trực tiếp từ query.",
    "latency": latency,
    "input": f"Số lượng văn bản ứng cử viên: {len(doc_agg)}",
    "output": "Xếp hạng các văn bản: " + str([f"{sh} (Score: {info['rank_score']:.4f})" for sh, info in ranked_docs[:5]]) + f"\nTop 5 Văn bản được chọn: {top_so_hieu}"
})

# ── 4. TIÊM VĂN BẢN THAM CHIẾU CHÉO (CROSS-REFERENCE PARENT INJECTION) ──
print("🔗 Chặng 4: Parent Cross-Reference Injection...")
t_start = time.time()
injected_docs = pipeline._inject_cross_refs(top_so_hieu, unique_docs)
t_end = time.time()
latency = t_end - t_start
trace_steps.append({
    "stage": "Chặng 4: Tiêm văn bản tham chiếu chéo (Cross-Reference Parent Injection)",
    "description": "Quét danh mục các Nghị định/Thông tư hướng dẫn thi hành và tự động 'tiêm' thêm Luật gốc vào danh sách văn bản ứng cử viên để tránh lỗi thiếu Luật gốc chéo.",
    "latency": latency,
    "input": f"Văn bản đầu vào: {top_so_hieu}",
    "output": f"Văn bản sau khi tiêm chéo: {injected_docs}"
})

# ── 5. XÂY DỰNG MỤC LỤC CHI TIẾT (ENRICHED TOC GENERATION) ──
print("📚 Chặng 5: Enriched TOC Generation...")
t_start = time.time()
toc_parts = []
for sh in injected_docs[:4]:
    canonical_sh = resolve_sh(sh, index.doc_registry)
    toc = index.build_enriched_toc(canonical_sh, snippet_len=200)
    if toc:
        toc_parts.append(toc)
toc_text = "\n\n".join(toc_parts)

t_end = time.time()
latency = t_end - t_start
trace_steps.append({
    "stage": "Chặng 5: Xây dựng Mục lục chi tiết kèm Snippet (Enriched TOC Generation)",
    "description": "Xây dựng mục lục đầy đủ của top 4 văn bản pháp luật hàng đầu, mỗi Điều khoản được đính kèm thêm 200 ký tự tóm tắt nội dung để giúp LLM ở chặng sau chọn chính xác Điều.",
    "latency": latency,
    "input": f"Top 4 văn bản xây dựng mục lục: {injected_docs[:4]}",
    "output": f"Kích thước văn bản mục lục TOC (ký tự): {len(toc_text)}\nXem trước 300 ký tự mục lục: \"{toc_text[:300]}...\""
})

# ── 6. LỰA CHỌN ĐIỀU KHOẢN COT (STAGE 2 COT TOC SELECTION) ──
print("🧠 Chặng 6: Stage 2 CoT TOC Selection...")
t_start = time.time()
context_text = "\n".join([
    f"[{d.get('metadata',{}).get('ten_van_ban','')}] Điều {d.get('metadata',{}).get('so_dieu','')}: {(d.get('content','') or d.get('text',''))[:150]}..."
    for d in docs[:7]
])

prompt_320b = f"""Bạn là một chuyên gia cao cấp về pháp luật giáo dục Việt Nam. Hãy đọc kỹ MỤC LỤC CHI TIẾT (bao gồm nội dung tóm tắt của từng Điều) và chọn các Điều khoản cần thiết để trả lời câu hỏi.

Câu hỏi: \"{query}\"

Các trích đoạn liên quan (tham khảo):
{context_text}

Mục lục chi tiết các văn bản pháp luật (bao gồm tóm tắt nội dung):
{toc_text}

NGUYÊN TẮC CHỌN ĐIỀU BẮT BUỘC:
1. ĐỌC KỸ SNIPPET NỘI DUNG (dòng bắt đầu bằng →) của từng Điều để chọn chính xác. KHÔNG chỉ dựa vào tiêu đề.
2. Nếu câu hỏi liên quan đến VĂN BẢN SỬA ĐỔI → LUÔN chọn ĐỒNG THỜI: Điều gốc VÀ Điều sửa đổi (VD: Điều 4 TT 01/2021 VÀ Điều 1 TT 08/2023).
3. Nếu câu hỏi về điều kiện/tiêu chuẩn CỤ THỂ (con số, thời gian, bằng cấp) → chọn Điều chứa CON SỐ CỤ THỂ trong snippet, KHÔNG chọn Điều chỉ nêu nguyên tắc chung.
4. Nếu câu hỏi cần cả Luật gốc lẫn Nghị định hướng dẫn → chọn ĐỒNG THỜI từ cả 2 văn bản.

HƯỚNG DẪN CHUYÊN NGÀNH:
1. CHỨC DANH GIÁO VIÊN (TT 01,02,03,04/2021 + sửa đổi TT 08/2023):
   - Mầm non: Chọn Điều gốc TT 01/2021 VÀ Điều 1 TT 08/2023.
   - Tiểu học: Chọn Điều gốc TT 02/2021 VÀ Điều 2 TT 08/2023.
   - THCS: Chọn Điều gốc TT 03/2021 VÀ Điều 3 TT 08/2023.
   - THPT: Chọn Điều gốc TT 04/2021 VÀ Điều 4 TT 08/2023.
2. SINH VIÊN SƯ PHẠM (NĐ 116/2020):
   - Mức hỗ trợ: Điều 4. Bảo lưu/nghỉ tạm thời: Điều 6. Bồi hoàn: Điều 6 + Điều 8. Thu hồi: Điều 9.
3. ĐẠI HỌC (Luật 34/2018 + NĐ 99/2019): Luôn chọn ĐỒNG THỜI Điều 1 Luật 34/2018 VÀ Điều tương ứng NĐ 99/2019.
4. MẦM NON KHU CÔNG NGHIỆP: Chọn ĐỒNG THỜI Luật 43/2019 (Điều 17/26/102) VÀ NĐ 105/2020 (Điều 5).
5. HỌC BỔNG CỬ TUYỂN: Luôn chọn Điều 9 NĐ 84/2020/NĐ-CP.
6. THI TỐT NGHIỆP: Chọn Điều 2 + Điều 3 TT 24/2024.
7. CTGDPT MỚI: Chọn ĐỒNG THỜI Điều 2 + Điều 3 TT 32/2018.
8. QUYỀN NHÀ GIÁO (Luật 43/2019): Thỉnh giảng → Điều 70. Hành vi bị cấm → Điều 22.
9. NÂNG CHUẨN GIÁO VIÊN (NĐ 71/2020): Đối tượng áp dụng → Điều 2. Lộ trình → Điều 5/6.
10. KIỂM TRA ĐÁNH GIÁ HỌC SINH (TT 22/2021): Kiểm tra bù → Điều 7. Miễn thực hành → Điều 10. Lên lớp → Điều 12. Đánh giá lại → Điều 14.
11. SỞ HỮU TÀI SẢN TRƯỜNG TƯ (Luật 43/2019): Luôn chọn Điều 102.
12. HỌC PHÍ TIỂU HỌC TƯ THỤC (Luật 43/2019): Chọn Điều 14 + Điều 99.

Nhiệm vụ:
1. Lập luận TÓM TẮT (1-2 câu).
2. Đặt JSON trong <selected_clauses>...</selected_clauses>.

<selected_clauses>
[
  {{"so_hieu": "01/2021/TT-BGDĐT", "dieu": 5}},
  {{"so_hieu": "08/2023/TT-BGDĐT", "dieu": 1}}
]
</selected_clauses>"""

res_320b = llm_320b.generate(prompt_320b, temperature=0.1)
final_node_ids, stage2_articles = pipeline._parse_320b(res_320b, unique_docs)

t_end = time.time()
latency = t_end - t_start
trace_steps.append({
    "stage": "Chặng 6: Trích xuất Điều khoản chính xác bằng LLM (Stage 2 CoT Selection)",
    "description": "Gửi mục lục chi tiết và các trích đoạn tham khảo đến mô hình LLM 320B để thực hiện suy luận chuỗi lập luận (Chain-of-Thought) để chọn ra chính xác các số Điều chứa câu trả lời.",
    "latency": latency,
    "input": f"Kích thước Prompt gửi đến LLM (ký tự): {len(prompt_320b)}",
    "output": f"Câu trả lời CoT của LLM: \"{res_320b.strip()}\"\nCác Điều khoản phân tích được: {stage2_articles}"
})

# ── 7. LIÊN KẾT THAM CHIẾU PHÁP LÝ ĐỘNG (DYNAMIC LEGAL RESOLVER) ──
print("🔗 Chặng 7: Dynamic Legal Cross-Reference Linking...")
t_start = time.time()
injected_pairings = pipeline._dynamic_resolve_pairings(stage2_articles, injected_docs, query)
injected_list = []
for target_node_id, disp_str in injected_pairings:
    if target_node_id not in final_node_ids:
        final_node_ids.add(target_node_id)
        if disp_str not in stage2_articles:
            stage2_articles.append(disp_str)
            injected_list.append(disp_str)

t_end = time.time()
latency = t_end - t_start
trace_steps.append({
    "stage": "Chặng 7: Liên kết tham chiếu chéo động (Dynamic Legal Cross-Reference Linking)",
    "description": "Sử dụng đồ thị quan hệ tự động phân tích ngữ cảnh (same-doc & cross-doc) để tự động 'tiêm' thêm các Điều khoản sửa đổi hoặc Điều khoản liên kết hữu cơ bị bỏ sót.",
    "latency": latency,
    "input": f"Các Điều khoản đã chọn: {stage2_articles}\nCác Văn bản đang hoạt động: {injected_docs}",
    "output": f"Các Điều khoản liên kết chéo động được tiêm thêm: {injected_list}\nDanh sách Điều khoản sau cùng: {stage2_articles}"
})

# ── 8. SINH CÂU TRẢ LỜI RAG (STAGE 3 GENERATION) ──
print("✍️  Chặng 8: Stage 3 Generation...")
t_start = time.time()
final_nodes = list(final_node_ids)
final_docs = []
for nid in final_nodes:
    d = retriever._node_to_doc(nid)
    if d:
        final_docs.append(d)
        
if not final_docs:
    final_docs = docs[:8]
else:
    final_docs = final_docs[:8]
    
context = build_context(final_docs, max_chars=16000)
generation_input_prompt = GENERATION_PROMPT.format(query=query, context=context)

answer = llm_320b.generate(
    generation_input_prompt,
    system_prompt=GENERATION_SYSTEM_PROMPT,
    temperature=0.1,
).strip()

t_end = time.time()
latency = t_end - t_start
trace_steps.append({
    "stage": "Chặng 8: Sinh câu trả lời RAG lần một (Stage 3 Generation)",
    "description": "Xây dựng context hoàn chỉnh lên tới tối đa 16000 ký tự chứa toàn bộ nội dung chi tiết của các Điều khoản đã chọn lọc và truyền vào LLM 320B để sinh ra câu trả lời chi tiết kèm trích dẫn.",
    "latency": latency,
    "input": f"Kích thước Context (ký tự): {len(context)}\nKích thước Prompt gửi đến LLM (ký tự): {len(generation_input_prompt)}",
    "output": f"Câu trả lời thô của LLM: \"{answer}\""
})

# ── 9. TỰ KIỂM DUYỆT TRÍCH DẪN (CITATION REFLECTION & REWRITE) ──
print("🔄 Chặng 9: Citation Reflection & Rewrite...")
t_start = time.time()
missing_any = False
cited_dieu = set()
for match in re.finditer(r'[Đđ]iều\s+([0-9\s,vàhoặc]+)', answer):
    nums = re.findall(r'\d+', match.group(0))
    for num in nums:
        cited_dieu.add(int(num))
        
for item_str in stage2_articles:
    m_num = re.search(r'Điều\s+(\d+)', item_str)
    if m_num:
        num_val = int(m_num.group(1))
        if num_val not in cited_dieu:
            missing_any = True
            break
            
reflected_answer_raw = ""
if missing_any:
    reflection_prompt = f"""Bạn là một chuyên gia kiểm duyệt pháp lý tối cao. 
Câu hỏi của người dùng: \"{query}\"
Câu trả lời hiện tại:
\"{answer}\"

Yêu cầu bắt buộc: Hiệu chỉnh lại câu trả lời trên để tích hợp trực tiếp và tự nhiên các trích dẫn Điều khoản pháp lý cụ thể sau đây: {stage2_articles}.
Vui lòng viết lại câu trả lời, đảm bảo giữ nguyên tính chính xác, ngắn gọn, và chèn các số Điều đã chọn một cách chính xác nhất."""

    reflected_answer_raw = llm_320b.generate(
        reflection_prompt,
        system_prompt="Bạn là chuyên gia hiệu chỉnh pháp lý chính xác và chuyên nghiệp.",
        temperature=0.1,
    ).strip()
    if reflected_answer_raw and not reflected_answer_raw.startswith("[LLM Error"):
        answer = reflected_answer_raw

t_end = time.time()
latency = t_end - t_start
trace_steps.append({
    "stage": "Chặng 9: Tự kiểm duyệt và hiệu chỉnh trích dẫn (Citation Reflection)",
    "description": "Quét và đối chiếu câu trả lời thô của LLM với danh sách Điều khoản bắt buộc ở Stage 2. Nếu phát hiện thiếu bất kỳ Điều khoản nào, lập tức chạy quy trình Reflection tự sửa đổi và chèn bổ sung trích dẫn.",
    "latency": latency,
    "input": f"Thiếu trích dẫn bắt buộc: {missing_any}\nCác Điều đã trích dẫn ở bước trước: {cited_dieu}",
    "output": f"Câu trả lời sau tự hiệu chỉnh Reflection: \"{answer}\""
})

# ── 10. HẬU XỬ LÝ VÀ ĐẦU RA SAU CÙNG (CITATION POSTPROCESSING & OUTPUT) ──
print("💾 Chặng 10: Citation Postprocessing & Output...")
t_start = time.time()
final_answer = postprocess_citations(answer, final_docs)
t_end = time.time()
latency = t_end - t_start
trace_steps.append({
    "stage": "Chặng 10: Hậu xử lý và Đầu ra sau cùng (Citation Postprocessing & Output)",
    "description": "Quy chuẩn hóa và làm đẹp toàn bộ định dạng trích dẫn pháp lý, định hình cấu trúc đầu ra Markdown hoàn thiện nhất trả về cho người dùng.",
    "latency": latency,
    "input": f"Câu trả lời đầu vào: \"{answer}\"",
    "output": f"Câu trả lời cuối cùng: \"{final_answer}\"\nCác nguồn trích dẫn đính kèm: {[fmt_source(d) for d in final_docs]}"
})

t_total_end = time.time()
total_latency = t_total_end - t_total_start

# ── XÂY DỰNG BÁO CÁO TRACE LOGS CỰC ĐẸP (MD FILE) ──
print("💾 Đang xuất báo cáo log chi tiết ra outputs/detailed_pipeline_trace.md...")

output_lines = [
    "# 🔬 BÁO CÁO LOG TRUY VẾT CHI TIẾT HỆ THỐNG PIPELINE RAG (DETAILED TRACE LOGS)",
    "",
    "Báo cáo này phân tích cực kỳ chi tiết **từng chặng (stage)** nhỏ nhất trong luồng xử lý Agentic RAG Pipeline của câu hỏi thực tế. Mỗi bước được ghi nhận cụ thể về **mục tiêu, thuật toán, thời gian thực thi (latency), dữ liệu đầu vào và dữ liệu đầu ra thực tế**.",
    "",
    "## 📊 1. Thống Kê Hiệu Năng Tổng Quan",
    "",
    f"- **Câu hỏi kiểm thử:** \"{query}\"",
    f"- **Tổng thời gian xử lý toàn luồng (Total Latency):** **{total_latency:.2f} giây**",
    "- **Danh sách các chặng thực thi:**",
    ""
]

# Generate run statistics in list format
for idx, step in enumerate(trace_steps):
    step_num = idx + 1
    pct = (step["latency"] / total_latency) * 100
    output_lines.append(f"{step_num}. **{step['stage']}:** **{step['latency']:.2f}s** ({pct:.1f}% tổng thời gian)")

output_lines.append("")
output_lines.append("---")
output_lines.append("## 🔍 2. Nhật Ký Chi Tiết Từng Chặng (Detailed Stages Log)")
output_lines.append("")

for idx, step in enumerate(trace_steps):
    step_num = idx + 1
    pct = (step["latency"] / total_latency) * 100
    output_lines.append(f"### 📍 Chặng {step_num}: {step['stage']}")
    output_lines.append(f"- **Mô tả chặng:** {step['description']}")
    output_lines.append(f"- **Thời gian chạy:** **{step['latency']:.2f}s** ({pct:.1f}%)")
    output_lines.append("- **📥 Dữ liệu đầu vào (Input):**")
    output_lines.append("```text")
    output_lines.append(step["input"])
    output_lines.append("```")
    output_lines.append("- **📤 Kết quả đầu ra (Output):**")
    output_lines.append("```text")
    output_lines.append(step["output"])
    output_lines.append("```")
    output_lines.append("")
    output_lines.append("---")

output_lines.append("")
output_lines.append("## 💾 3. Bản Sao Câu Trả Lời Cuối Cùng")
output_lines.append("")
output_lines.append("```markdown")
output_lines.append(final_answer)
output_lines.append("```")

out_path = os.path.join(PROJECT_ROOT, "outputs/detailed_pipeline_trace.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print(f"\n✅ Đã lưu báo cáo trace log chi tiết tại: {out_path}")
