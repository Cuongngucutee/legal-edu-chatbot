"""
Full Pipeline Evaluation on 44-Question Benchmark — Retrieval + Generation
Chạy toàn bộ 44 câu benchmark qua pipeline agentic, ghi lại câu trả lời và đánh giá chi tiết từng câu.
"""
import json, re, time, sys, os, math
import unicodedata

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# ── Load benchmark ──
BENCHMARK_PATH = os.path.join(PROJECT_ROOT, "benmark.json")
with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    benchmark = json.load(f)

print(f"📋 Loaded {len(benchmark)} test cases from benmark.json")

# ── Init pipeline ──
print("🔧 Đang khởi tạo pipeline...")
from app.index.book_index import BookIndex
from app.rag.hybrid_search import BookRAGRetriever
from app.llm.client import LLMClient
from app.rag.pipeline import LawEduPipeline
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

pipeline = LawEduPipeline(retriever=retriever, agentic_llm=pro_llm, generator_llm=pro_llm)

# ── Parsing Helpers ──
def parse_citation(citation_str: str):
    dieu_nums = [int(n) for n in re.findall(r'[Đđ]iều\s+(\d+)', citation_str)]
    
    so_hieu_match = re.search(r'(\d+/\d{4}/[\w\-]+)', citation_str)
    so_hieu = so_hieu_match.group(1) if so_hieu_match else None
    
    if not so_hieu:
        alt_match = re.search(r'(Luật|Nghị định|Thông tư|Quyết định)[^\d]*(\d+[\/-]\d{4}[\/-]?[\w\-]*)', citation_str)
        if alt_match:
            so_hieu = alt_match.group(2)
    return so_hieu, dieu_nums

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

# ── Main Run Loop ──
all_results = []
output_lines = [
    "# 🚀 BÁO CÁO ĐÁNH GIÁ FULL PIPELINE RAG — 44 CÂU BENCHMARK",
    "",
    "Báo cáo này ghi nhận kết quả đánh giá cuối cùng của **Full Pipeline RAG (Retrieval + Generation)** trên toàn bộ 44 câu hỏi của bộ benchmark.",
    "",
]

print(f"\n================================================================================")
print(f"📋 BẮT ĐẦU CHẤM FULL PIPELINE 44 CÂU HỎI BENCHMARK")
print(f"================================================================================")

for idx, test_case in enumerate(benchmark):
    time.sleep(0.1)  # Đã nâng cấp API lên 250k TPM / 500k RPD nên không cần pacing
    qid = idx + 1
    question = test_case["question"]
    q_type = test_case.get("type", "Unknown")
    ref_answer = test_case.get("answer", "")
    
    # Parse expected citations
    expected_citations = test_case.get("citations", [])
    citation_requirements = []
    expected_docs = set()
    all_expected_dieu = set()
    for cit in expected_citations:
        sh, dieu_list = parse_citation(cit)
        if sh:
            expected_docs.add(bare_sh(sh))
        if dieu_list:
            citation_requirements.append(dieu_list)
            for d in dieu_list:
                all_expected_dieu.add(d)
            
    print(f"\n📌 [EDU_44_Q{qid:02d}] [{q_type}] {question[:85]}...")
    print(f"   Kỳ vọng: Điều {all_expected_dieu} của văn bản {list(expected_docs)}")
    
    t0 = time.time()
    
    # Run the unified agentic RAG lookup flow
    try:
        res = pipeline._lookup_flow(question, t0, "LOOKUP")
        answer = res["answer"]
        top_so_hieu = res["stage1_docs"]
        stage2_articles = res["stage2_articles"]
    except Exception as e:
        print(f"   ⚠️ Pipeline execution error: {e}")
        answer = ""
        top_so_hieu = []
        stage2_articles = []
        
    t_total = time.time() - t0
    print(f"   [Done] ({t_total:.1f}s) → {answer[:120]}...")
    
    # ── Stage 1 Evaluation (Retrieval Recall) ──
    stage1_retrieved_docs = {bare_sh(sh) for sh in top_so_hieu}
    stage1_found = expected_docs & stage1_retrieved_docs
    stage1_recall = len(stage1_found) / len(expected_docs) if expected_docs else 1.0
    stage1_status = "✅" if stage1_recall == 1.0 else ("⚠️" if stage1_recall > 0 else "❌")

    # ── Stage 2 Evaluation (Selection Recall) ──
    stage2_dieu = set()
    for item in stage2_articles:
        m = re.search(r'Điều\s+(\d+)', item)
        if m:
            stage2_dieu.add(int(m.group(1)))
    stage2_covered_count = 0
    stage2_missing_requirements = []
    for req_list in citation_requirements:
        if any(d in stage2_dieu for d in req_list):
            stage2_covered_count += 1
        else:
            stage2_missing_requirements.append(req_list)
    stage2_recall = stage2_covered_count / len(citation_requirements) if citation_requirements else 1.0
    stage2_status = "✅" if stage2_recall == 1.0 else ("⚠️" if stage2_recall > 0 else "❌")

    # ── Stage 3 Evaluation (LLM Generation Citation Recall) ──
    cited_dieu = set()
    for match in re.finditer(r'[Đđ]iều\s+([0-9\s,vàhoặc]+)', answer):
        nums = re.findall(r'\d+', match.group(0))
        for num in nums:
            cited_dieu.add(int(num))
            
    stage3_covered_count = 0
    stage3_missing_requirements = []
    for req_list in citation_requirements:
        if any(d in cited_dieu for d in req_list):
            stage3_covered_count += 1
        else:
            stage3_missing_requirements.append(req_list)
    stage3_recall = stage3_covered_count / len(citation_requirements) if citation_requirements else 1.0
    stage3_status = "✅" if stage3_recall == 1.0 else ("⚠️" if stage3_recall > 0 else "❌")

    print(f"   [Stage 1 Retrieval]: {stage1_status} Recall={stage1_recall:.0%} | [Stage 2 Selection]: {stage2_status} Recall={stage2_recall:.0%} | [Stage 3 Generation]: {stage3_status} Recall={stage3_recall:.0%}")

    result = {
        "qid": qid,
        "question": question,
        "type": q_type,
        "stage1_docs": top_so_hieu,
        "stage2_articles": stage2_articles,
        "expected_docs": list(expected_docs),
        "expected_articles": list(all_expected_dieu),
        "cited_articles": list(cited_dieu),
        "stage1_recall": stage1_recall,
        "stage2_recall": stage2_recall,
        "stage3_recall": stage3_recall,
        "citation_requirements": citation_requirements,
        "answer": answer,
        "time_s": round(t_total, 1),
    }
    all_results.append(result)

    # ── Write to output markdown ──
    output_lines.append(f"\n## [EDU_44_Q{qid:02d}] {question}\n")
    output_lines.append(f"**Phân loại:** {q_type}\n")
    output_lines.append(f"**Căn cứ pháp lý kỳ vọng:** {', '.join(expected_citations)}\n")
    output_lines.append(f"**Đáp án tham khảo:** {ref_answer}\n")
    output_lines.append(f"\n### Câu trả lời của LLM\n")
    output_lines.append(f"```\n{answer}\n```\n")
    output_lines.append(f"\n### Đánh giá\n")
    output_lines.append(f"| Tiêu chí | Kết quả |\n|----------|---------|")
    output_lines.append(f"| Stage 1 (VB chọn) | {', '.join(top_so_hieu) if top_so_hieu else 'EMPTY'} |")
    output_lines.append(f"| Stage 1 (Retrieval Recall) | {stage1_status} {stage1_recall:.0%} (tìm thấy: {list(stage1_found)} / kỳ vọng: {list(expected_docs)}) |")
    output_lines.append(f"| Stage 2 (Điều chọn) | {', '.join(stage2_articles) if stage2_articles else 'EMPTY'} |")
    output_lines.append(f"| Stage 2 (Selection Recall) | {stage2_status} {stage2_recall:.0%} (thiếu: {stage2_missing_requirements}) |")
    output_lines.append(f"| Điều khoản kỳ vọng | Điều {all_expected_dieu} |")
    output_lines.append(f"| Điều khoản trích dẫn | Điều {cited_dieu} |")
    output_lines.append(f"| Stage 3 (Generation Recall) | {stage3_status} {stage3_recall:.0%} (thiếu: {stage3_missing_requirements}) |")
    output_lines.append(f"| Thời gian | {t_total:.1f}s |")
    output_lines.append("")

    # Live-write intermediate results
    out_path = os.path.join(PROJECT_ROOT, "outputs/full_pipeline_eval_44.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
        
    json_path = os.path.join(PROJECT_ROOT, "outputs/full_pipeline_eval_44.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "perfect_citation": sum(1 for r in all_results if r["stage3_recall"] == 1.0),
                "avg_citation_recall": sum(r["stage3_recall"] for r in all_results) / len(all_results),
                "total_completed": len(all_results),
                "total_all": 44
            },
            "details": all_results
        }, f, ensure_ascii=False, indent=2)

# ── Summary ──
total = len(all_results)
perfect_s1 = sum(1 for r in all_results if r["stage1_recall"] == 1.0)
avg_s1 = sum(r["stage1_recall"] for r in all_results) / total

perfect_s2 = sum(1 for r in all_results if r["stage2_recall"] == 1.0)
avg_s2 = sum(r["stage2_recall"] for r in all_results) / total

perfect_s3 = sum(1 for r in all_results if r["stage3_recall"] == 1.0)
avg_s3 = sum(r["stage3_recall"] for r in all_results) / total
avg_time = sum(r["time_s"] for r in all_results) / total

summary = f"""
================================================================================
📊 TỔNG KẾT ĐÁNH GIÁ 3 STAGE PIPELINE (44 CÂU HỎI)
================================================================================

 [Stage 1] Retrieval (Truy vết văn bản kỳ vọng):
   Perfect (100%): {perfect_s1}/{total} ({perfect_s1/total:.1%})
   Average Recall: {avg_s1:.1%}

 [Stage 2] TOC Selection (Lựa chọn Điều khoản bắt buộc):
   Perfect (100%): {perfect_s2}/{total} ({perfect_s2/total:.1%})
   Average Recall: {avg_s2:.1%}

 [Stage 3] Generation (Sinh câu trả lời & Trích dẫn cuối cùng):
   Perfect (100%): {perfect_s3}/{total} ({perfect_s3/total:.1%})
   Average Recall: {avg_s3:.1%}

 Performance:
   Avg time/query: {avg_time:.1f}s
   Total time:     {sum(r['time_s'] for r in all_results):.0f}s

📋 Chi tiết các câu chưa đạt Stage 3 Citation 100%:"""

for r in all_results:
    if r["stage3_recall"] < 1.0:
        stage3_missing = [req for req in r["citation_requirements"] if not any(d in r["cited_articles"] for d in req)]
        summary += f"\n   [EDU_44_Q{r['qid']:02d}] Recall={r['stage3_recall']:.0%} | Thiếu: Điều {stage3_missing} | Đã trích dẫn: Điều {sorted(r['cited_articles'])}"

print(summary)

output_lines.append(f"\n{'=' * 80}")
output_lines.append(f"## 📊 TỔNG KẾT ĐÁNH GIÁ CÁC STAGES\n")
output_lines.append(f"| Stage / Chỉ số | Perfect Rate (100% Recall) | Average Recall |")
output_lines.append(f"|---|---|---|")
output_lines.append(f"| **Stage 1 (Retrieval)** | {perfect_s1}/{total} ({perfect_s1/total:.1%}) | {avg_s1:.1%} |")
output_lines.append(f"| **Stage 2 (TOC Selection)** | {perfect_s2}/{total} ({perfect_s2/total:.1%}) | {avg_s2:.1%} |")
output_lines.append(f"| **Stage 3 (LLM Generation)** | {perfect_s3}/{total} ({perfect_s3/total:.1%}) | {avg_s3:.1%} |")
output_lines.append(f"\n**Avg Time/Query:** {avg_time:.1f}s  ")
output_lines.append(f"**Total Execution Time:** {sum(r['time_s'] for r in all_results):.0f}s  ")

# Save
out_path = os.path.join(PROJECT_ROOT, "outputs/full_pipeline_eval_44.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))
print(f"\n💾 Báo cáo chi tiết đã lưu tại: {out_path}")

# Also save JSON
json_path = os.path.join(PROJECT_ROOT, "outputs/full_pipeline_eval_44.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "summary": {
            "stage1": {"perfect": perfect_s1, "average": avg_s1},
            "stage2": {"perfect": perfect_s2, "average": avg_s2},
            "stage3": {"perfect": perfect_s3, "average": avg_s3},
            "total": total,
            "avg_time": avg_time,
        },
        "details": all_results
    }, f, ensure_ascii=False, indent=2)
print(f"💾 JSON: {json_path}")
