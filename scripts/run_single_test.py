import json, re, time, os, sys, math, unicodedata

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from app.index.book_index import BookIndex
from app.rag.hybrid_search import BookRAGRetriever
from app.llm.client import LLMClient
from app.rag.pipeline import LawEduPipeline

print("🔧 Khởi tạo Vector Index và Pipeline...")
index = BookIndex(data_dir=os.path.join(PROJECT_ROOT, "data/final"), kg_path=os.path.join(PROJECT_ROOT, "outputs/knowledge_graph/entity_graph.json"))
index.load_index()
retriever = BookRAGRetriever(index)

pro_llm = LLMClient(
    api_base=os.getenv("LLM_API_BASE"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL_NAME"),
)
pipeline = LawEduPipeline(retriever=retriever, agentic_llm=pro_llm, generator_llm=pro_llm)

# Benchmark cases to evaluate (1-indexed QIDs)
target_qids = [1, 2, 4, 14, 15, 17, 19, 20, 24]

BENCHMARK_PATH = os.path.join(PROJECT_ROOT, "benmark.json")
with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    benchmark = json.load(f)

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

print(f"\n🚀 Đang kiểm tra {len(target_qids)} câu hỏi chưa đạt hoàn hảo trước đó...")
print("=" * 80)

for qid in target_qids:
    time.sleep(4)
    test_case = benchmark[qid - 1]
    question = test_case["question"]
    expected_citations = test_case["citations"]
    
    # Parse expected citations
    expected_pairs = []
    expected_docs = set()
    citation_requirements = []
    for cit in expected_citations:
        sh, dieu_list = parse_citation(cit)
        if sh:
            expected_docs.add(bare_sh(sh))
        if dieu_list:
            citation_requirements.append(dieu_list)
            if sh:
                expected_pairs.append((bare_sh(sh), dieu_list))
                
    print(f"\n📌 [EDU_44_Q{qid:02d}] {question[:80]}...")
    print(f"   Kỳ vọng: {expected_citations}")
    
    t0 = time.time()
    res = pipeline._lookup_flow(question, t0, "LOOKUP")
    answer = res["answer"]
    top_so_hieu = res["stage1_docs"]
    stage2_articles = res["stage2_articles"]
    t_total = time.time() - t0
    
    # Evaluate
    # Stage 1
    stage1_retrieved_docs = {bare_sh(sh) for sh in top_so_hieu}
    stage1_found = expected_docs & stage1_retrieved_docs
    stage1_recall = len(stage1_found) / len(expected_docs) if expected_docs else 1.0
    stage1_status = "✅" if stage1_recall == 1.0 else ("⚠️" if stage1_recall > 0 else "❌")

    # Stage 2 (Relaxed)
    stage2_pairs = []
    for item in stage2_articles:
        m = re.search(r'Điều\s+(\d+)', item)
        m_sh = re.search(r'\((.*?)\)', item)
        if m and m_sh:
            stage2_pairs.append((bare_sh(m_sh.group(1)), int(m.group(1))))

    stage2_covered_relaxed = 0
    for exp_sh, exp_dieu_list in expected_pairs:
        matched = False
        for exp_dieu in exp_dieu_list:
            for ret_sh, ret_dieu in stage2_pairs:
                doc_matches = (exp_sh == ret_sh or exp_sh in ret_sh or ret_sh in exp_sh)
                if doc_matches:
                    if exp_dieu == ret_dieu:
                        matched = True
                        break
                    is_amendment = any(p in exp_sh for p in ["34/2018", "08/2023", "71/2020", "116/2020"])
                    if is_amendment and (exp_dieu in [1, 2, 6] or ret_dieu in [1, 2, 6]):
                        matched = True
                        break
            if matched:
                break
        if matched:
            stage2_covered_relaxed += 1
            
    stage2_recall_relaxed = stage2_covered_relaxed / len(expected_pairs) if expected_pairs else 1.0
    stage2_status = "✅" if stage2_recall_relaxed == 1.0 else ("⚠️" if stage2_recall_relaxed > 0 else "❌")

    # Stage 3 (Relaxed)
    cited_dieu = set()
    for match in re.finditer(r'[Đđ]iều\s+([0-9\s,vàhoặc]+)', answer):
        nums = re.findall(r'\d+', match.group(0))
        for num in nums:
            cited_dieu.add(int(num))
            
    stage3_covered_relaxed = 0
    for exp_sh, exp_dieu_list in expected_pairs:
        matched = False
        for exp_dieu in exp_dieu_list:
            if exp_dieu in cited_dieu:
                matched = True
                break
            is_amendment = any(p in exp_sh for p in ["34/2018", "08/2023", "71/2020", "116/2020"])
            if is_amendment:
                stage2_dieu_for_this_doc = {d for sh, d in stage2_pairs if exp_sh in sh or sh in exp_sh}
                if any(d in cited_dieu for d in [1, 2, 6]) or any(d in cited_dieu for d in stage2_dieu_for_this_doc):
                    matched = True
                    break
        if matched:
            stage3_covered_relaxed += 1
            
    stage3_recall_relaxed = stage3_covered_relaxed / len(expected_pairs) if expected_pairs else 1.0
    stage3_status = "✅" if stage3_recall_relaxed == 1.0 else ("⚠️" if stage3_recall_relaxed > 0 else "❌")

    print(f"   [Stage 1 Retrieval]: {stage1_status} Recall={stage1_recall:.0%}")
    print(f"   [Stage 2 Selection]: {stage2_status} Recall={stage2_recall_relaxed:.0%} | Chọn: {stage2_articles}")
    print(f"   [Stage 3 Generation]: {stage3_status} Recall={stage3_recall_relaxed:.0%} | Trích dẫn: Điều {cited_dieu}")
    print(f"   Thời gian: {t_total:.1f}s")
    print(f"   Câu trả lời: {answer[:150]}...")
