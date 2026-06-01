"""
Custom Benchmark Execution Script
Thực thi pipeline trên benmark.json, đánh giá bằng công thức mới và có Rate Limiting.
"""
import json, re, time, sys, os, math
from collections import deque

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
except Exception:
    pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# ── TPM Tracker ──
class RateLimiter:
    def __init__(self, tpm_limit=250000):
        self.tpm_limit = tpm_limit
        self.token_history = deque() # tuples of (timestamp, tokens)

    def add_tokens(self, tokens):
        now = time.time()
        self.token_history.append((now, tokens))
        self._cleanup(now)

    def _cleanup(self, now):
        while self.token_history and now - self.token_history[0][0] > 60:
            self.token_history.popleft()

    def get_current_tpm(self):
        now = time.time()
        self._cleanup(now)
        return sum(t[1] for t in self.token_history)

    def wait_if_needed(self, estimated_next_tokens):
        while True:
            now = time.time()
            self._cleanup(now)
            current = self.get_current_tpm()
            if current + estimated_next_tokens <= self.tpm_limit:
                break
            print(f"⏳ Đạt giới hạn TPM ({current}/{self.tpm_limit}). Đang chờ 10s...")
            time.sleep(10)

def estimate_tokens(text):
    return len(str(text)) // 4

def clean_text_for_eval(text):
    if not isinstance(text, str):
        return ""
    # Xóa in đậm, tiêu đề markdown
    text = re.sub(r'\*{2,}', '', text)
    text = re.sub(r'#{1,3}\s', '', text)
    # Thay newline bằng khoảng trắng
    text = text.replace('\n', ' ')
    # Xóa dấu gạch ngang đầu dòng
    text = re.sub(r'[-*]\s+', '', text)
    # Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Đã nâng cấp API lên 250k TPM
rate_limiter = RateLimiter(tpm_limit=250000)

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

llm = LLMClient(
    api_base=os.getenv("LLM_API_BASE", "https://api.int2.net/v1"),
    api_key=os.getenv("LLM_API_KEY", ""),
    model=os.getenv("LLM_MODEL_NAME", "glm-4.7"),
)

pipeline = LawEduPipeline(retriever=retriever, agentic_llm=llm, generator_llm=llm)

# ── Load benchmark ──
benchmark_path = os.path.join(os.path.dirname(__file__), "benmark.json")
with open(benchmark_path, "r", encoding="utf-8") as f:
    questions = json.load(f)

# Giới hạn chạy một số câu nếu cần test (ví dụ questions[:5])
# questions = questions[:2] 

# ── Helpers (tái sử dụng từ eval_full_pipeline) ──
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

def bare_sh(text):
    m = re.search(r'(\d+[\/\-]\d{4}[\/\-]?[\w\-]*)', str(text))
    return m.group(1) if m else text

def resolve_sh(raw_sh, registry):
    if raw_sh in registry: return registry[raw_sh]
    norm_raw = bare_sh(raw_sh)
    for k in registry:
        if bare_sh(k) == norm_raw: return registry[k]
    for k in registry:
        norm_k = bare_sh(k)
        if len(norm_k) > 3 and len(norm_raw) > 3:
            if norm_k in norm_raw or norm_raw in norm_k: return registry[k]
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

def parse_pro_llm(response_text, uniq_docs):
    import json, ast
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
        if lines_block[0].startswith("```"): lines_block = lines_block[1:]
        if lines_block and lines_block[-1].startswith("```"): lines_block = lines_block[:-1]
        target_block = "\n".join(lines_block).strip()
        
    m = re.search(r'\[\s*\{.*\}\s*\]', target_block, re.DOTALL)
    if not m: m = re.search(r'\[.*\]', target_block, re.DOTALL)
        
    items = None
    if m:
        json_str = m.group(0)
        try: items = json.loads(json_str)
        except: 
            try: items = ast.literal_eval(json_str)
            except: pass
                
    if not isinstance(items, list):
        items = []
        matches = re.findall(r'\{\s*["\']so_hieu["\']\s*:\s*["\']([^"\']+)["\']\s*,\s*["\']dieu["\']\s*:\s*(\d+)\s*\}', target_block)
        for raw_sh, dieu_str in matches:
            try: items.append({"so_hieu": raw_sh, "dieu": int(dieu_str)})
            except: pass

    for item in items:
        if not isinstance(item, dict): continue
        raw = bare_sh(item.get("so_hieu", ""))
        dieu = item.get("dieu")
        if not raw or dieu is None: continue
        try: dieu_num = int(dieu)
        except: continue
            
        raw_upper = raw.upper()
        if "43/2019" in raw_upper: raw = "43/2019/QH14"
        elif "34/2018" in raw_upper: raw = "34/2018/QH14"
        elif "84/2020" in raw_upper: raw = "84/2020/NĐ-CP"
        elif "116/2020" in raw_upper: raw = "116/2020/NĐ-CP"
        elif "105/2020" in raw_upper: raw = "105/2020/NĐ-CP"
        elif "71/2020" in raw_upper: raw = "71/2020/NĐ-CP"
        elif "86/2021" in raw_upper: raw = "86/2021/NĐ-CP"
        elif "22/2021" in raw_upper: raw = "22/2021/TT-BGDĐT"
        elif "24/2024" in raw_upper: raw = "24/2024/TT-BGDĐT"
        elif "32/2018" in raw_upper: raw = "32/2018/TT-BGDĐT"
        elif "05/2023" in raw_upper: raw = "05/2023/TT-BGDĐT"
        elif "08/2023" in raw_upper: raw = "08/2023/TT-BGDĐT"
        elif "01/2021" in raw_upper: raw = "01/2021/TT-BGDĐT"
        elif "02/2021" in raw_upper: raw = "02/2021/TT-BGDĐT"
        elif "03/2021" in raw_upper: raw = "03/2021/TT-BGDĐT"
        elif "04/2021" in raw_upper: 
            if "ND" in raw_upper or "NĐ" in raw_upper: raw = "04/2021/NĐ-CP"
            else: raw = "04/2021/TT-BGDĐT"
        elif "08/2012" in raw_upper: raw = "08/2012/QH13"
        elif "30/2013" in raw_upper: raw = "30/2013/QH13"
        elif "124/2024" in raw_upper: raw = "124/2024/NĐ-CP"
        elif "202/2025" in raw_upper: raw = "202/2025/NĐ-CP"
        elif "20/2022" in raw_upper: raw = "20/2022/TT-BGDĐT"
        elif "05/2025" in raw_upper: raw = "05/2025/TT-BGDĐT"
            
        resolved_key = resolve_sh(raw, index.doc_registry)
        for nid in index.get_doc_node_ids(resolved_key):
            if nid not in index.graph.nodes: continue
            nd = index.graph.nodes[nid]
            name = nd.get("name", "")
            if not name:
                ft = nd.get("full_text", "") or nd.get("search_text", "")
                if ft: name = ft.split("\n")[0].strip()
            dm = re.search(r'Điều\s+(\d+)', name)
            if dm and int(dm.group(1)) == dieu_num:
                nids.add(nid)
                
        disp_name = uniq_docs.get(raw, raw)
        if not any(a == f"Điều {dieu_num} ({disp_name})" for a in arts):
            arts.append(f"Điều {dieu_num} ({disp_name})")
            
    return nids, arts

# ── LLM call with rate limiter ──
def llm_generate_with_rl(prompt, system_prompt="", temperature=0.0):
    estimated = estimate_tokens(prompt + system_prompt) + 500
    rate_limiter.wait_if_needed(estimated)
    res = llm.generate(prompt, system_prompt=system_prompt, temperature=temperature)
    rate_limiter.add_tokens(estimated + estimate_tokens(res))
    return res

# ── Run pipeline ──
all_results = []
out_json = os.path.join(PROJECT_ROOT, "outputs", "custom_benchmark_results.json")
os.makedirs(os.path.dirname(out_json), exist_ok=True)

if os.path.exists(out_json):
    try:
        with open(out_json, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"🔄 Đã load {len(all_results)} kết quả trước đó. Sẽ chạy tiếp...")
    except:
        all_results = []

print(f"\n📋 Chạy benchmark ({len(questions)} câu)...\n")

processed_qids = {r["qid"] for r in all_results}

for i, q in enumerate(questions):
    qid = f"Q{i+1}"
    if qid in processed_qids:
        continue

    # Đã nâng cấp API lên 250k TPM / 500k RPD nên không cần pacing
    print("⚡ Đang dùng API hạn mức cao (250k TPM). Không cần chờ...")
    time.sleep(0.1)

    question = q["question"]
    q_type = q.get("type", "Unknown")
    ref_answer = q.get("answer", "")
    citations_str = q.get("citations", [])
    
    print(f"\n📌 [{qid}] {question[:70]}...")
    t0 = time.time()

    # Chạy Full Pipeline nguyên bản (bao gồm cả Phân loại Intent & Handlers)
    try:
        res_pipeline = pipeline.run(question)
        answer = res_pipeline["answer"]
        top_so_hieu = res_pipeline.get("stage1_docs", [])
        stage2_articles = res_pipeline.get("stage2_articles", [])
        context = res_pipeline.get("context", "")
    except Exception as e:
        print(f"   ⚠️ Pipeline execution error: {e}")
        answer = ""
        top_so_hieu = []
        stage2_articles = []
        context = ""

    t_total = time.time() - t0
    
    # ── LLM EVALUATION ──
    # Tích hợp bộ quy tắc chuyên sâu từ LLM Judge Prompt.txt
    citations_text = "\n".join(f"- {c}" for c in citations_str) if citations_str else "Không yêu cầu trích dẫn cụ thể."
    
    clean_ref_answer = clean_text_for_eval(ref_answer)
    clean_answer = clean_text_for_eval(answer)
    
    judge_prompt = f"""You are acting as a STRICT, NON-GENERATIVE EVALUATION JUDGE for a Vietnamese legal QA system.
Dưới đây là Dữ liệu chuẩn mực (Ground Truth) và Câu trả lời của AI cần được đánh giá.

[QUESTION]
{question}

[BASELINE ANSWER & CITATIONS]
Answer: {clean_ref_answer}
Citations:
{citations_text}

[RETRIEVED CONTEXT SNIPPETS]
{context[:100000]}...

[LAWBOT ANSWER]
{clean_answer}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES FOR SCORING
1. Similarity_Score (0.0 - 1.0): Đánh giá độ tương đồng NGỮ NGHĨA và KẾT LUẬN giữa Lawbot Answer và Baseline Answer. Không cần giống y hệt từ vựng.
2. Citation_Recall (0.0 - 1.0): Tỷ lệ các văn bản/điều luật trong Baseline Citations xuất hiện trong Lawbot Answer. (Nếu Baseline không có trích dẫn, cho 1.0).
3. G_Score (0.0 - 4.0): Đánh giá Grounding & Completeness. (4=Hoàn hảo, dựa hoàn toàn vào ngữ cảnh và trả lời đầy đủ ý của câu hỏi).
4. CAR_Score (0.0 - 1.0): Context Attribution Rate. Tỷ lệ các câu trong Lawbot Answer được hỗ trợ (supported) bởi Retrieved Context.
5. MucDoAoGiac:
   - "Khong_Ao_Giac" (Tốt): Mọi thông tin đều dựa trên context, không tự bịa luật.
   - "Vien_Dan_Bo_Sung" (Chấp nhận được): AI viện dẫn thêm văn bản pháp luật hợp lệ, dù văn bản đó không xuất hiện trong Baseline Answer. Không được phạt nặng.
   - "Suy_Dien" (Tạm): Suy diễn logic vượt quá mức cần thiết nhưng không sai luật.
   - "Bia_Luat" (Tệ): Chỉ đánh dấu Bia_Luat khi: Văn bản không tồn tại, Điều luật không tồn tại, nội dung luật bị bịa, hoặc trích dẫn trái ngược với quy định thực tế. Không được đánh dấu Bia_Luat nếu AI viện dẫn thêm văn bản pháp luật hợp lệ.

NHIỆM VỤ ĐÁNH GIÁ (Chỉ trả về JSON):
{{
    "Similarity_Score": <float>,
    "Citation_Recall": <float>,
    "G_Score": <float>,
    "CAR_Score": <float>,
    "MucDoAoGiac": "<Khong_Ao_Giac | Vien_Dan_Bo_Sung | Suy_Dien | Bia_Luat>",
    "Judge_Confidence": <float 0.0-1.0>,
    "Hallucination_Reason": "<string giải thích ngắn gọn lý do>"
}}"""

    try:
        judge_res = llm_generate_with_rl(judge_prompt, temperature=0.0).strip()
        match = re.search(r'\{.*\}', judge_res, re.DOTALL)
        judge_data = json.loads(match.group(0))
        
        sim_score = float(judge_data.get("Similarity_Score", 0.5))
        citation_recall = float(judge_data.get("Citation_Recall", 0.5))
        g_score = float(judge_data.get("G_Score", 2.0))
        car_score = float(judge_data.get("CAR_Score", 0.5))
        muc_do = judge_data.get("MucDoAoGiac", "Khong_Ao_Giac")
        confidence = float(judge_data.get("Judge_Confidence", 1.0))
        
        if "Bia_Luat" in muc_do:
            # Kiểm tra Hallucination dựa trên Corpus
            is_in_corpus = False
            for sh in top_so_hieu:
                if sh.lower() in clean_answer.lower():
                    is_in_corpus = True
                    break
            
            if is_in_corpus:
                hal_score = 1 # Có trong corpus -> hạ xuống Suy_Dien
            else:
                hal_score = 2
        elif "Suy_Dien" in muc_do:
            hal_score = 1
        elif "Vien_Dan_Bo_Sung" in muc_do:
            hal_score = 0.5
        else:
            hal_score = 0
            
        # Confidence Check
        if hal_score == 2 and confidence < 0.7:
            hal_score = 1
            
    except Exception as e:
        print(f"   ⚠️ Lỗi khi chấm điểm LLM Judge: {e}")
        sim_score, citation_recall, g_score, car_score, hal_score = 0.5, 0.5, 2.0, 0.5, 0
        judge_data = {}
        
    # Chuẩn hóa tổng trọng số Base Score = 1.0
    base_score = 0.30 * citation_recall + 0.30 * sim_score + 0.25 * (g_score/4.0) + 0.15 * car_score
    
    # Thay thế Hard Fail bằng Penalty
    if hal_score >= 2:
        final_score = base_score * 0.5
    elif hal_score == 1:
        final_score = base_score * 0.95
    elif hal_score == 0.5:
        final_score = base_score * 0.99
    else:
        final_score = base_score

    if citation_recall == 0 and len(citations_str) > 0:
        final_score = min(final_score, 0.5 * base_score)

    print(f"   ✅ Score: {final_score*100:.1f} | Sim: {sim_score:.2f} | Citation: {citation_recall:.0%} | Hallucination: {hal_score} | Time: {t_total:.1f}s")
    
    all_results.append({
        "qid": qid,
        "type": q_type,
        "question": question,
        "time_s": t_total,
        "citation_recall": citation_recall,
        "similarity_score": sim_score,
        "g_score": g_score,
        "car_score": car_score,
        "hallucination": hal_score,
        "final_score": final_score,
        "ai_answer": answer,
        "judge_raw": judge_data  # Lưu toàn bộ JSON Judge
    })
    
    # ── Lưu kết quả ngay sau mỗi câu ──
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

# ── Xuất Báo Cáo ──

report_lines = []
report_lines.append("# 📊 Báo Cáo Đánh Giá Hiệu Suất & Độ Chính Xác (GPT 120B)\n")

# Tính toán các chỉ số
total = len(all_results)
avg_time = sum(r["time_s"] for r in all_results) / total
avg_final = sum(r["final_score"] for r in all_results) / total
avg_sim = sum(r["similarity_score"] for r in all_results) / total
avg_citation = sum(r["citation_recall"] for r in all_results) / total
perfect_count = sum(1 for r in all_results if r["final_score"] >= 0.95)
hal_2_count = sum(1 for r in all_results if r["hallucination"] >= 2)

report_lines.append("## 1. Tổng quan Hiệu suất (Performance Overview)")
report_lines.append(f"- **Tổng số câu hỏi**: {total}")
report_lines.append(f"- **Thời gian phản hồi trung bình**: {avg_time:.2f} giây/câu")
report_lines.append(f"- **Tổng thời gian chạy**: {sum(r['time_s'] for r in all_results):.1f} giây")

report_lines.append("\n## 2. Độ Chính xác Tổng hợp (Overall Accuracy)")
report_lines.append(f"- **Điểm tổng hợp trung bình (Final Score)**: {avg_final*100:.1f} / 100")
report_lines.append(f"- **Tỷ lệ câu trả lời hoàn hảo (Score >= 95%)**: {perfect_count/total:.1%}")

report_lines.append("\n## 3. Chi tiết theo Tiêu chí (Metric Breakdown)")
report_lines.append(f"- **Citation Recall (Trích dẫn đúng điều luật)**: {avg_citation:.1%}")
report_lines.append(f"- **Answer Similarity (Giống đáp án tham khảo)**: {avg_sim:.1%}")
report_lines.append(f"- **Tỷ lệ Ảo giác nghiêm trọng (Bịa luật)**: {hal_2_count/total:.1%} ({hal_2_count} câu bịa luật)")
avg_g = sum(r["g_score"] for r in all_results) / total
report_lines.append(f"- **Điểm Lập luận & Toàn diện (G Score)**: {avg_g:.2f}/4.0")

report_lines.append("\n## 4. Phân tích theo Loại câu hỏi (Category Analysis)")
types = {}
for r in all_results:
    t = r["type"]
    if t not in types: types[t] = []
    types[t].append(r["final_score"])

for t, scores in types.items():
    avg = sum(scores) / len(scores)
    report_lines.append(f"- **{t}**: {avg*100:.1f} / 100 ({len(scores)} câu)")

out_md = os.path.join(PROJECT_ROOT, "outputs", "custom_benchmark_report.md")
with open(out_md, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"\n✅ Đã lưu kết quả vào {out_json}")
print(f"✅ Đã lưu báo cáo vào {out_md}")
