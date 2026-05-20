"""
Agentic RAG Retrieval Evaluator
Đánh giá pipeline: Retrieval → 7B filter → 320B catalog scan → node matching
Chỉ đánh giá retrieval (Stage 1 + Stage 2), KHÔNG đánh giá generation.
"""
import json
import re
import time
import sys
import os

# Setup paths — benchmarks/ is a subfolder
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# ── Load benchmark ──────────────────────────────────────────
with open(os.path.join(os.path.dirname(__file__), "education_benchmark_v2.json"), "r", encoding="utf-8") as f:
    benchmark = json.load(f)

questions = benchmark["questions"]

# ── Init pipeline components ────────────────────────────────
print("🔧 Đang khởi tạo pipeline...")
from app.rag.pipeline import LawEduPipeline
from app.query.query_expander import EducationQueryExpander
from dotenv import load_dotenv
load_dotenv()

# Init BookIndex + Retriever
from app.index.book_index import BookIndex
from app.rag.hybrid_search import BookRAGRetriever
from app.llm.client import LLMClient

data_dir = os.getenv("DATA_DIR", "data/Nghi dinh")
kg_path = os.getenv("KG_PATH", "outputs/knowledge_graph/legal_kg.graphml")

index = BookIndex(data_dir=data_dir, kg_path=kg_path)
index.load_index()

retriever = BookRAGRetriever(index)

llm_7b = LLMClient(
    api_base=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key="ollama",
    model=os.getenv("GENERATOR_MODEL", "qwen2.5:7b"),
)

llm_320b = LLMClient(
    api_base=os.getenv("LLM_API_BASE"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL", "glm-4.7"),
)

expander = EducationQueryExpander()

# ── Helper: extract bare so_hieu ────────────────────────────
def bare_sh(text):
    m = re.search(r'(\d+[\/-]\d{4}[\/-]?[\w\-]*)', str(text))
    return m.group(1) if m else text

def resolve_sh(raw_sh, registry):
    """Resolve so_hieu to doc_registry key. Exact match first."""
    if raw_sh in registry:
        return raw_sh
    for variant in [raw_sh.lower(), raw_sh.replace('-', '/'), raw_sh.replace('/', '-')]:
        if variant in registry:
            return variant
    for k in registry:
        if len(k) > 3 and len(raw_sh) > 3:
            if k == raw_sh or (raw_sh.startswith(k.split('/')[0]) and raw_sh.endswith(k.split('/')[-1]) and abs(len(k)-len(raw_sh)) < 5):
                return k
    return raw_sh


# ── Run evaluation ──────────────────────────────────────────
results = []
print(f"\n📋 Đánh giá {len(questions)} câu hỏi...\n")
print("=" * 80)

for q in questions:
    qid = q["id"]
    question = q["question"]
    expected_docs = q["ground_truth"]["relevant_docs"]
    expected_articles = q["ground_truth"]["relevant_articles"]
    legal_basis = q["legal_basis"]
    
    print(f"\n📌 [{qid}] {question[:70]}...")
    print(f"   Đáp án: {legal_basis}")
    
    t0 = time.time()
    
    # ── Step 1: Wide Retrieval + RRF Fusion ────────────────────
    import math
    
    def rrf_merge(ranked_lists, k=60):
        scores = {}
        chunk_map = {}
        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked):
                cid = doc.get("chunk_id", "")
                if not cid: continue
                scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
                if cid not in chunk_map:
                    chunk_map[cid] = doc
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [chunk_map[cid] for cid in sorted_ids], scores

    main_docs = retriever.retrieve_as_docs(question, top_k=20)
    all_ranked = [main_docs]
    expanded = expander.expand(question)
    for eq in expanded[:3]:
        eq_docs = retriever.retrieve_as_docs(eq, top_k=8)
        if eq_docs:
            all_ranked.append(eq_docs)
    docs, rrf_scores = rrf_merge(all_ranked)
    
    t_retrieve = time.time() - t0
    
    # ── Step 2: Doc-Level Vote Aggregation + 7B Selection ────
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

    unique_docs = {}
    ranked_docs = []
    if doc_agg:
        for bare, info in doc_agg.items():
            avg_rrf = info["total_rrf"] / info["count"] if info["count"] else 0
            info["rank_score"] = info["max_rrf"] + 0.3 * math.log(1 + info["count"]) * avg_rrf
            unique_docs[bare] = info["ten"]
        ranked_docs = sorted(doc_agg.items(), key=lambda x: x[1]["rank_score"], reverse=True)
    
    top_so_hieu = []
    if unique_docs:
        top_candidates = ranked_docs[:8]
        doc_options = "\n".join([
            f"- {sh} ({info['ten']}) [chunks={info['count']}, score={info['rank_score']:.3f}]"
            for sh, info in top_candidates
        ])
        context_text = "\n".join([
            f"[{d.get('metadata',{}).get('ten_van_ban','')}] Điều {d.get('metadata',{}).get('so_dieu','')}: {(d.get('content','') or d.get('text',''))[:150]}..."
            for d in docs[:20]
        ])
        filter_prompt = f"""Dựa vào câu hỏi: "{question}"
Các trích đoạn (đã xếp hạng theo độ liên quan):
{context_text}

Danh sách văn bản (xếp hạng theo mức liên quan):
{doc_options}

Chọn TỐI ĐA 2 số hiệu văn bản liên quan nhất để trả lời câu hỏi.
Chỉ xuất mảng JSON. Ví dụ: ["02/2021/TT-BGDĐT", "13/2024/TT-BGDĐT"]"""
        
        try:
            import ast
            filter_res = llm_7b.generate(filter_prompt, temperature=0.1)
            match = re.search(r'\[(.*?)\]', filter_res)
            if match:
                parsed = ast.literal_eval(f"[{match.group(1)}]")
                for p in parsed:
                    bare_p = bare_sh(p)
                    if bare_p in unique_docs:
                        top_so_hieu.append(bare_p)
                    else:
                        for k in unique_docs:
                            if bare_p == k or (len(bare_p) > 5 and len(k) > 5 and (bare_p in k or k in bare_p)):
                                top_so_hieu.append(k)
                                break
            if not top_so_hieu:
                top_so_hieu = [sh for sh, _ in ranked_docs[:2]]
        except:
            top_so_hieu = [sh for sh, _ in ranked_docs[:2]]
    
    t_stage1 = time.time() - t0
    print(f"   [Stage 1] 7B chọn: {top_so_hieu} ({t_stage1 - t_retrieve:.1f}s)")
    
    # ── Step 3: 320B phân tích mục lục (with retry) ──────────
    toc_parts = []
    for sh in top_so_hieu:
        toc = index.build_toc(sh)
        if toc:
            toc_parts.append(toc)
    
    stage2_articles = []
    final_node_ids = set()

    def parse_320b(response_text):
        """Parse 320B response into (node_ids, articles)."""
        nids = set()
        arts = []
        m = re.search(r'\[.*\]', response_text, re.DOTALL)
        if not m:
            return nids, arts
        try:
            items = json.loads(m.group(0))
        except json.JSONDecodeError:
            return nids, arts
        for item in items:
            raw = bare_sh(item.get("so_hieu", ""))
            dieu = item.get("dieu")
            if not raw or dieu is None:
                continue
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
                if dm and int(dm.group(1)) == int(dieu):
                    nids.add(nid)
                    arts.append(f"Điều {dieu} ({unique_docs.get(raw, raw)})")
        return nids, arts
    
    if toc_parts:
        toc_text = "\n\n".join(toc_parts)

        # Attempt 1: Standard prompt
        prompt_320b = f"""Câu hỏi: "{question}"

Mục lục các văn bản:
{toc_text}

Nhiệm vụ: Chọn các Điều khoản chứa thông tin để trả lời câu hỏi.
Trả về JSON: [{{"so_hieu": "...", "dieu": <số>}}]
Chỉ xuất mảng JSON, không giải thích."""
        
        try:
            res_320b = llm_320b.generate(prompt_320b, temperature=0.1)
            final_node_ids, stage2_articles = parse_320b(res_320b)
        except Exception as e:
            print(f"   ⚠️ 320B attempt 1 error: {e}")

        # Attempt 2: Softer prompt if <2 articles
        if len(final_node_ids) < 2:
            retry_prompt = f"""Câu hỏi: "{question}"

Mục lục các văn bản:
{toc_text}

Nhiệm vụ: Liệt kê TẤT CẢ các Điều khoản có thể liên quan đến câu hỏi, kể cả các điều liên quan gián tiếp (ví dụ: điều định nghĩa, điều quy định phạm vi áp dụng, điều quy định đối tượng).
Chọn ÍT NHẤT 2 điều khoản.
Trả về JSON: [{{"so_hieu": "...", "dieu": <số>}}]
Chỉ xuất mảng JSON, không giải thích."""
            try:
                res_retry = llm_320b.generate(retry_prompt, temperature=0.2)
                retry_nodes, retry_arts = parse_320b(res_retry)
                if len(retry_nodes) > len(final_node_ids):
                    final_node_ids = retry_nodes
                    stage2_articles = retry_arts
            except:
                pass

        # Attempt 3: Expand to next-ranked docs if still empty
        if not final_node_ids and len(ranked_docs) > 2:
            extra_shs = [sh for sh, _ in ranked_docs[2:5] if sh not in top_so_hieu]
            extra_tocs = []
            for sh in extra_shs:
                toc = index.build_toc(sh)
                if toc:
                    extra_tocs.append(toc)
            if extra_tocs:
                expanded_toc = "\n\n".join(toc_parts + extra_tocs)
                expand_prompt = f"""Câu hỏi: "{question}"

Mục lục các văn bản:
{expanded_toc}

Nhiệm vụ: Chọn các Điều khoản chứa thông tin để trả lời câu hỏi.
Chọn ÍT NHẤT 2 điều khoản.
Trả về JSON: [{{"so_hieu": "...", "dieu": <số>}}]
Chỉ xuất mảng JSON, không giải thích."""
                try:
                    res_expand = llm_320b.generate(expand_prompt, temperature=0.2)
                    expand_nodes, expand_arts = parse_320b(res_expand)
                    if expand_nodes:
                        final_node_ids = expand_nodes
                        stage2_articles = expand_arts
                except:
                    pass
    
    t_stage2 = time.time() - t0
    print(f"   [Stage 2] 320B chọn: {stage2_articles if stage2_articles else '❌ EMPTY'} ({t_stage2 - t_stage1:.1f}s)")
    
    # ── Evaluate retrieval ──────────────────────────────────
    # Check Stage 1: did we get the right document?
    stage1_docs_hit = False
    for exp_doc in expected_docs:
        exp_bare = bare_sh(exp_doc.replace("_", "/"))
        for sh in top_so_hieu:
            if exp_bare in sh or sh in exp_bare:
                stage1_docs_hit = True
                break
    
    # Check Stage 2: did we get the right articles?
    retrieved_dieu_nums = set()
    for art in stage2_articles:
        m = re.search(r'Điều\s+(\d+)', art)
        if m:
            retrieved_dieu_nums.add(int(m.group(1)))
    
    expected_dieu_nums = set()
    for art in expected_articles:
        m = re.search(r'(\d+)', art)
        if m:
            expected_dieu_nums.add(int(m.group(1)))
    
    articles_found = expected_dieu_nums & retrieved_dieu_nums
    articles_missing = expected_dieu_nums - retrieved_dieu_nums
    articles_extra = retrieved_dieu_nums - expected_dieu_nums
    
    recall = len(articles_found) / len(expected_dieu_nums) if expected_dieu_nums else 0
    precision = len(articles_found) / len(retrieved_dieu_nums) if retrieved_dieu_nums else 0
    
    status = "✅" if recall == 1.0 else ("⚠️" if recall > 0 else "❌")
    
    print(f"   {status} Stage1 Doc: {'✅' if stage1_docs_hit else '❌'} | "
          f"Stage2 Recall: {recall:.0%} (found {articles_found}, missing {articles_missing}, extra {articles_extra})")
    
    results.append({
        "qid": qid,
        "question": question[:60],
        "legal_basis": legal_basis,
        "stage1_docs": top_so_hieu,
        "stage1_doc_hit": stage1_docs_hit,
        "stage2_articles": stage2_articles,
        "expected_articles": list(expected_dieu_nums),
        "found_articles": list(articles_found),
        "missing_articles": list(articles_missing),
        "recall": recall,
        "precision": precision,
        "time_s": round(t_stage2, 1),
    })

# ── Summary ─────────────────────────────────────────────────
print("\n" + "=" * 80)
print("📊 TỔNG KẾT ĐÁNH GIÁ RETRIEVAL")
print("=" * 80)

total = len(results)
stage1_hits = sum(1 for r in results if r["stage1_doc_hit"])
perfect_recall = sum(1 for r in results if r["recall"] == 1.0)
partial_recall = sum(1 for r in results if 0 < r["recall"] < 1.0)
zero_recall = sum(1 for r in results if r["recall"] == 0)
avg_recall = sum(r["recall"] for r in results) / total if total else 0

print(f"\n Stage 1 (7B Doc Selection):")
print(f"   Chọn đúng văn bản: {stage1_hits}/{total} ({stage1_hits/total:.0%})")

print(f"\n Stage 2 (320B Article Selection):")
print(f"   Perfect recall (100%): {perfect_recall}/{total}")
print(f"   Partial recall:        {partial_recall}/{total}")
print(f"   Zero recall (0%):      {zero_recall}/{total}")
print(f"   Average recall:        {avg_recall:.1%}")

print(f"\n📋 Chi tiết các câu bị thiếu:")
for r in results:
    if r["recall"] < 1.0:
        print(f"   [{r['qid']}] Expected: Điều {r['expected_articles']} | Got: {r['stage2_articles'] or 'EMPTY'} | Missing: Điều {r['missing_articles']}")

# Save results
with open(os.path.join(PROJECT_ROOT, "outputs/agentic_retrieval_eval.json"), "w", encoding="utf-8") as f:
    json.dump({"summary": {"stage1_accuracy": stage1_hits/total, "avg_recall": avg_recall, "perfect_recall": perfect_recall, "total": total}, "details": results}, f, ensure_ascii=False, indent=2)

print(f"\n💾 Kết quả chi tiết: outputs/agentic_retrieval_eval.json")
