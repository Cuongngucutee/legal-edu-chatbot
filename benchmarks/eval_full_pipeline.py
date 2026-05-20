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
load_dotenv()

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

# ── Helpers ──
def bare_sh(text):
    m = re.search(r'(\d+[\/\-]\d{4}[\/\-]?[\w\-]*)', str(text))
    return m.group(1) if m else text

def resolve_sh(raw_sh, registry):
    if raw_sh in registry: return raw_sh
    for v in [raw_sh.lower(), raw_sh.replace('-','/'), raw_sh.replace('/','-')]:
        if v in registry: return v
    for k in registry:
        if len(k) > 3 and len(raw_sh) > 3:
            if k == raw_sh or (raw_sh.startswith(k.split('/')[0]) and raw_sh.endswith(k.split('/')[-1]) and abs(len(k)-len(raw_sh)) < 5):
                return k
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

def parse_320b(response_text, uniq_docs):
    nids, arts = set(), []
    m = re.search(r'\[.*\]', response_text, re.DOTALL)
    if not m: return nids, arts
    try: items = json.loads(m.group(0))
    except: return nids, arts
    for item in items:
        raw = bare_sh(item.get("so_hieu",""))
        dieu = item.get("dieu")
        if not raw or dieu is None: continue
        rk = resolve_sh(raw, index.doc_registry)
        for nid in index.get_doc_node_ids(rk):
            if nid not in index.graph.nodes: continue
            nd = index.graph.nodes[nid]
            name = nd.get("name","")
            if not name:
                ft = nd.get("full_text","") or nd.get("search_text","")
                if ft: name = ft.split("\n")[0].strip()
            dm = re.search(r'Điều\s+(\d+)', name)
            if dm and int(dm.group(1)) == int(dieu):
                nids.add(nid)
                arts.append(f"Điều {dieu} ({uniq_docs.get(raw, raw)})")
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

    top_so_hieu = []
    if unique_docs:
        top_cands = ranked_docs[:8]
        doc_options = "\n".join([f"- {sh} ({info['ten']}) [chunks={info['count']}]" for sh, info in top_cands])
        ctx_text = "\n".join([f"[{d.get('metadata',{}).get('ten_van_ban','')}] Điều {d.get('metadata',{}).get('so_dieu','')}: {(d.get('content','') or '')[:150]}..." for d in docs[:20]])
        fp = f'Dựa vào câu hỏi: "{question}"\nCác trích đoạn:\n{ctx_text}\n\nDanh sách văn bản:\n{doc_options}\n\nChọn TỐI ĐA 2 số hiệu văn bản liên quan nhất.\nChỉ xuất mảng JSON. Ví dụ: ["02/2021/TT-BGDĐT"]'
        try:
            import ast
            fr = llm_7b.generate(fp, temperature=0.1)
            match = re.search(r'\[(.*?)\]', fr)
            if match:
                parsed = ast.literal_eval(f"[{match.group(1)}]")
                for p in parsed:
                    bp = bare_sh(p)
                    if bp in unique_docs: top_so_hieu.append(bp)
                    else:
                        for k in unique_docs:
                            if bp == k or (len(bp) > 5 and len(k) > 5 and (bp in k or k in bp)):
                                top_so_hieu.append(k); break
            if not top_so_hieu: top_so_hieu = [sh for sh, _ in ranked_docs[:2]]
        except: top_so_hieu = [sh for sh, _ in ranked_docs[:2]]

    t_s1 = time.time() - t0
    print(f"   [Stage 1] 7B: {top_so_hieu} ({t_s1:.1f}s)")

    # ── Stage 2: 320B TOC scan (with retry) ──
    toc_parts = []
    for sh in top_so_hieu:
        toc = index.build_toc(sh)
        if toc: toc_parts.append(toc)

    final_node_ids, stage2_articles = set(), []
    if toc_parts:
        toc_text = "\n\n".join(toc_parts)
        p1 = f'Câu hỏi: "{question}"\n\nMục lục các văn bản:\n{toc_text}\n\nNhiệm vụ: Chọn các Điều khoản chứa thông tin để trả lời câu hỏi.\nTrả về JSON: [{{"so_hieu": "...", "dieu": <số>}}]\nChỉ xuất mảng JSON, không giải thích.'
        try:
            r1 = llm_320b.generate(p1, temperature=0.1)
            final_node_ids, stage2_articles = parse_320b(r1, unique_docs)
        except: pass

        if len(final_node_ids) < 2:
            p2 = f'Câu hỏi: "{question}"\n\nMục lục các văn bản:\n{toc_text}\n\nNhiệm vụ: Liệt kê TẤT CẢ các Điều khoản có thể liên quan đến câu hỏi, kể cả gián tiếp.\nChọn ÍT NHẤT 2 điều khoản.\nTrả về JSON: [{{"so_hieu": "...", "dieu": <số>}}]\nChỉ xuất mảng JSON, không giải thích.'
            try:
                r2 = llm_320b.generate(p2, temperature=0.2)
                rn, ra = parse_320b(r2, unique_docs)
                if len(rn) > len(final_node_ids): final_node_ids, stage2_articles = rn, ra
            except: pass

        if not final_node_ids and len(ranked_docs) > 2:
            extras = [sh for sh, _ in ranked_docs[2:5] if sh not in top_so_hieu]
            et = []
            for sh in extras:
                toc = index.build_toc(sh)
                if toc: et.append(toc)
            if et:
                p3 = f'Câu hỏi: "{question}"\n\nMục lục các văn bản:\n{"chr(10)*2".join(toc_parts + et)}\n\nNhiệm vụ: Chọn các Điều khoản chứa thông tin để trả lời câu hỏi.\nChọn ÍT NHẤT 2 điều khoản.\nTrả về JSON: [{{"so_hieu": "...", "dieu": <số>}}]'
                try:
                    r3 = llm_320b.generate(p3, temperature=0.2)
                    en, ea = parse_320b(r3, unique_docs)
                    if en: final_node_ids, stage2_articles = en, ea
                except: pass

    t_s2 = time.time() - t0
    print(f"   [Stage 2] 320B: {stage2_articles if stage2_articles else 'EMPTY'} ({t_s2 - t_s1:.1f}s)")

    # ── Stage 3: Build context + Generate ──
    if final_node_ids:
        final_docs = []
        for nid in final_node_ids:
            d = retriever._node_to_doc(nid)
            if d: final_docs.append(d)
        if final_docs: docs = final_docs

    context = build_context(docs[:10])
    prompt = GENERATION_PROMPT.format(query=question, context=context)
    answer = llm_7b.generate(prompt, system_prompt=GENERATION_SYSTEM_PROMPT, temperature=0.1)

    t_total = time.time() - t0
    print(f"   [Generate] ({t_total - t_s2:.1f}s) → {answer[:120]}...")

    # ── Evaluate ──
    # Check if answer mentions the expected articles
    expected_dieu = set()
    for art in expected_articles:
        m = re.search(r'(\d+)', art)
        if m: expected_dieu.add(int(m.group(1)))

    cited_dieu = set()
    for m in re.finditer(r'[Đđ]iều\s+(\d+)', answer):
        cited_dieu.add(int(m.group(1)))

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
