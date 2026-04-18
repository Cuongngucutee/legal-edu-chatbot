"""
Benchmark v3 — Ground Truth Retrieval Evaluation
===================================================
25 câu hỏi × 7 intent types.
Mỗi câu hỏi gắn với ground truth node IDs cụ thể (chunk IDs đã có trong data).

SCORING:
  - Retrieval: 0 hoặc 1. Đạt 1 khi context chứa ĐỦ các chunk cần thiết.
  - Intent Classification: 0 hoặc 1. Đạt 1 khi phân loại đúng type.

Chạy:
    python backend/eval/benchmark.py           # Qwen 3B
    python backend/eval/benchmark.py --regex    # Regex-only (nhanh)
"""
import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import time
import sys
import json
import glob
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "core"))


# ══════════════════════════════════════════════════════════════════════════════
# 25 BENCHMARK QUERIES — Ground Truth from actual data/final/*.json
# ══════════════════════════════════════════════════════════════════════════════

BENCHMARKS = [
    # ═══════════════════════════════════════════════════════════════════════
    # SINGLE_LOOKUP (5 câu) — Tra cứu 1 điều khoản cụ thể
    # ═══════════════════════════════════════════════════════════════════════
    {
        "id": "Q01",
        "name": "Hành vi bị cấm — Điều 22 Luật GD 2019",
        "intent_type": "single_lookup",
        "query": "Điều 22 Luật Giáo dục 2019 quy định những hành vi nào bị nghiêm cấm trong cơ sở giáo dục?",
        "required_chunks": ["Luật43_D22"],
        "description": "Cần chunk Điều 22 Luật 43/2019 liệt kê 6 hành vi bị cấm",
    },
    {
        "id": "Q02",
        "name": "Tiền lương nhà giáo — Điều 76 Luật GD 2019",
        "intent_type": "single_lookup",
        "query": "Điều 76 Luật Giáo dục 2019 quy định gì về tiền lương của nhà giáo?",
        "required_chunks": ["Luật43_D76"],
        "description": "Cần chunk Điều 76 về lương nhà giáo được ưu tiên phụ cấp",
    },
    {
        "id": "Q03",
        "name": "Nguyên tắc học phí — Điều 7 NĐ 238",
        "intent_type": "single_lookup",
        "query": "Điều 7 Nghị định 238/2025/NĐ-CP quy định gì về nguyên tắc xác định học phí?",
        "required_chunks": ["238_2025_NĐ_CP_D7"],
        "description": "Cần chunk Điều 7 NĐ 238 về nguyên tắc xác định mức học phí",
    },
    {
        "id": "Q04",
        "name": "Hệ thống giáo dục quốc dân — Điều 6 Luật GD",
        "intent_type": "single_lookup",
        "query": "Điều 6 Luật Giáo dục 2019 quy định hệ thống giáo dục quốc dân gồm những cấp học gì?",
        "required_chunks": ["Luật43_D6"],
        "description": "Cần chunk Điều 6 liệt kê các cấp học HTGDQD",
    },
    {
        "id": "Q05",
        "name": "Học phí — Điều 99 Luật GD 2019",
        "intent_type": "single_lookup",
        "query": "Điều 99 Luật Giáo dục 2019 quy định thế nào về học phí?",
        "required_chunks": ["Luật43_D99"],
        "description": "Cần chunk Điều 99 về học phí và chi phí dịch vụ giáo dục",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # COMPARISON (3 câu) — So sánh giữa 2+ văn bản
    # ═══════════════════════════════════════════════════════════════════════
    {
        "id": "Q06",
        "name": "So sánh học phí NĐ 238 vs NĐ 84",
        "intent_type": "comparison",
        "query": "So sánh quy định về mức học phí đại học công lập giữa Nghị định 238/2025/NĐ-CP và Nghị định 84/2020/NĐ-CP",
        "required_chunks": ["238_2025_NĐ_CP_D10", "84_2020_NĐ_CP_D5"],
        "description": "Cần chunks mức học phí từ CẢ HAI NĐ",
    },
    {
        "id": "Q07",
        "name": "So sánh quyền nhà giáo: Luật GD vs Luật Nhà giáo",
        "intent_type": "comparison",
        "query": "So sánh quyền của nhà giáo theo Luật Giáo dục 2019 và Luật Nhà giáo 2025",
        "required_chunks": ["Luật43_D70", "Luật73_D8"],
        "description": "Cần Điều 70 Luật GD (quyền nhà giáo) VÀ Điều 8 Luật Nhà giáo (quyền nhà giáo)",
    },
    {
        "id": "Q08",
        "name": "So sánh nghĩa vụ nhà giáo 2 Luật",
        "intent_type": "comparison",
        "query": "Sự khác nhau về nghĩa vụ của nhà giáo giữa Luật Giáo dục 2019 và Luật Nhà giáo 2025",
        "required_chunks": ["Luật43_D70", "Luật73_D9"],
        "description": "Cần Điều 70 Luật GD VÀ Điều 9 Luật 73 về nghĩa vụ nhà giáo",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # LISTING (4 câu) — Liệt kê danh sách
    # ═══════════════════════════════════════════════════════════════════════
    {
        "id": "Q09",
        "name": "Đối tượng miễn học phí NĐ 238",
        "intent_type": "listing",
        "query": "Liệt kê các đối tượng được miễn học phí theo Nghị định 238/2025/NĐ-CP",
        "required_chunks": ["238_2025_NĐ_CP_D15"],
        "description": "Cần chunk Điều 15 NĐ 238 liệt kê ĐT miễn HP",
    },
    {
        "id": "Q10",
        "name": "Đối tượng không phải đóng học phí NĐ 238",
        "intent_type": "listing",
        "query": "Những ai không phải đóng học phí theo Nghị định 238/2025?",
        "required_chunks": ["238_2025_NĐ_CP_D14"],
        "description": "Cần chunk Điều 14 NĐ 238 — đối tượng không phải đóng HP",
    },
    {
        "id": "Q11",
        "name": "Trường hợp không được dạy thêm",
        "intent_type": "listing",
        "query": "Các trường hợp không được dạy thêm, tổ chức dạy thêm theo Thông tư 29/2024",
        "required_chunks": ["29_2024_TT_BGDĐT_D4"],
        "description": "Cần chunk Điều 4 TT 29/2024 liệt kê trường hợp cấm dạy thêm",
    },
    {
        "id": "Q12",
        "name": "Những việc nhà giáo không được làm",
        "intent_type": "listing",
        "query": "Liệt kê những việc nhà giáo không được làm theo Luật Nhà giáo 2025",
        "required_chunks": ["Luật73_D11"],
        "description": "Cần chunk Điều 11 Luật 73/2025",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # ELIGIBILITY_CHECK (3 câu) — Kiểm tra điều kiện
    # ═══════════════════════════════════════════════════════════════════════
    {
        "id": "Q13",
        "name": "Sinh viên sư phạm bồi hoàn kinh phí",
        "intent_type": "eligibility_check",
        "query": "Sinh viên sư phạm trong trường hợp nào phải bồi hoàn kinh phí đào tạo?",
        "required_chunks": ["116_2020_NĐ_CP_D6"],
        "description": "Cần chunk Điều 6 NĐ 116/2020 về bồi hoàn kinh phí",
    },
    {
        "id": "Q14",
        "name": "Mức hỗ trợ sinh viên sư phạm",
        "intent_type": "eligibility_check",
        "query": "Sinh viên sư phạm được hỗ trợ bao nhiêu tiền sinh hoạt phí mỗi tháng?",
        "required_chunks": ["116_2020_NĐ_CP_D4"],
        "description": "Cần chunk Điều 4 NĐ 116/2020 về mức 3,63 triệu/tháng",
    },
    {
        "id": "Q15",
        "name": "Tiêu chuẩn giảng viên hạng III",
        "intent_type": "eligibility_check",
        "query": "Tiêu chuẩn chức danh giảng viên đại học hạng III là gì?",
        "required_chunks": ["40_2020_TT_BGDĐT_D5"],
        "description": "Cần chunk Điều 5 TT 40/2020 về giảng viên hạng III",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # PROCEDURE (3 câu) — Thủ tục, hồ sơ, quy trình
    # ═══════════════════════════════════════════════════════════════════════
    {
        "id": "Q16",
        "name": "Thủ tục thành lập trường mầm non",
        "intent_type": "procedure",
        "query": "Hồ sơ thành lập trường mầm non tư thục gồm những gì?",
        "required_chunks": ["125_2024_NĐ_CP_D4"],
        "description": "Cần chunk Điều 4 NĐ 125/2024 về thủ tục thành lập trường mầm non",
    },
    {
        "id": "Q17",
        "name": "Điều kiện thành lập trường mầm non",
        "intent_type": "procedure",
        "query": "Điều kiện thành lập trường mầm non theo Nghị định 125/2024 là gì?",
        "required_chunks": ["125_2024_NĐ_CP_D3"],
        "description": "Cần chunk Điều 3 NĐ 125/2024 về điều kiện",
    },
    {
        "id": "Q18",
        "name": "Thủ tục đăng ký hỗ trợ SV sư phạm",
        "intent_type": "procedure",
        "query": "Thủ tục đăng ký hỗ trợ tiền đóng học phí cho sinh viên sư phạm theo Nghị định 116/2020?",
        "required_chunks": ["116_2020_NĐ_CP_D7"],
        "description": "Cần chunk Điều 7 NĐ 116/2020 về thủ tục đăng ký",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # CROSS_DOCUMENT (4 câu) — Truy vấn xuyên văn bản
    # ═══════════════════════════════════════════════════════════════════════
    {
        "id": "Q19",
        "name": "Quyền và nghĩa vụ nhà giáo toàn diện",
        "intent_type": "cross_document",
        "query": "Tổng hợp quyền và nghĩa vụ của nhà giáo theo quy định pháp luật hiện hành",
        "required_chunks": ["Luật73_D8", "Luật73_D9"],
        "description": "Cần Điều 8 (quyền) VÀ Điều 9 (nghĩa vụ) từ Luật Nhà giáo",
    },
    {
        "id": "Q20",
        "name": "Chính sách hỗ trợ sinh viên sư phạm",
        "intent_type": "cross_document",
        "query": "Chính sách hỗ trợ tiền đóng học phí và sinh hoạt phí cho sinh viên sư phạm",
        "required_chunks": ["116_2020_NĐ_CP_D4", "116_2020_NĐ_CP_D6"],
        "description": "Cần cả mức hỗ trợ (Đ4) lẫn quy định bồi hoàn (Đ6)",
    },
    {
        "id": "Q21",
        "name": "Quy định dạy thêm học thêm",
        "intent_type": "cross_document",
        "query": "Quy định về dạy thêm, học thêm trong và ngoài nhà trường",
        "required_chunks": ["29_2024_TT_BGDĐT_D5", "29_2024_TT_BGDĐT_D6"],
        "description": "Cần Điều 5 (trong nhà trường) VÀ Điều 6 (ngoài nhà trường)",
    },
    {
        "id": "Q22",
        "name": "Đạo đức nhà giáo theo Luật Nhà giáo",
        "intent_type": "cross_document",
        "query": "Quy định về đạo đức nhà giáo và những việc nhà giáo không được làm",
        "required_chunks": ["Luật73_D10", "Luật73_D11"],
        "description": "Cần Điều 10 (đạo đức) VÀ Điều 11 (cấm)",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # DEFINITION (3 câu) — Tra cứu định nghĩa
    # ═══════════════════════════════════════════════════════════════════════
    {
        "id": "Q23",
        "name": "Định nghĩa dạy thêm học thêm",
        "intent_type": "definition",
        "query": "Dạy thêm, học thêm được định nghĩa như thế nào theo Thông tư 29/2024?",
        "required_chunks": ["29_2024_TT_BGDĐT_D2"],
        "description": "Cần chunk Điều 2 TT 29/2024 giải thích từ ngữ",
    },
    {
        "id": "Q24",
        "name": "Định nghĩa nhà giáo",
        "intent_type": "definition",
        "query": "Nhà giáo, giáo viên, giảng viên được định nghĩa khác nhau thế nào theo Luật Nhà giáo 2025?",
        "required_chunks": ["Luật73_D4"],
        "description": "Cần chunk Điều 4 Luật 73/2025 giải thích từ ngữ",
    },
    {
        "id": "Q25",
        "name": "Định nghĩa chức danh nhà giáo",
        "intent_type": "definition",
        "query": "Chức danh nhà giáo là gì theo Luật Nhà giáo 2025?",
        "required_chunks": ["Luật73_D12"],
        "description": "Cần chunk Điều 12 Luật 73/2025 về chức danh nhà giáo",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE — Binary (0/1)
# ══════════════════════════════════════════════════════════════════════════════

def check_retrieval(context: str, required_chunks: list, graph) -> dict:
    """
    Kiểm tra context có chứa ĐỦ các chunk cần thiết không.
    
    Cách kiểm tra: với mỗi required_chunk_id, tìm full_text của node đó
    trong graph, rồi check xem đoạn text đặc trưng có xuất hiện trong context.
    
    Returns: {"pass": bool, "found": [...], "missing": [...]}
    """
    found = []
    missing = []
    
    for chunk_id in required_chunks:
        node_id = f"node:{chunk_id}"
        node_data = graph.nodes.get(node_id, {})
        full_text = node_data.get("full_text", "")
        
        if not full_text:
            # Node không có full_text → mark missing
            missing.append(chunk_id)
            continue
        
        # Lấy đoạn đặc trưng để check (30 ký tự đầu tiên có nghĩa)
        # Bỏ qua khoảng trắng đầu
        check_text = full_text.strip()[:80]
        
        if check_text and check_text in context:
            found.append(chunk_id)
        else:
            # Fallback: check bằng 2-3 đoạn ngắn hơn xen kẽ
            words = full_text.split()
            if len(words) >= 10:
                # Lấy 5 từ liên tiếp ở vị trí 20% và 50% của text
                pos1 = max(0, len(words) // 5)
                pos2 = max(0, len(words) // 2)
                snippet1 = " ".join(words[pos1:pos1+5])
                snippet2 = " ".join(words[pos2:pos2+5])
                
                if snippet1 in context or snippet2 in context:
                    found.append(chunk_id)
                else:
                    missing.append(chunk_id)
            elif full_text.strip() and full_text.strip()[:40] in context:
                found.append(chunk_id)
            else:
                missing.append(chunk_id)
    
    return {
        "pass": len(missing) == 0,
        "found": found,
        "missing": missing,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(retriever, intent_classifier) -> dict:
    """Run all benchmark queries and return results."""
    
    graph = retriever.index.graph
    
    print("\n" + "=" * 80)
    print("BENCHMARK v3 — Ground Truth Retrieval Evaluation")
    print(f"  Queries: {len(BENCHMARKS)}")
    print(f"  Scoring: Retrieval (0/1 binary) + Intent Classification (0/1)")
    print("=" * 80)
    
    results = {}
    total_retrieval = 0
    total_intent = 0
    type_stats = defaultdict(lambda: {"retrieval_pass": 0, "intent_pass": 0, "count": 0})
    
    for bm in BENCHMARKS:
        qid = bm["id"]
        intent_type = bm["intent_type"]
        
        print(f"\n{'─'*70}")
        print(f" {qid}: {bm['name']} [{intent_type}]")
        print(f" Query: {bm['query']}")
        print(f" Required: {bm['required_chunks']}")
        
        # ── Step 1: Classify intent ──
        t0 = time.time()
        intent = intent_classifier.analyze(bm["query"])
        intent_time = time.time() - t0
        
        intent_match = intent.type == intent_type
        
        classifier_label = "Regex" if intent._fallback_used else "Qwen-FT"
        print(f" Intent ({classifier_label}): {intent.type} | docs={intent.documents[:3]}")
        print(f" Topic: '{intent.topic or '(none)'}'")
        if intent.sub_queries:
            print(f" Sub-queries ({len(intent.sub_queries)}): {intent.sub_queries}")
        
        if not intent_match:
            print(f" ⚠️ Expected: {intent_type}, Got: {intent.type}")
        
        # ── Step 2: Retrieve ──
        t1 = time.time()
        context = retriever.retrieve(bm["query"], intent=intent, top_k=5)
        retrieval_time = time.time() - t1
        
        # ── Step 3: Check retrieval ──
        check = check_retrieval(context, bm["required_chunks"], graph)
        retrieval_pass = check["pass"]
        
        # ── Scoring ──
        retrieval_score = 1 if retrieval_pass else 0
        intent_score = 1 if intent_match else 0
        
        total_retrieval += retrieval_score
        total_intent += intent_score
        
        type_stats[intent_type]["retrieval_pass"] += retrieval_score
        type_stats[intent_type]["intent_pass"] += intent_score
        type_stats[intent_type]["count"] += 1
        
        # ── Display ──
        r_icon = "✅" if retrieval_pass else "❌"
        i_icon = "✅" if intent_match else "❌"
        
        print(f" Retrieval: {r_icon} | Found: {check['found']} | Missing: {check['missing']}")
        print(f" Intent:    {i_icon} | Time: {intent_time*1000:.0f}ms | Retrieval: {retrieval_time:.2f}s | Context: {len(context)} chars")
        
        results[qid] = {
            "name": bm["name"],
            "intent_type": intent_type,
            "classified_as": intent.type,
            "topic": intent.topic,
            "sub_queries": intent.sub_queries or [],
            "intent_correct": intent_match,
            "retrieval_pass": retrieval_pass,
            "found_chunks": check["found"],
            "missing_chunks": check["missing"],
            "context_length": len(context),
            "retrieval_time": round(retrieval_time, 2),
        }
    
    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 100)
    print("SCORE TABLE — Benchmark v3")
    print("=" * 100)
    
    header = f"| {'#':<4} | {'Tên câu hỏi':<45} | {'Type':<18} | {'Classified':<18} | {'Topic':<30} | {'Intent':>6} | {'Retrieval':>9} |"
    print(header)
    print("|" + "-"*6 + "|" + "-"*47 + "|" + "-"*20 + "|" + "-"*20 + "|" + "-"*32 + "|" + "-"*8 + "|" + "-"*11 + "|")
    
    for qid in sorted(results.keys()):
        r = results[qid]
        i_mark = "✓" if r["intent_correct"] else "✗"
        r_mark = "✓" if r["retrieval_pass"] else "✗"
        topic_display = (r.get('topic', '') or '')[:30]
        line = f"| {qid:<4} | {r['name']:<45} | {r['intent_type']:<18} | {r['classified_as']:<18} | {topic_display:<30} | {i_mark:>6} | {r_mark:>9} |"
        print(line)
    
    print("|" + "-"*6 + "|" + "-"*47 + "|" + "-"*20 + "|" + "-"*20 + "|" + "-"*32 + "|" + "-"*8 + "|" + "-"*11 + "|")
    print(f"| {'':4} | {'TOTAL':<45} | {'':18} | {'':18} | {'':30} | {total_intent:>4}/{len(BENCHMARKS):<1} | {total_retrieval:>7}/{len(BENCHMARKS):<1} |")
    
    # ── Type Breakdown ──
    print("\n" + "=" * 80)
    print("BREAKDOWN BY INTENT TYPE")
    print("=" * 80)
    print(f"| {'Intent Type':<20} | {'Retrieval':>12} | {'Intent':>12} | {'Count':>5} |")
    print("|" + "-"*22 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*7 + "|")
    
    for itype in ["single_lookup", "comparison", "listing", "eligibility_check",
                   "procedure", "cross_document", "definition"]:
        ts = type_stats[itype]
        if ts["count"] == 0:
            continue
        print(f"| {itype:<20} | {ts['retrieval_pass']:>4}/{ts['count']:<6} | {ts['intent_pass']:>4}/{ts['count']:<6} | {ts['count']:>5} |")
    
    # ── Final Summary ──
    print(f"\n{'='*60}")
    print(f" RETRIEVAL ACCURACY:  {total_retrieval}/{len(BENCHMARKS)} ({total_retrieval/len(BENCHMARKS)*100:.0f}%)")
    print(f" INTENT ACCURACY:     {total_intent}/{len(BENCHMARKS)} ({total_intent/len(BENCHMARKS)*100:.0f}%)")
    print(f"{'='*60}")
    
    if total_retrieval >= 20:
        print(f"\n✅ PASSED — Retrieval {total_retrieval}/{len(BENCHMARKS)} ≥ 20 threshold")
    else:
        print(f"\n⚠️ NEEDS IMPROVEMENT — Retrieval {total_retrieval}/{len(BENCHMARKS)} < 20 threshold")
    
    return {
        "results": results,
        "total_retrieval": total_retrieval,
        "total_intent": total_intent,
        "max_score": len(BENCHMARKS),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("Benchmark v3 — Initializing...")
    print("=" * 80)
    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "final")
    kg_path = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "knowledge_graph", "entity_graph.json")
    data_dir = os.path.abspath(data_dir)
    kg_path = os.path.abspath(kg_path)
    
    # ── Validate ground truth chunks exist before loading heavy models ──
    print("\n[Pre-check] Validating ground truth chunk IDs...")
    all_chunk_ids = set()
    for bm in BENCHMARKS:
        for cid in bm["required_chunks"]:
            all_chunk_ids.add(cid)
    
    found_in_files = set()
    data_path = os.path.join(data_dir)
    for fp in glob.glob(os.path.join(data_path, "*.json")):
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            for c in chunks:
                cid = c.get("id", "")
                if cid in all_chunk_ids:
                    found_in_files.add(cid)
        except Exception:
            pass
    
    missing_ids = all_chunk_ids - found_in_files
    if missing_ids:
        print(f"  ⚠️ WARNING: {len(missing_ids)} chunk IDs not found in data files:")
        for mid in sorted(missing_ids):
            print(f"    - {mid}")
        print("  These queries will fail. Consider updating the benchmark.")
    else:
        print(f"  ✅ All {len(all_chunk_ids)} ground truth chunk IDs verified.")
    
    # ── Load models ──
    from query_intent import QwenIntentClassifier
    
    use_regex = "--regex" in sys.argv
    mode_label = "Regex-only" if use_regex else "Qwen2.5-1.5B Fine-tuned"
    print(f"\n[Step 1] Loading QwenIntentClassifier... Mode: {mode_label}")
    intent_classifier = QwenIntentClassifier(
        model_name="manhcuong2005/qwen2.5-1.5b-legal-edu-v5",
        use_regex_only=use_regex,
    )
    if not use_regex:
        intent_classifier._load_model()
    
    from book_index import BookIndex
    from retriever import BookRAGRetriever
    
    print(f"\n[Step 2] Loading BookIndex...")
    t0 = time.time()
    index = BookIndex(data_dir, kg_path)
    index.load_index()
    print(f"Index loaded in {time.time() - t0:.2f}s")
    
    intent_classifier.doc_registry = index.doc_registry
    
    print(f"\n[Step 3] Loading Retriever...")
    retriever = BookRAGRetriever(index)
    
    # ── Run benchmark ──
    summary = run_benchmark(retriever, intent_classifier)
    
    # ── Save results ──
    output_path = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "benchmark_v3.json")
    output_path = os.path.abspath(output_path)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "regex" if use_regex else "qwen",
            "total_retrieval": summary["total_retrieval"],
            "total_intent": summary["total_intent"],
            "max_score": summary["max_score"],
            "retrieval_pct": round(summary["total_retrieval"] / summary["max_score"] * 100, 1),
            "intent_pct": round(summary["total_intent"] / summary["max_score"] * 100, 1),
            "results": summary["results"],
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
