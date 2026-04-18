"""
Benchmark TOC + Catalog Injection — Kiểm tra riêng cho câu hỏi tóm tắt & thống kê.

Benchmark này KHÔNG load Qwen model (tốn RAM + chậm trên MPS).
Thay vào đó, nó mock QueryIntent để test trực tiếp logic TOC/Catalog trong Retriever.
"""
import os
import sys
import time
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "core"))

from book_index import BookIndex
from retriever import BookRAGRetriever, _SUMMARY_PATTERNS, _STAT_PATTERNS
from query_intent import QueryIntent
import re

# =====================================================
# CẤU HÌNH
# =====================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KG_PATH = os.path.join(ROOT_DIR, "outputs", "knowledge_graph", "entity_graph.json")
DATA_DIR = os.path.join(ROOT_DIR, "data", "final")

# =====================================================
# TEST CASES — TOC (Summary)
# =====================================================
SUMMARY_TESTS = [
    {
        "id": "S01",
        "name": "Tóm tắt NĐ 238",
        "query": "Nghị định 238/2025/NĐ-CP quy định gì?",
        "doc_ref": "238/2025/NĐ-CP",
        "expect_toc": True,
        "expect_min_articles": 10,
        "type": "summary",
    },
    {
        "id": "S02",
        "name": "Tóm tắt Luật Giáo dục",
        "query": "Luật Giáo dục 2019 nói về điều gì?",
        "doc_ref": "43/2019/QH14",
        "expect_toc": True,
        "expect_min_articles": 20,
        "type": "summary",
    },
    {
        "id": "S03",
        "name": "Tóm tắt Luật Nhà giáo",
        "query": "Luật Nhà giáo 2025 gồm những gì?",
        "doc_ref": "73/2025/QH15",
        "expect_toc": True,
        "expect_min_articles": 10,
        "type": "summary",
    },
    {
        "id": "S04",
        "name": "Cấu trúc TT 29",
        "query": "Thông tư 29/2024/TT-BGDĐT có bao nhiêu điều?",
        "doc_ref": "29/2024/TT-BGDĐT",
        "expect_toc": True,
        "expect_min_articles": 3,
        "type": "summary",
    },
    {
        "id": "S05",
        "name": "Nội dung NĐ 84",
        "query": "Nghị định 84/2020/NĐ-CP quy định về vấn đề gì?",
        "doc_ref": "84/2020/NĐ-CP",
        "expect_toc": True,
        "expect_min_articles": 5,
        "type": "summary",
    },
]

# Câu hỏi KHÔNG phải summary → phải KHÔNG inject TOC
NON_SUMMARY_TESTS = [
    {
        "id": "N01",
        "name": "Tra cứu cụ thể (không phải tóm tắt)",
        "query": "Mức học phí đại học công lập theo Nghị định 238 là bao nhiêu?",
        "doc_ref": "238/2025/NĐ-CP",
        "expect_toc": False,
        "type": "non_summary",
    },
    {
        "id": "N02",
        "name": "Tra cứu điều khoản cụ thể",
        "query": "Điều 22 Luật Giáo dục 2019 quy định những hành vi nào bị nghiêm cấm?",
        "doc_ref": "43/2019/QH14",
        "expect_toc": False,
        "type": "non_summary",
    },
    {
        "id": "N03",
        "name": "So sánh (không phải tóm tắt)",
        "query": "So sánh quyền nhà giáo giữa Luật Giáo dục và Luật Nhà giáo",
        "doc_ref": "",
        "expect_toc": False,
        "type": "non_summary",
    },
]

# =====================================================
# TEST CASES — Catalog (Statistical/Listing)
# =====================================================
STAT_TESTS = [
    {
        "id": "C01",
        "name": "Nghị định năm 2026",
        "query": "Những nghị định nào được ban hành trong năm 2026?",
        "expect_catalog": True,
        "expect_year": "2026",
        "expect_doc_type": "Nghị định",
        "expect_min_results": 1,
    },
    {
        "id": "C02",
        "name": "Thông tư về tuyển sinh",
        "query": "Những thông tư nào nói về tuyển sinh?",
        "expect_catalog": True,
        "expect_year": None,
        "expect_doc_type": "Thông tư",
        "expect_min_results": 3,
    },
    {
        "id": "C03",
        "name": "Văn bản năm 2025",
        "query": "Những văn bản nào được ban hành năm 2025?",
        "expect_catalog": True,
        "expect_year": "2025",
        "expect_doc_type": None,
        "expect_min_results": 5,
    },
    {
        "id": "C04",
        "name": "Luật liên quan nhà giáo",
        "query": "Những luật nào liên quan đến nhà giáo?",
        "expect_catalog": True,
        "expect_year": None,
        "expect_doc_type": "Luật",
        "expect_min_results": 1,
    },
    {
        "id": "C05",
        "name": "Nghị định về học phí",
        "query": "Liệt kê các nghị định về học phí",
        "expect_catalog": True,
        "expect_year": None,
        "expect_doc_type": "Nghị định",
        "expect_min_results": 2,
    },
]

# Câu hỏi KHÔNG phải thống kê → KHÔNG inject catalog
NON_STAT_TESTS = [
    {
        "id": "X01",
        "name": "Tra cứu chi tiết (không phải thống kê)",
        "query": "Mức học phí đại học công lập là bao nhiêu?",
        "expect_catalog": False,
    },
    {
        "id": "X02",
        "name": "Định nghĩa (không phải thống kê)",
        "query": "Nhà giáo được định nghĩa thế nào theo Luật Nhà giáo 2025?",
        "expect_catalog": False,
    },
]


def run_benchmark():
    print("=" * 70)
    print("Benchmark TOC + Catalog Injection — Summary & Statistics")
    print("=" * 70)

    # ── Phase 1: Unit test _is_summary_query ──
    print("\n[Phase 1] Testing _is_summary_query() regex patterns...\n")
    
    phase1_pass = 0
    phase1_total = 0
    all_summary_tests = SUMMARY_TESTS + NON_SUMMARY_TESTS
    
    for test in all_summary_tests:
        phase1_total += 1
        query = test["query"]
        expected = test["expect_toc"]
        actual = BookRAGRetriever._is_summary_query(query)
        passed = actual == expected
        if passed:
            phase1_pass += 1
        
        icon = "✅" if passed else "❌"
        print(f"  {icon} {test['id']}: \"{query[:60]}...\"")
        if not passed:
            print(f"      Expected: {expected}, Got: {actual}")
    
    print(f"\n  Summary Pattern Detection: {phase1_pass}/{phase1_total}")

    # ── Phase 1b: Unit test _is_statistical_query ──
    print(f"\n{'=' * 70}")
    print("[Phase 1b] Testing _is_statistical_query() regex patterns...\n")
    
    phase1b_pass = 0
    phase1b_total = 0
    
    for test in STAT_TESTS:
        phase1b_total += 1
        query = test["query"]
        actual = BookRAGRetriever._is_statistical_query(query)
        passed = actual == test["expect_catalog"]
        if passed:
            phase1b_pass += 1
        icon = "✅" if passed else "❌"
        print(f"  {icon} {test['id']}: \"{query[:60]}...\"")
        
        # Also test filter extraction
        year = BookRAGRetriever._extract_year_filter(query)
        doc_type = BookRAGRetriever._extract_doc_type_filter(query)
        year_ok = year == test["expect_year"]
        type_ok = doc_type == test["expect_doc_type"]
        if not year_ok:
            print(f"      Year: expected={test['expect_year']}, got={year}")
        if not type_ok:
            print(f"      DocType: expected={test['expect_doc_type']}, got={doc_type}")
        if year_ok and type_ok:
            print(f"      Filters: year={year}, type={doc_type} ✓")
    
    for test in NON_STAT_TESTS:
        phase1b_total += 1
        query = test["query"]
        actual = BookRAGRetriever._is_statistical_query(query)
        passed = actual == test["expect_catalog"]
        if passed:
            phase1b_pass += 1
        icon = "✅" if passed else "❌"
        print(f"  {icon} {test['id']}: \"{query[:60]}...\"")
        if not passed:
            print(f"      Expected: {test['expect_catalog']}, Got: {actual}")
    
    print(f"\n  Statistical Pattern Detection: {phase1b_pass}/{phase1b_total}")

    # ── Phase 2: Test build_toc() với BookIndex thật ──
    print(f"\n{'=' * 70}")
    print("[Phase 2] Loading BookIndex (no Qwen model needed)...")
    
    t0 = time.time()
    index = BookIndex(kg_path=KG_PATH, data_dir=DATA_DIR)
    index.load_index()
    print(f"  BookIndex loaded in {time.time() - t0:.1f}s\n")
    
    print("Testing build_toc() for each document...\n")
    
    phase2_pass = 0
    phase2_total = 0
    
    for test in SUMMARY_TESTS:
        phase2_total += 1
        doc_ref = test["doc_ref"]
        toc = index.build_toc(doc_ref)
        
        if not toc:
            print(f"  ❌ {test['id']}: build_toc(\"{doc_ref}\") returned EMPTY")
            canonical = index.doc_registry.get(doc_ref, None)
            print(f"      doc_registry lookup: {doc_ref} → {canonical}")
            if canonical:
                nodes = index.doc_nodes.get(canonical, set())
                print(f"      doc_nodes count: {len(nodes)}")
            continue
        
        article_count = toc.count("- Điều")
        min_expected = test["expect_min_articles"]
        passed = article_count >= min_expected
        
        if passed:
            phase2_pass += 1
        
        icon = "✅" if passed else "⚠️"
        print(f"  {icon} {test['id']}: {test['name']}")
        print(f"      DOC: {doc_ref}")
        print(f"      Articles in TOC: {article_count} (min expected: {min_expected})")
        print(f"      TOC size: {len(toc)} chars")
        
        toc_preview = "\n".join(toc.split("\n")[:8])
        print(f"      Preview:\n{_indent(toc_preview, 8)}")
        print()
    
    print(f"  TOC Generation: {phase2_pass}/{phase2_total}")

    # ── Phase 2b: Test query_catalog() ──
    print(f"\n{'=' * 70}")
    print("[Phase 2b] Testing query_catalog()...\n")
    
    phase2b_pass = 0
    phase2b_total = 0
    
    for test in STAT_TESTS:
        phase2b_total += 1
        
        result = index.query_catalog(
            query=test["query"],
            doc_type=test["expect_doc_type"],
            year=test["expect_year"],
        )
        
        if not result:
            print(f"  ❌ {test['id']}: query_catalog returned EMPTY")
            continue
        
        # Count results
        result_count = result.count("\n\n") + (1 if result else 0)
        # Better: count numbered entries
        result_count = len(re.findall(r'\n\d+\.', result))
        min_expected = test["expect_min_results"]
        passed = result_count >= min_expected
        
        if passed:
            phase2b_pass += 1
        
        icon = "✅" if passed else "⚠️"
        print(f"  {icon} {test['id']}: {test['name']}")
        print(f"      Query: \"{test['query']}\"")
        print(f"      Filters: year={test['expect_year']}, type={test['expect_doc_type']}")
        print(f"      Results: {result_count} (min expected: {min_expected})")
        print(f"      Output size: {len(result)} chars")
        
        # Show first 5 lines
        preview = "\n".join(result.split("\n")[:8])
        print(f"      Preview:\n{_indent(preview, 8)}")
        print()
    
    print(f"  Catalog Query: {phase2b_pass}/{phase2b_total}")
    
    # ── Phase 3: Full integration test with Retriever (mocked intent) ──
    print(f"\n{'=' * 70}")
    print("[Phase 3] Testing full retrieval + TOC injection...\n")
    
    retriever = BookRAGRetriever(index)
    
    phase3_pass = 0
    phase3_total = 0
    
    for test in SUMMARY_TESTS:
        phase3_total += 1
        
        intent = QueryIntent(
            type="single_lookup",
            topic=test["name"],
            keywords=[],
            sub_queries=[test["query"]],
            documents=[test["doc_ref"]],
        )
        
        t0 = time.time()
        context = retriever.retrieve(test["query"], intent=intent, top_k=5)
        elapsed = time.time() - t0
        
        has_toc = "=== MỤC LỤC:" in context if context else False
        passed = has_toc == test["expect_toc"]
        
        if passed:
            phase3_pass += 1
        
        icon = "✅" if passed else "❌"
        print(f"  {icon} {test['id']}: {test['name']}")
        print(f"      TOC injected: {has_toc} | Context: {len(context)} chars | Time: {elapsed:.2f}s")
        print()
    
    for test in NON_SUMMARY_TESTS:
        phase3_total += 1
        
        intent = QueryIntent(
            type="single_lookup",
            topic=test["name"],
            keywords=[],
            sub_queries=[test["query"]],
            documents=[test["doc_ref"]] if test["doc_ref"] else [],
        )
        
        t0 = time.time()
        context = retriever.retrieve(test["query"], intent=intent, top_k=5)
        elapsed = time.time() - t0
        
        has_toc = "=== MỤC LỤC:" in context if context else False
        passed = has_toc == test["expect_toc"]
        
        if passed:
            phase3_pass += 1
        
        icon = "✅" if passed else "❌"
        print(f"  {icon} {test['id']}: {test['name']}")
        print(f"      TOC injected: {has_toc} (expected: {test['expect_toc']}) | Time: {elapsed:.2f}s")
        print()
    
    print(f"  TOC Integration: {phase3_pass}/{phase3_total}")

    # ── Phase 4: Full integration — Catalog injection ──
    print(f"\n{'=' * 70}")
    print("[Phase 4] Testing full retrieval + Catalog injection...\n")
    
    phase4_pass = 0
    phase4_total = 0
    
    for test in STAT_TESTS:
        phase4_total += 1
        
        intent = QueryIntent(
            type="listing",
            topic=test["name"],
            keywords=[],
            sub_queries=[test["query"]],
            documents=[],
        )
        
        t0 = time.time()
        context = retriever.retrieve(test["query"], intent=intent, top_k=5)
        elapsed = time.time() - t0
        
        has_catalog = "=== KẾT QUẢ TÌM KIẾM VĂN BẢN" in context if context else False
        passed = has_catalog == test["expect_catalog"]
        
        if passed:
            phase4_pass += 1
        
        icon = "✅" if passed else "❌"
        print(f"  {icon} {test['id']}: {test['name']}")
        print(f"      Query: \"{test['query']}\"")
        print(f"      Catalog injected: {has_catalog} | Context: {len(context) if context else 0} chars | Time: {elapsed:.2f}s")
        
        if has_catalog:
            result_count = len(re.findall(r'\n\d+\.', context))
            print(f"      Catalog entries: {result_count}")
        print()
    
    # Non-statistical queries should NOT trigger catalog
    for test in NON_STAT_TESTS:
        phase4_total += 1
        
        intent = QueryIntent(
            type="single_lookup",
            topic=test["name"],
            keywords=[],
            sub_queries=[test["query"]],
            documents=[],
        )
        
        t0 = time.time()
        context = retriever.retrieve(test["query"], intent=intent, top_k=5)
        elapsed = time.time() - t0
        
        has_catalog = "=== KẾT QUẢ TÌM KIẾM VĂN BẢN" in context if context else False
        passed = has_catalog == test["expect_catalog"]
        
        if passed:
            phase4_pass += 1
        
        icon = "✅" if passed else "❌"
        print(f"  {icon} {test['id']}: {test['name']}")
        print(f"      Catalog injected: {has_catalog} (expected: {test['expect_catalog']}) | Time: {elapsed:.2f}s")
        print()
    
    print(f"  Catalog Integration: {phase4_pass}/{phase4_total}")
    
    # ── Final Summary ──
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"  Phase 1  — Summary Patterns:     {phase1_pass}/{phase1_total}")
    print(f"  Phase 1b — Statistical Patterns:  {phase1b_pass}/{phase1b_total}")
    print(f"  Phase 2  — TOC Generation:        {phase2_pass}/{phase2_total}")
    print(f"  Phase 2b — Catalog Query:          {phase2b_pass}/{phase2b_total}")
    print(f"  Phase 3  — TOC Integration:       {phase3_pass}/{phase3_total}")
    print(f"  Phase 4  — Catalog Integration:   {phase4_pass}/{phase4_total}")
    total_pass = phase1_pass + phase1b_pass + phase2_pass + phase2b_pass + phase3_pass + phase4_pass
    total_all = phase1_total + phase1b_total + phase2_total + phase2b_total + phase3_total + phase4_total
    print(f"  ────────────────────────────────")
    print(f"  TOTAL:                            {total_pass}/{total_all}")
    print(f"{'=' * 70}")


def _indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.split("\n"))


if __name__ == "__main__":
    run_benchmark()
