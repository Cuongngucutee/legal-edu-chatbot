"""
Script kiểm tra Độ Phủ (Coverage) của Knowledge Graph so với Data gốc.
Quét toàn bộ 355 file JSON trong data/final, đối chiếu với mappings 
trong entity_graph.json để tìm ra các Điều/Khoản/Điểm bị thiếu.

Cách chạy:
    python backend/scripts/check_coverage.py
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# ============================================================
# CẤU HÌNH
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data" / "final"
KG_PATH  = ROOT_DIR / "outputs" / "knowledge_graph" / "entity_graph.json"
REPORT_PATH = ROOT_DIR / "outputs" / "coverage_report.txt"

MIN_TEXT_LENGTH = 30  # Giống threshold trong extracting.py


def main():
    # Đảm bảo encoding cho Windows Terminal
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("=" * 70)
    print("🔍 KIỂM TRA ĐỘ PHỦ KNOWLEDGE GRAPH vs DATA GỐC")
    print("=" * 70)

    # ──────────────────────────────────────────────────────
    # 1. Thu thập toàn bộ chunk_id từ data gốc
    # ──────────────────────────────────────────────────────
    print(f"\n📂 Đang quét thư mục: {DATA_DIR}")
    
    all_chunks = {}  # chunk_id -> metadata dict
    file_chunk_count = {}  # filename -> count
    skipped_short = 0
    
    json_files = sorted(DATA_DIR.glob("*.json"))
    print(f"   Tổng số file JSON: {len(json_files)}")
    
    for fpath in json_files:
        with open(fpath, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        count = 0
        for chunk in chunks:
            chunk_id = chunk.get("id", "").strip()
            full_text = chunk.get("content", {}).get("full_text", "")
            
            # Bỏ qua chunk quá ngắn (giống logic extracting.py)
            if len(full_text.strip()) < MIN_TEXT_LENGTH:
                skipped_short += 1
                continue
            
            if not chunk_id:
                continue
                
            meta = chunk.get("metadata", {})
            all_chunks[chunk_id] = {
                "file": fpath.name,
                "source": meta.get("source", ""),
                "article_number": str(meta.get("article_number", "")),
                "article_title": meta.get("article_title", ""),
                "text_length": len(full_text),
            }
            count += 1
        
        file_chunk_count[fpath.name] = count
    
    total_chunks = len(all_chunks)
    print(f"   Tổng chunks hợp lệ (>= {MIN_TEXT_LENGTH} ký tự): {total_chunks}")
    print(f"   Chunks bị bỏ qua (quá ngắn): {skipped_short}")

    # ──────────────────────────────────────────────────────
    # 2. Thu thập danh sách tree_node từ Knowledge Graph
    # ──────────────────────────────────────────────────────
    print(f"\n📊 Đang đọc Knowledge Graph: {KG_PATH}")
    
    with open(KG_PATH, "r", encoding="utf-8") as f:
        kg = json.load(f)
    
    graph_data = kg["bookrag_entity_graph"]
    mappings = graph_data.get("mappings", [])
    
    # Trích xuất danh sách tree_node duy nhất (bỏ prefix "node:")
    covered_chunk_ids = set()
    for m in mappings:
        tree_node = m.get("tree_node", "")
        if tree_node.startswith("node:"):
            covered_chunk_ids.add(tree_node[5:])  # bỏ "node:"
        else:
            covered_chunk_ids.add(tree_node)
    
    print(f"   Tổng mappings: {len(mappings)}")
    print(f"   Chunks đã được cover (unique): {len(covered_chunk_ids)}")

    # ──────────────────────────────────────────────────────
    # 3. So sánh và tìm THIẾU
    # ──────────────────────────────────────────────────────
    missing_ids = set(all_chunks.keys()) - covered_chunk_ids
    covered_ids = set(all_chunks.keys()) & covered_chunk_ids
    
    coverage_pct = (len(covered_ids) / total_chunks * 100) if total_chunks > 0 else 0
    
    print(f"\n{'=' * 70}")
    print(f"📈 KẾT QUẢ TỔNG QUAN")
    print(f"{'=' * 70}")
    print(f"   ✅ Đã cover  : {len(covered_ids)}/{total_chunks} ({coverage_pct:.1f}%)")
    print(f"   ❌ Đang THIẾU : {len(missing_ids)}/{total_chunks} ({100-coverage_pct:.1f}%)")

    # ──────────────────────────────────────────────────────
    # 4. Phân loại chi tiết các mục thiếu
    # ──────────────────────────────────────────────────────
    missing_by_file = defaultdict(list)
    missing_by_type = defaultdict(int)  # CAN_CU, Điều, Khoản...
    
    for cid in sorted(missing_ids):
        info = all_chunks[cid]
        missing_by_file[info["file"]].append(info)
        
        art_num = info["article_number"]
        if art_num == "CAN_CU":
            missing_by_type["Căn cứ pháp lý"] += 1
        elif art_num.isdigit():
            missing_by_type["Điều luật"] += 1
        else:
            missing_by_type["Khác (Phụ lục, Khoản riêng...)"] += 1
    
    print(f"\n📋 PHÂN LOẠI THIẾU:")
    for mtype, count in sorted(missing_by_type.items(), key=lambda x: -x[1]):
        print(f"   • {mtype}: {count}")

    # ──────────────────────────────────────────────────────
    # 5. Thống kê file nào thiếu nhiều nhất
    # ──────────────────────────────────────────────────────
    print(f"\n📁 TOP 20 FILE THIẾU NHIỀU NHẤT:")
    print(f"   {'File':<55} {'Thiếu':>6} / {'Tổng':>6}")
    print(f"   {'-'*55} {'-'*6}   {'-'*6}")
    
    sorted_missing = sorted(missing_by_file.items(), key=lambda x: -len(x[1]))
    for fname, items in sorted_missing[:20]:
        total_in_file = file_chunk_count.get(fname, "?")
        print(f"   {fname:<55} {len(items):>6} / {total_in_file:>6}")

    # ──────────────────────────────────────────────────────
    # 6. Xuất báo cáo chi tiết ra file
    # ──────────────────────────────────────────────────────
    os.makedirs(REPORT_PATH.parent, exist_ok=True)
    
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("BÁO CÁO ĐỘ PHỦ KNOWLEDGE GRAPH\n")
        f.write(f"Cover: {len(covered_ids)}/{total_chunks} ({coverage_pct:.1f}%)\n")
        f.write(f"Thiếu: {len(missing_ids)}/{total_chunks}\n")
        f.write("=" * 80 + "\n\n")
        
        for fname, items in sorted_missing:
            f.write(f"\n{'─' * 60}\n")
            f.write(f"📄 {fname} (Thiếu {len(items)} chunks)\n")
            f.write(f"{'─' * 60}\n")
            for info in items:
                art_num = info['article_number']
                title = info['article_title'][:80] if info['article_title'] else "(không có tiêu đề)"
                f.write(f"  ❌ [{art_num}] {title}\n")
    
    print(f"\n💾 Báo cáo chi tiết đã lưu tại: {REPORT_PATH}")

    # ──────────────────────────────────────────────────────
    # 7. In ra danh sách thiếu cụ thể (giới hạn 30 dòng)
    # ──────────────────────────────────────────────────────
    if missing_ids:
        print(f"\n{'=' * 70}")
        print(f"📝 MẪU CÁC MỤC ĐANG THIẾU (tối đa 30 mục đầu tiên):")
        print(f"{'=' * 70}")
        for i, cid in enumerate(sorted(missing_ids)):
            if i >= 30:
                print(f"   ... và {len(missing_ids) - 30} mục nữa. Xem chi tiết tại: {REPORT_PATH}")
                break
            info = all_chunks[cid]
            title = info['article_title'][:60] if info['article_title'] else ""
            print(f"   [{info['file']}] Điều {info['article_number']}: {title}")
    else:
        print("\n🎉 TUYỆT VỜI! Knowledge Graph đã cover 100% dữ liệu gốc!")

    print(f"\n{'=' * 70}")
    print("✅ Hoàn tất kiểm tra.")


if __name__ == "__main__":
    main()
