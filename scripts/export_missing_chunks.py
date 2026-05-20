"""
Bước 1: Gom 235 chunks bị thiếu từ data gốc thành 1 file JSON duy nhất.
Script này sẽ đọc coverage_report.txt + data gốc để tạo ra:
  -> data/missing/missing_chunks.json

File output có cùng cấu trúc mảng chunk như data gốc,
nên build_graph.py có thể đọc thẳng mà không cần sửa gì.

Cách chạy:
    python backend/scripts/export_missing_chunks.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data" / "final"
KG_PATH  = ROOT_DIR / "outputs" / "knowledge_graph" / "entity_graph.json"
OUTPUT_DIR = ROOT_DIR / "data" / "missing"
OUTPUT_FILE = OUTPUT_DIR / "missing_chunks.json"

MIN_TEXT_LENGTH = 30


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("📦 GOM CÁC CHUNK BỊ THIẾU THÀNH 1 FILE JSON")
    print("=" * 60)

    # 1. Thu thập toàn bộ chunk từ data gốc
    print("\n📂 Đang quét data gốc...")
    all_chunks_by_id = {}
    json_files = sorted(DATA_DIR.glob("*.json"))

    for fpath in json_files:
        with open(fpath, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        for chunk in chunks:
            chunk_id = chunk.get("id", "").strip()
            full_text = chunk.get("content", {}).get("full_text", "")
            if len(full_text.strip()) < MIN_TEXT_LENGTH:
                continue
            if chunk_id:
                all_chunks_by_id[chunk_id] = chunk

    print(f"   Tổng chunks hợp lệ: {len(all_chunks_by_id)}")

    # 2. Đọc Knowledge Graph để tìm chunks đã cover
    print(f"\n📊 Đang đọc Knowledge Graph...")
    with open(KG_PATH, "r", encoding="utf-8") as f:
        kg = json.load(f)

    mappings = kg["bookrag_entity_graph"].get("mappings", [])
    covered_ids = set()
    for m in mappings:
        tn = m.get("tree_node", "")
        covered_ids.add(tn[5:] if tn.startswith("node:") else tn)

    print(f"   Chunks đã cover: {len(covered_ids)}")

    # 3. Tìm chunks thiếu
    missing_ids = set(all_chunks_by_id.keys()) - covered_ids
    print(f"\n❌ Chunks bị thiếu: {len(missing_ids)}")

    # 4. Gom chunks thiếu vào 1 mảng
    missing_chunks = []
    for cid in sorted(missing_ids):
        missing_chunks.append(all_chunks_by_id[cid])

    # 5. Xuất file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(missing_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Đã lưu {len(missing_chunks)} chunks thiếu vào:")
    print(f"   {OUTPUT_FILE}")
    print(f"\n👉 Bước tiếp theo: Chạy build_graph.py với data_dir trỏ vào thư mục 'data/missing'")
    print(f"   và output_path trỏ vào 'data/missing/entity_graph_patch.json'")


if __name__ == "__main__":
    main()
