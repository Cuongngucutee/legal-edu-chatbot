"""
Bước 3: Gộp kết quả trích xuất mới (patch) vào Knowledge Graph gốc.

Input:
  - outputs/knowledge_graph/entity_graph.json  (Graph gốc - 4678 chunks)
  - data/missing/entity_graph_patch.json       (Graph bổ sung - 235 chunks)

Output:
  - outputs/knowledge_graph/entity_graph.json  (Graph gộp hoàn chỉnh)

Cách chạy:
    python backend/scripts/merge_graph.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR    = Path(__file__).resolve().parent.parent.parent
MAIN_GRAPH  = ROOT_DIR / "outputs" / "knowledge_graph" / "entity_graph.json"
PATCH_GRAPH = ROOT_DIR / "outputs" / "knowledge_graph" / "missing_entity_graph.json"
BACKUP_PATH = ROOT_DIR / "outputs" / "knowledge_graph" / "entity_graph_BACKUP.json"


def load_graph(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("🔗 GỘP KNOWLEDGE GRAPH (PATCH → GỐC)")
    print("=" * 60)

    # 1. Đọc graph gốc
    print(f"\n📊 Đang đọc graph gốc: {MAIN_GRAPH}")
    main_data = load_graph(MAIN_GRAPH)
    main_graph = main_data["bookrag_entity_graph"]

    main_entities:  Dict[str, dict] = {e["id"]: e for e in main_graph["entities"]}
    main_relations: List[dict]       = main_graph["relations"]
    main_mappings:  List[dict]       = main_graph["mappings"]

    print(f"   Entities : {len(main_entities)}")
    print(f"   Relations: {len(main_relations)}")
    print(f"   Mappings : {len(main_mappings)}")

    # 2. Đọc graph patch
    print(f"\n📦 Đang đọc graph patch: {PATCH_GRAPH}")
    patch_data = load_graph(PATCH_GRAPH)
    patch_graph = patch_data["bookrag_entity_graph"]

    patch_entities:  List[dict] = patch_graph["entities"]
    patch_relations: List[dict] = patch_graph["relations"]
    patch_mappings:  List[dict] = patch_graph["mappings"]

    print(f"   Entities : {len(patch_entities)}")
    print(f"   Relations: {len(patch_relations)}")
    print(f"   Mappings : {len(patch_mappings)}")

    # 3. Merge entities (ưu tiên description dài hơn)
    new_ent_count = 0
    updated_ent_count = 0
    for ent in patch_entities:
        eid = ent["id"]
        if eid not in main_entities:
            main_entities[eid] = ent
            new_ent_count += 1
        else:
            # Giữ description dài hơn
            if len(ent.get("description", "")) > len(main_entities[eid].get("description", "")):
                main_entities[eid] = ent
                updated_ent_count += 1

    print(f"\n✅ Entities mới thêm    : {new_ent_count}")
    print(f"   Entities cập nhật    : {updated_ent_count}")

    # 4. Merge relations (loại trùng)
    existing_rel_keys = set()
    for rel in main_relations:
        existing_rel_keys.add((rel["source"], rel["target"], rel["relation"]))

    new_rel_count = 0
    for rel in patch_relations:
        key = (rel["source"], rel["target"], rel["relation"])
        if key not in existing_rel_keys:
            main_relations.append(rel)
            existing_rel_keys.add(key)
            new_rel_count += 1

    print(f"   Relations mới thêm   : {new_rel_count}")

    # 5. Merge mappings (loại trùng)
    existing_map_keys = set()
    for m in main_mappings:
        existing_map_keys.add((m["tree_node"], m["entity"], m["relation"]))

    new_map_count = 0
    for m in patch_mappings:
        key = (m["tree_node"], m["entity"], m["relation"])
        if key not in existing_map_keys:
            main_mappings.append(m)
            existing_map_keys.add(key)
            new_map_count += 1

    print(f"   Mappings mới thêm    : {new_map_count}")

    # 6. Backup graph cũ
    print(f"\n💾 Backup graph cũ -> {BACKUP_PATH}")
    import shutil
    shutil.copy2(MAIN_GRAPH, BACKUP_PATH)

    # 7. Lưu graph gộp
    merged = {
        "bookrag_entity_graph": {
            "metadata": {
                "model": main_graph["metadata"].get("model", "merged"),
                "total_entities":  len(main_entities),
                "total_relations": len(main_relations),
                "total_mappings":  len(main_mappings),
                "entity_resolution_applied": main_graph["metadata"].get("entity_resolution_applied", False),
                "patch_applied": True,
            },
            "entities":  list(main_entities.values()),
            "relations": main_relations,
            "mappings":  main_mappings,
        }
    }

    with open(MAIN_GRAPH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Đã gộp xong! Kết quả cuối cùng:")
    print(f"   Entities : {len(main_entities)}")
    print(f"   Relations: {len(main_relations)}")
    print(f"   Mappings : {len(main_mappings)}")
    print(f"\n   File đã lưu tại: {MAIN_GRAPH}")


if __name__ == "__main__":
    main()
