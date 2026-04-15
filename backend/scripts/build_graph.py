"""
BookRAG Entity Graph Builder
============================
Model  : Qwen/Qwen3-30B-A3B-Instruct-2507 (local via vLLM)
Paper  : BookRAG – arXiv:2512.03413

Schema chunk (Luật_123_2025_QH15_chunked.json):
  {
    "id": "Luật123_D1_Mod_D6",          ← dùng làm tree_node_id
    "metadata": {
      "source": "Luật 123/2025/QH15",
      "chapter": "Chương I",
      "section": "",                      ← có thể rỗng
      "article_number": "1",
      "article_title": "Điều 1. ...",
      "tags": [...],
      "target_article": "Điều 6",        ← điều luật thực sự bị sửa đổi
      "is_sub_split": true
    },
    "content": {
      "full_text": "...",
      "clause": [                         ← optional, danh sách khoản
        {"clause_id": "1", "text": "..."},
        ...
      ]
    }
  }
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME    = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# JSON Schema cho guided decoding (vLLM outlines backend)
ENTITY_GRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name":        {"type": "string"},
                    "type":        {
                        "type": "string",
                        "enum": [
                            "Organization",   # Bộ, Ủy ban, Quốc hội, Chính phủ...
                            "Concept",        # Khái niệm: kiểm định chất lượng, chuyển đổi số...
                            "Role",           # Vai trò: Bộ trưởng, Hiệu trưởng, Thủ tướng...
                            "Document",       # Luật, Nghị định, Quyết định, Giấy chứng nhận...
                            "EducationLevel", # Cấp học: Mầm non, THCS, THPT, Đại học...
                            "Policy",         # Chính sách, quy định cụ thể
                            "Entity",         # Fallback chung
                        ]
                    },
                    "description": {"type": "string"},
                },
                "required": ["name", "type", "description"],
                "additionalProperties": False,
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source":   {"type": "string"},
                    "target":   {"type": "string"},
                    "relation": {"type": "string"},
                },
                "required": ["source", "target", "relation"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities", "relations"],
    "additionalProperties": False,
}

ENTITY_RESOLUTION_SCHEMA = {
    "type": "object",
    "properties": {
        "merge_groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "canonical": {"type": "string"},
                    "aliases":   {"type": "array", "items": {"type": "string"}},
                },
                "required": ["canonical", "aliases"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["merge_groups"],
    "additionalProperties": False,
}


# ─────────────────────────────────────────────────────────────────────────────
# BUILDER
# ─────────────────────────────────────────────────────────────────────────────
class EntityGraphBuilder:
    """
    BookRAG Entity Graph Builder sử dụng Qwen3-30B-A3B-Instruct-2507 local.

    Pipeline:
      1. process_all()  → đọc từng file JSON, extract entities + relations
      2. resolve_entities() → merge coreferent entities (BookRAG Section 4)
      3. save()         → ghi entity_graph.json
    """

    # ── Prompts ──────────────────────────────────────────────────────────────
    _SYS_EXTRACT = (
        "Bạn là chuyên gia pháp lý Việt Nam và chuyên gia xây dựng Knowledge Graph.\n"
        "Nhiệm vụ: Trích xuất Thực thể (Entity) và Mối quan hệ (Relation) từ văn bản luật\n"
        "để xây dựng BookRAG Knowledge Graph phục vụ hệ thống hỏi đáp pháp luật giáo dục.\n\n"
        "Quy tắc BẮT BUỘC:\n"
        "1. Chỉ trích xuất thực thể THỰC SỰ xuất hiện hoặc được đề cập trong văn bản.\n"
        "2. Tên thực thể phải ĐẦY ĐỦ, không viết tắt.\n"
        "   VD: 'Bộ Giáo dục và Đào tạo' KHÔNG phải 'Bộ GD&ĐT' hay 'Bộ GD'.\n"
        "3. Quan hệ dùng động từ VIẾT_HOA_GẠCHDưới.\n"
        "   VD: QUẢN_LÝ, BAN_HÀNH, ÁP_DỤNG_CHO, QUY_ĐỊNH, SỬA_ĐỔI, BỔ_SUNG, THUỘC_VỀ.\n"
        "4. Trả về JSON hợp lệ theo schema — KHÔNG markdown, KHÔNG giải thích thêm."
    )

    _SYS_RESOLVE = (
        "Bạn là chuyên gia Entity Resolution cho Knowledge Graph pháp lý Việt Nam.\n"
        "Nhiệm vụ: Tìm các nhóm thực thể GIỐNG NHAU (cùng chỉ một đối tượng thực tế)\n"
        "và gom thành merge_groups để hợp nhất node.\n\n"
        "Quy tắc:\n"
        "- Chỉ gom khi CHẮC CHẮN là cùng thực thể (không phải chỉ tương tự).\n"
        "- 'canonical' là tên ĐẦY ĐỦ nhất trong nhóm.\n"
        "- 'aliases' là các tên khác (không bao gồm canonical).\n"
        "- Bỏ qua nhóm chỉ có 1 phần tử.\n"
        "- Trả về JSON hợp lệ — KHÔNG markdown, KHÔNG giải thích thêm."
    )

    def __init__(
        self,
        data_dir: str,
        output_path: str,
        vllm_base_url: str = VLLM_BASE_URL,
        model_name: str    = MODEL_NAME,
        enable_thinking: bool  = False,   # True → chính xác hơn nhưng chậm ~3x
        entity_resolution: bool = True,   # Merge coreferent entities (BookRAG §4)
        min_text_length: int   = 30,      # Bỏ qua chunk quá ngắn
        request_timeout: float = 120.0,
        retry_attempts: int    = 3,
        retry_delay: float     = 2.0,
    ):
        self.client = OpenAI(
            base_url=vllm_base_url,
            api_key="EMPTY",           # vLLM không cần real key
            timeout=request_timeout,
        )
        self.model_name       = model_name
        self.enable_thinking  = enable_thinking
        self.entity_resolution = entity_resolution
        self.min_text_length  = min_text_length
        self.retry_attempts   = retry_attempts
        self.retry_delay      = retry_delay

        self.data_dir    = Path(data_dir)
        self.output_path = Path(output_path)

        # Internal state
        self.entities: Dict[str, Dict]  = {}   # eid → entity
        self.relations: List[Dict]       = []
        self.node_entity_links: List[Dict] = []

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def slugify(text: str) -> str:
        """Chuẩn hóa tên thành id an toàn."""
        text = str(text).lower().strip()
        # Giữ lại ký tự tiếng Việt có dấu để slug có nghĩa hơn
        text = re.sub(r'\s+', '_', text)
        text = re.sub(r'[^\w_]', '', text, flags=re.UNICODE)
        return text.strip('_') or "unknown"

    def _build_tree_node_id(self, chunk: dict) -> str:
        """
        Tạo tree_node_id từ chunk.

        Ưu tiên dùng chunk['id'] (đã có sẵn, structured).
        Fallback về cách tạo thủ công nếu không có.
        """
        chunk_id = chunk.get("id", "").strip()
        if chunk_id:
            return f"node:{chunk_id}"

        # Fallback
        meta       = chunk.get("metadata", {})
        source     = meta.get("source", "unknown")
        art_num    = str(meta.get("article_number", "")).strip()
        target     = meta.get("target_article", "").strip()
        doc_slug   = self.slugify(source)
        suffix     = self.slugify(target) if target else f"d{art_num}"
        return f"node:{doc_slug}:{suffix}"

    def _build_context_header(self, chunk: dict) -> str:
        """Tạo context header để đưa vào prompt, giúp model hiểu bối cảnh chunk."""
        meta        = chunk.get("metadata", {})
        source      = meta.get("source", "")
        chapter     = meta.get("chapter", "")
        section     = meta.get("section", "")
        art_title   = meta.get("article_title", "")
        target      = meta.get("target_article", "")
        tags        = ", ".join(meta.get("tags", []))

        parts = []
        if source:      parts.append(f"Văn bản: {source}")
        if chapter:     parts.append(f"Chương: {chapter}")
        if section:     parts.append(f"Mục: {section}")
        if art_title:   parts.append(f"Điều khoản cha: {art_title}")
        if target:      parts.append(f"Nội dung sửa đổi: {target}")
        if tags:        parts.append(f"Chủ đề: {tags}")

        return "\n".join(parts)

    # ── LLM Call ──────────────────────────────────────────────────────────────

    def _chat(
        self,
        system_prompt: str,
        user_prompt: str,
        guided_json: Optional[dict] = None,
    ) -> str:
        """
        Gọi vLLM với OpenAI-compatible API.
        - guided_json: schema để vLLM đảm bảo output luôn valid JSON
        - enable_thinking: Qwen3 thinking mode (chính xác hơn, chậm hơn)
        """
        extra_body: dict = {}

        # Qwen3 thinking mode
        if not self.enable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        # Guided JSON decoding (vLLM outlines backend)
        if guided_json:
            extra_body["guided_json"] = guided_json

        for attempt in range(1, self.retry_attempts + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=4096,
                    extra_body=extra_body or None,
                )
                return resp.choices[0].message.content.strip()

            except Exception as exc:
                if attempt == self.retry_attempts:
                    raise
                print(f"  [WARN] LLM call failed (attempt {attempt}/{self.retry_attempts}): {exc}")
                time.sleep(self.retry_delay * attempt)

        return "{}"  # unreachable

    # ── Step 1: Extraction ────────────────────────────────────────────────────

    def extract_from_chunk(self, chunk: dict) -> dict:
        """
        Trích xuất entities + relations từ một chunk.
        Sử dụng cả full_text lẫn clause array (nếu có) để context phong phú hơn.
        """
        content    = chunk.get("content", {})
        full_text  = content.get("full_text", "")
        clauses    = content.get("clause", [])

        # Ưu tiên full_text; nếu quá ngắn thì ghép thêm clauses
        text = full_text
        if clauses and len(full_text) < 200:
            clause_texts = "\n".join(
                f"  Khoản {c.get('clause_id', '')}: {c.get('text', '')}"
                for c in clauses
            )
            text = full_text + "\n\nChi tiết các khoản:\n" + clause_texts

        ctx_header = self._build_context_header(chunk)

        user_prompt = (
            f"=== THÔNG TIN VĂN BẢN ===\n{ctx_header}\n\n"
            f"=== NỘI DUNG CẦN TRÍCH XUẤT ===\n{text}\n\n"
            "Hãy trích xuất tất cả thực thể và mối quan hệ quan trọng trong văn bản trên."
        )

        try:
            raw    = self._chat(self._SYS_EXTRACT, user_prompt, guided_json=ENTITY_GRAPH_SCHEMA)
            result = json.loads(raw)
            return result
        except (json.JSONDecodeError, Exception) as exc:
            print(f"  [WARN] Extract parse error: {exc}")
            return {"entities": [], "relations": []}

    # ── Step 2: Entity Resolution (BookRAG §4) ────────────────────────────────

    def resolve_entities(self) -> Dict[str, str]:
        """
        Phát hiện và hợp nhất coreferent entities.
        VD: 'Bộ GD&ĐT' == 'Bộ Giáo dục và Đào tạo'

        Trả về dict: alias_eid → canonical_eid
        """
        if len(self.entities) < 2:
            return {}

        entity_list = [
            {"id": eid, "name": e["name"], "type": e["type"]}
            for eid, e in self.entities.items()
        ]

        user_prompt = (
            f"Danh sách {len(entity_list)} thực thể đã trích xuất:\n"
            f"{json.dumps(entity_list, ensure_ascii=False, indent=2)}\n\n"
            "Tìm các nhóm thực thể cùng chỉ một đối tượng thực tế và trả về merge_groups.\n"
            "Ví dụ: ['Bộ GD&ĐT', 'Bộ Giáo dục', 'Bộ Giáo dục và Đào tạo'] "
            "→ canonical = 'Bộ Giáo dục và Đào tạo'"
        )

        alias_map: Dict[str, str] = {}
        try:
            raw    = self._chat(self._SYS_RESOLVE, user_prompt, guided_json=ENTITY_RESOLUTION_SCHEMA)
            result = json.loads(raw)

            for group in result.get("merge_groups", []):
                canonical_name = group.get("canonical", "").strip()
                aliases        = group.get("aliases", [])
                if not canonical_name or not aliases:
                    continue

                canonical_eid = f"ent:{self.slugify(canonical_name)}"

                # Đảm bảo canonical tồn tại trong entities dict
                if canonical_eid not in self.entities:
                    # Lấy thông tin từ alias đầu tiên tìm được
                    for alias_name in aliases:
                        src_eid = f"ent:{self.slugify(alias_name)}"
                        if src_eid in self.entities:
                            self.entities[canonical_eid] = {
                                **self.entities[src_eid],
                                "id":   canonical_eid,
                                "name": canonical_name,
                            }
                            break

                for alias_name in aliases:
                    alias_eid = f"ent:{self.slugify(alias_name)}"
                    if alias_eid != canonical_eid and alias_eid in self.entities:
                        alias_map[alias_eid] = canonical_eid

        except Exception as exc:
            print(f"  [WARN] Entity resolution error: {exc}")

        return alias_map

    def _apply_resolution(self, alias_map: Dict[str, str]) -> None:
        """Áp dụng alias_map: xóa duplicates, update relations và links."""
        if not alias_map:
            return

        def resolve(eid: str) -> str:
            return alias_map.get(eid, eid)

        # Xóa alias entities
        for alias_eid in alias_map:
            self.entities.pop(alias_eid, None)

        # Deduplicate relations
        seen_rels: set = set()
        new_rels: List[Dict] = []
        for rel in self.relations:
            src = resolve(rel["source"])
            tgt = resolve(rel["target"])
            key = (src, tgt, rel["relation"])
            if src != tgt and key not in seen_rels:
                new_rels.append({"source": src, "target": tgt, "relation": rel["relation"]})
                seen_rels.add(key)
        self.relations = new_rels

        # Deduplicate node_entity_links
        seen_links: set = set()
        new_links: List[Dict] = []
        for link in self.node_entity_links:
            eid = resolve(link["entity"])
            key = (link["tree_node"], eid, link["relation"])
            if key not in seen_links:
                new_links.append({
                    "tree_node": link["tree_node"],
                    "entity":    eid,
                    "relation":  link["relation"],
                })
                seen_links.add(key)
        self.node_entity_links = new_links

    # ── Main Pipeline ─────────────────────────────────────────────────────────

    def process_all(self, limit: Optional[int] = None) -> None:
        """
        Đọc toàn bộ file JSON trong data_dir, extract entities + relations.

        Args:
            limit: Giới hạn số chunk xử lý (None = tất cả)
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"data_dir không tồn tại: {self.data_dir}")

        json_files = sorted(self.data_dir.glob("*.json"))
        if not json_files:
            print(f"[WARN] Không tìm thấy file JSON trong: {self.data_dir}")
            return

        print(f"📂 Tìm thấy {len(json_files)} file JSON.")
        count      = 0
        skipped    = 0
        t_start    = time.time()

        for file_path in json_files:
            print(f"\n📄 Đang xử lý: {file_path.name}")

            with open(file_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            if not isinstance(chunks, list):
                print(f"  [SKIP] File không phải list: {file_path.name}")
                continue

            for chunk in chunks:
                if limit is not None and count >= limit:
                    break

                # ── Lọc chunk không đủ điều kiện ──────────────────────────
                content  = chunk.get("content", {})
                full_text = content.get("full_text", "")
                if len(full_text.strip()) < self.min_text_length:
                    skipped += 1
                    continue

                tree_node_id = self._build_tree_node_id(chunk)
                meta         = chunk.get("metadata", {})

                print(
                    f"  [{count + 1}] {tree_node_id}  "
                    f"| {meta.get('target_article', meta.get('article_title', ''))[:40]}"
                    f"  ({len(full_text)} chars)"
                )

                extracted = self.extract_from_chunk(chunk)

                # ── Lưu entities ───────────────────────────────────────────
                for ent in extracted.get("entities", []):
                    ename = ent.get("name", "").strip()
                    if not ename:
                        continue
                    eid = f"ent:{self.slugify(ename)}"

                    # Giữ description dài hơn
                    existing = self.entities.get(eid)
                    if not existing or len(ent.get("description", "")) > len(existing.get("description", "")):
                        self.entities[eid] = {
                            "id":          eid,
                            "name":        ename,
                            "type":        ent.get("type", "Entity"),
                            "description": ent.get("description", ""),
                        }

                    self.node_entity_links.append({
                        "tree_node": tree_node_id,
                        "entity":    eid,
                        "relation":  "MENTIONS",
                    })

                # ── Lưu relations ──────────────────────────────────────────
                for rel in extracted.get("relations", []):
                    src_name = rel.get("source", "").strip()
                    tgt_name = rel.get("target", "").strip()
                    relation  = rel.get("relation", "RELATED_TO").strip()
                    if src_name and tgt_name and relation:
                        self.relations.append({
                            "source":   f"ent:{self.slugify(src_name)}",
                            "target":   f"ent:{self.slugify(tgt_name)}",
                            "relation": relation,
                        })

                count += 1

            if limit is not None and count >= limit:
                break

        elapsed = time.time() - t_start
        print(
            f"\n✅ Hoàn thành: {count} chunks ({skipped} skipped) "
            f"trong {elapsed:.1f}s\n"
            f"   Raw: {len(self.entities)} entities | {len(self.relations)} relations"
        )

        # ── Entity Resolution (BookRAG §4) ─────────────────────────────────
        if self.entity_resolution and len(self.entities) >= 2:
            print("\n🔗 Đang chạy Entity Resolution (BookRAG gradient-based merge)...")
            alias_map = self.resolve_entities()
            self._apply_resolution(alias_map)
            print(
                f"   Merged {len(alias_map)} entity aliases\n"
                f"   Final: {len(self.entities)} entities | {len(self.relations)} relations"
            )

    def save(self) -> None:
        """Ghi kết quả ra file JSON."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "bookrag_entity_graph": {
                "metadata": {
                    "model":            self.model_name,
                    "total_entities":   len(self.entities),
                    "total_relations":  len(self.relations),
                    "total_mappings":   len(self.node_entity_links),
                },
                "entities": list(self.entities.values()),
                "relations": self.relations,
                "mappings":  self.node_entity_links,
            }
        }

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(
            f"\n💾 Đã lưu: {self.output_path}\n"
            f"   Entities: {len(self.entities)}\n"
            f"   Relations: {len(self.relations)}\n"
            f"   Mappings (node→entity): {len(self.node_entity_links)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    builder = EntityGraphBuilder(
        data_dir    = "d:/legal-edu-chatbot/data/final",
        output_path = "d:/legal-edu-chatbot/data/entity_graph.json",

        vllm_base_url   = "http://localhost:8000/v1",
        model_name      = "Qwen/Qwen3-30B-A3B-Instruct-2507",

        enable_thinking  = False,   # True → chính xác hơn, chậm hơn ~3x
        entity_resolution = True,   # BookRAG entity merge

        min_text_length = 30,       # Bỏ qua chunk quá ngắn (VD: header rỗng)
        retry_attempts  = 3,
        retry_delay     = 2.0,
    )

    print("🚀 BookRAG Entity Graph Builder")
    print(f"   Model : {builder.model_name}")
    print(f"   Data  : {builder.data_dir}")
    print(f"   Output: {builder.output_path}\n")

    # Đặt limit=5 để test nhanh; bỏ limit (hoặc limit=None) để chạy toàn bộ
    builder.process_all(limit=5)
    builder.save()
    print("\n✨ Xây dựng thành công.")