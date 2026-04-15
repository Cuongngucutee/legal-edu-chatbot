import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM, SamplingParams
try:
    from vllm.sampling_params import GuidedDecodingParams
except ImportError:
    GuidedDecodingParams = None

# =========================================================================================
# CẤU HÌNH (CONFIG)
# =========================================================================================
MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"

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
                        "enum": ["Organization","Concept","Role","Document",
                                 "EducationLevel","Policy","Entity"]
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

# =========================================================================================
# TRÌNH XÂY DỰNG ĐỒ THỊ — tối ưu batch inference
# =========================================================================================
class EntityGraphBuilder:
    _SYS_EXTRACT = (
        "Bạn là chuyên gia pháp lý Việt Nam và chuyên gia xây dựng Knowledge Graph.\n"
        "Nhiệm vụ: Trích xuất Thực thể (Entity) và Mối quan hệ (Relation) từ văn bản luật\n"
        "Quy tắc BẮT BUỘC:\n"
        "1. Chỉ trích xuất thực thể THỰC SỰ xuất hiện hoặc được đề cập trong văn bản.\n"
        "2. Tên thực thể phải ĐẦY ĐỦ, không viết tắt. VD: 'Bộ Giáo dục và Đào tạo'.\n"
        "3. Quan hệ dùng động từ VIẾT_HOA_GẠCH_DƯỚI. VD: QUẢN_LÝ, BAN_HÀNH, QUY_ĐỊNH.\n"
        "4. Trả về JSON hợp lệ theo schema — KHÔNG giải thích thêm."
    )

    _SYS_EXTRACT_CAN_CU = (
        "Bạn là chuyên gia pháp lý Việt Nam và chuyên gia xây dựng Knowledge Graph.\n"
        "Nhiệm vụ: Trích xuất Thực thể và Mối quan hệ từ phần CĂN CỨ PHÁP LÝ của văn bản.\n"
        "Quy tắc BẮT BUỘC:\n"
        "1. Mỗi văn bản pháp luật được viện dẫn (Luật, Nghị định, Quyết định, Thông tư...) "
        "là một thực thể type='Document'.\n"
        "2. Cơ quan ban hành (Chính phủ, Bộ GD&ĐT...) là thực thể type='Organization'.\n"
        "3. Quan hệ giữa văn bản hiện tại và văn bản viện dẫn: CĂN_CỨ_THEO.\n"
        "4. Quan hệ giữa cơ quan và văn bản: BAN_HÀNH, ĐỀ_NGHỊ.\n"
        "5. Tên thực thể phải ĐẦY ĐỦ, bao gồm số hiệu nếu có.\n"
        "6. Trả về JSON hợp lệ theo schema — KHÔNG giải thích thêm."
    )

    _SYS_RESOLVE = (
        "Bạn là chuyên gia Entity Resolution cho Knowledge Graph pháp lý.\n"
        "Nhiệm vụ: Tìm các nhóm thực thể GIỐNG NHAU (cùng chỉ một đối tượng thực tế) "
        "và gom thành merge_groups để hợp nhất node.\n"
        "Quy tắc:\n"
        "- Chỉ gom khi CHẮC CHẮN là cùng thực thể. VD: 'Bộ GD&ĐT' và 'Bộ Giáo dục và Đào tạo'.\n"
        "- KHÔNG gom các thực thể chỉ có quan hệ (VD: 'Luật Giáo dục' và 'Luật Giáo dục đại học' là KHÁC nhau).\n"
        "- 'canonical' là tên ĐẦY ĐỦ nhất, 'aliases' là các tên viết tắt / khác.\n"
        "- Nếu không có nhóm nào cần gom, trả về {\"merge_groups\": []}.\n"
        "- Trả về JSON hợp lệ."
    )

    def __init__(
        self,
        data_dir: str,
        output_path: str,
        model_name: str               = MODEL_NAME,
        enable_thinking: bool         = False,
        entity_resolution: bool       = True,
        min_text_length: int          = 30,
        tensor_parallel_size: int     = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int            = 32768,
        batch_size: int               = 64,    # Tối ưu cho RTX 6000 96GB VRAM
        resolution_batch_size: int    = 200,   # Số entity mỗi batch resolution
    ):
        self.model_name        = model_name
        self.enable_thinking   = enable_thinking
        self.entity_resolution = entity_resolution
        self.min_text_length   = min_text_length
        self.batch_size        = batch_size
        self.resolution_batch_size = resolution_batch_size

        self.data_dir    = Path(data_dir)
        self.output_path = Path(output_path)

        self.entities: Dict[str, Dict]    = {}
        self.relations: List[Dict]        = []
        self.node_entity_links: List[Dict] = []

        # ← FIX VĐ4: Set để dedup relations
        self._seen_relations: Set[Tuple[str, str, str]] = set()

        print(f"⏳ Tải Model Offline {self.model_name} vào VRAM...")
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

        # Chuẩn bị SamplingParams một lần duy nhất
        import inspect
        self._has_guided = "guided_decoding" in inspect.signature(SamplingParams).parameters
        self._sampling_extract  = self._make_sampling_params(ENTITY_GRAPH_SCHEMA)
        self._sampling_resolve  = self._make_sampling_params(ENTITY_RESOLUTION_SCHEMA)
        print("✅ Model LLM Offline đã được load vào VRAM!")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def slugify(text: str) -> str:
        text = str(text).lower().strip()
        text = re.sub(r'\s+', '_', text)
        text = re.sub(r'[^\w_]', '', text, flags=re.UNICODE)
        return text.strip('_') or "unknown"

    def _build_tree_node_id(self, chunk: dict) -> str:
        chunk_id = chunk.get("id", "").strip()
        if chunk_id:
            return f"node:{chunk_id}"
        meta     = chunk.get("metadata", {})
        source   = meta.get("source", "unknown")
        art_num  = str(meta.get("article_number", "")).strip()
        doc_slug = self.slugify(source)
        return f"node:{doc_slug}:d{art_num}"

    def _make_sampling_params(self, guided_json: Optional[dict] = None) -> SamplingParams:
        """Tạo SamplingParams một lần, tái sử dụng cho toàn bộ batch."""
        if guided_json and self._has_guided and GuidedDecodingParams is not None:
            guided = GuidedDecodingParams(json=json.dumps(guided_json))
            return SamplingParams(temperature=0.1, max_tokens=32768, guided_decoding=guided)
        return SamplingParams(temperature=0.1, max_tokens=32768)

    def _build_prompt(self, system_prompt: str, user_prompt: str,
                      guided_json: Optional[dict] = None) -> str:
        """Tạo prompt text từ template chat."""
        sys = system_prompt
        if guided_json and not self._has_guided:
            sys += (
                f"\n\nBẮT BUỘC TRẢ VỀ CHÍNH XÁC THEO JSON SCHEMA SAU MÀ KHÔNG GIẢI THÍCH "
                f"(KHÔNG DÙNG ```json):\n{json.dumps(guided_json, ensure_ascii=False)}"
            )
        messages = [
            {"role": "system", "content": sys},
            {"role": "user",   "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @staticmethod
    def _clean_json(raw: str) -> str:
        """Loại bỏ ```json fence nếu vLLM sinh ra."""
        raw = raw.strip()
        # Xử lý thinking block nếu có
        think_end = raw.find("</think>")
        if think_end != -1:
            raw = raw[think_end + 8:].strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        return raw.strip()

    # ------------------------------------------------------------------
    # FIX VĐ2 + VĐ3: Xây dựng prompt phù hợp với cấu trúc data thực tế
    # ------------------------------------------------------------------
    def _build_user_prompt(self, chunk: dict) -> str:
        """
        Tạo user prompt dựa trên loại chunk:
        - CAN_CU: prompt chuyên biệt cho phần căn cứ pháp lý
        - Điều luật thông thường: bao gồm metadata + full_text + clause + point
        """
        meta = chunk.get("metadata", {})
        content = chunk.get("content", {})
        full_text = content.get("full_text", "")
        is_can_cu = meta.get("is_base", False) or meta.get("article_number") == "CAN_CU"

        # --- Chunk CAN_CU (căn cứ pháp lý) ---
        if is_can_cu:
            source = meta.get("source", "Văn bản")
            return (
                f"=== CĂN CỨ PHÁP LÝ CỦA: {source} ===\n"
                f"{full_text}\n\n"
                "Hãy trích xuất:\n"
                "1. Tất cả các văn bản pháp luật được viện dẫn (Luật, Nghị định, Quyết định, Thông tư...) "
                "làm thực thể type='Document'.\n"
                "2. Các cơ quan ban hành/đề nghị làm thực thể type='Organization'.\n"
                f"3. Quan hệ CĂN_CỨ_THEO giữa '{source}' và từng văn bản viện dẫn.\n"
                "4. Quan hệ BAN_HÀNH hoặc ĐỀ_NGHỊ giữa cơ quan và văn bản.\n"
                "Trả về JSON theo schema."
            )

        # --- Chunk Điều luật thông thường ---
        # Metadata context giúp LLM hiểu ngữ cảnh tốt hơn
        context_parts = []
        if meta.get("source"):
            context_parts.append(f"Nguồn: {meta['source']}")
        if meta.get("chapter"):
            context_parts.append(meta["chapter"])
        if meta.get("section"):
            context_parts.append(meta["section"])
        if meta.get("article_title"):
            context_parts.append(meta["article_title"])
        context_line = " | ".join(context_parts) if context_parts else ""

        # Xây dựng nội dung chi tiết bao gồm clause + point
        clauses = content.get("clause", [])
        detail_parts = []

        if clauses:
            for c in clauses:
                clause_header = f"Khoản {c.get('clause_id', '?')}: {c.get('text', '')}"
                detail_parts.append(clause_header)

                # ← FIX VĐ3: Bổ sung point (điểm a, b, c...)
                points = c.get("point", [])
                if points:
                    for p in points:
                        detail_parts.append(
                            f"  Điểm {p.get('label', '?')}: {p.get('text', '')}"
                        )

        # Quyết định dùng full_text hay structured detail
        # full_text thường đã chứa toàn bộ nội dung nhưng không có cấu trúc
        # structured detail giúp LLM phân tích tốt hơn
        if detail_parts:
            structured_text = "\n".join(detail_parts)
            # Dùng structured nếu nó bổ sung thêm thông tin so với full_text
            text = (
                f"{full_text}\n\n"
                f"=== CẤU TRÚC CHI TIẾT ===\n{structured_text}"
            )
        else:
            text = full_text

        prompt = ""
        if context_line:
            prompt += f"=== METADATA ===\n{context_line}\n\n"
        prompt += (
            f"=== NỘI DUNG ===\n{text}\n\n"
            "Hãy trích xuất tất cả thực thể và mối quan hệ quan trọng theo JSON."
        )
        return prompt

    def _extract_batch(self, chunks: List[dict]) -> List[dict]:
        """
        Gửi toàn bộ batch vào llm.generate() một lần duy nhất.
        vLLM tự động lên lịch song song trên GPU → tận dụng tối đa RTX 6000.

        FIX VĐ2: Chunk CAN_CU dùng system prompt riêng (_SYS_EXTRACT_CAN_CU).
        """
        prompts = []
        for c in chunks:
            meta = c.get("metadata", {})
            is_can_cu = meta.get("is_base", False) or meta.get("article_number") == "CAN_CU"
            sys_prompt = self._SYS_EXTRACT_CAN_CU if is_can_cu else self._SYS_EXTRACT

            prompts.append(
                self._build_prompt(sys_prompt, self._build_user_prompt(c),
                                   guided_json=ENTITY_GRAPH_SCHEMA)
            )

        outputs = self.llm.generate(
            prompts=prompts,
            sampling_params=self._sampling_extract,
            use_tqdm=True,
        )
        results = []
        for out in outputs:
            raw = out.outputs[0].text
            try:
                results.append(json.loads(self._clean_json(raw)))
            except Exception as exc:
                print(f"  [WARN] Parse error: {exc}")
                results.append({"entities": [], "relations": []})
        return results

    # ------------------------------------------------------------------
    # FIX VĐ7: Entity Resolution — gom nhóm thực thể trùng lặp
    # ------------------------------------------------------------------
    def _resolve_entities_batch(self) -> None:
        """
        Gọi LLM để tìm và gom nhóm các entity trùng lặp/alias.
        Xử lý theo batch nhỏ vì danh sách entity có thể rất lớn.
        """
        if not self.entity_resolution:
            return

        entity_list = list(self.entities.values())
        if len(entity_list) < 5:
            print("  ⏭️  Quá ít entity, bỏ qua Entity Resolution.")
            return

        print(f"\n🔗 Bắt đầu Entity Resolution cho {len(entity_list)} entities...")

        all_merge_groups = []

        # Chia entity thành các batch nhỏ để LLM xử lý được
        for batch_start in range(0, len(entity_list), self.resolution_batch_size):
            batch_end = min(batch_start + self.resolution_batch_size, len(entity_list))
            batch_entities = entity_list[batch_start:batch_end]

            print(f"  📦 Resolution batch [{batch_start+1}–{batch_end}/{len(entity_list)}]")

            # Tạo danh sách entity với type để LLM hiểu context
            entity_lines = "\n".join(
                f"- {e['name']} (type={e['type']})"
                for e in batch_entities
            )

            user_prompt = (
                f"Dưới đây là danh sách {len(batch_entities)} thực thể đã trích xuất từ "
                "các văn bản pháp luật giáo dục Việt Nam.\n\n"
                f"{entity_lines}\n\n"
                "Hãy tìm các nhóm thực thể GIỐNG NHAU và gom lại.\n"
                "VD: 'Bộ GD&ĐT', 'Bộ Giáo dục và Đào tạo', 'Bộ GDĐT' → cùng 1 thực thể.\n"
                "VD: 'ĐHQG', 'Đại học quốc gia' → cùng 1 thực thể.\n"
                "CHÚ Ý: 'Luật Giáo dục' và 'Luật Giáo dục đại học' là KHÁC nhau, KHÔNG gom.\n"
                "Trả về JSON theo schema."
            )

            prompt = self._build_prompt(
                self._SYS_RESOLVE, user_prompt,
                guided_json=ENTITY_RESOLUTION_SCHEMA,
            )

            outputs = self.llm.generate(
                prompts=[prompt],
                sampling_params=self._sampling_resolve,
                use_tqdm=False,
            )

            raw = outputs[0].outputs[0].text
            try:
                result = json.loads(self._clean_json(raw))
                groups = result.get("merge_groups", [])
                all_merge_groups.extend(groups)
                print(f"    → Tìm thấy {len(groups)} nhóm trùng lặp")
            except Exception as exc:
                print(f"    [WARN] Resolution parse error: {exc}")

        # Áp dụng merge
        if all_merge_groups:
            self._apply_merge_groups(all_merge_groups)

    def _apply_merge_groups(self, merge_groups: List[dict]) -> None:
        """Áp dụng merge: thay thế alias bằng canonical trong entities, relations, links."""
        # Xây dựng bảng mapping alias → canonical
        alias_to_canonical: Dict[str, str] = {}
        for group in merge_groups:
            canonical = group.get("canonical", "").strip()
            aliases = group.get("aliases", [])
            if not canonical or not aliases:
                continue

            canonical_id = f"ent:{self.slugify(canonical)}"
            for alias in aliases:
                alias = alias.strip()
                if not alias or alias == canonical:
                    continue
                alias_id = f"ent:{self.slugify(alias)}"
                if alias_id != canonical_id:
                    alias_to_canonical[alias_id] = canonical_id

        if not alias_to_canonical:
            print("  ℹ️  Không có entity nào cần merge.")
            return

        print(f"  🔀 Merging {len(alias_to_canonical)} aliases vào canonical entities...")

        # Merge entity records: giữ canonical, xóa alias
        for alias_id, canonical_id in alias_to_canonical.items():
            alias_ent = self.entities.pop(alias_id, None)
            if alias_ent and canonical_id in self.entities:
                # Giữ description dài hơn
                canon_ent = self.entities[canonical_id]
                if len(alias_ent.get("description", "")) > len(canon_ent.get("description", "")):
                    canon_ent["description"] = alias_ent["description"]

        # Cập nhật relations: thay thế alias → canonical
        updated_relations = []
        seen: Set[Tuple[str, str, str]] = set()
        for rel in self.relations:
            src = alias_to_canonical.get(rel["source"], rel["source"])
            tgt = alias_to_canonical.get(rel["target"], rel["target"])
            rel_type = rel["relation"]
            key = (src, tgt, rel_type)
            if key not in seen and src != tgt:  # loại self-loop sau merge
                seen.add(key)
                updated_relations.append({
                    "source": src, "target": tgt, "relation": rel_type
                })
        self.relations = updated_relations
        self._seen_relations = seen

        # Cập nhật node_entity_links
        for link in self.node_entity_links:
            eid = link["entity"]
            if eid in alias_to_canonical:
                link["entity"] = alias_to_canonical[eid]

        print(f"  ✅ Sau merge: {len(self.entities)} entities, {len(self.relations)} relations")

    # ------------------------------------------------------------------
    # process_all: gom chunk theo batch_size rồi gọi _extract_batch
    # ------------------------------------------------------------------
    def process_all(self, limit: Optional[int] = None) -> None:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Thư mục không tồn tại: {self.data_dir}")

        # Thu thập tất cả chunk hợp lệ trước
        all_chunks: List[dict] = []
        all_node_ids: List[str] = []

        for file_path in sorted(self.data_dir.glob("*.json")):
            print(f"📄 Đọc file: {file_path.name}")
            with open(file_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            for chunk in chunks:
                full_text = chunk.get("content", {}).get("full_text", "")
                if len(full_text.strip()) < self.min_text_length:
                    continue
                all_chunks.append(chunk)
                all_node_ids.append(self._build_tree_node_id(chunk))
                if limit and len(all_chunks) >= limit:
                    break
            if limit and len(all_chunks) >= limit:
                break

        total   = len(all_chunks)
        t_start = time.time()
        print(f"\n🚀 Bắt đầu trích xuất {total} chunks (batch_size={self.batch_size})...")

        # Xử lý theo từng batch
        for batch_start in range(0, total, self.batch_size):
            batch_end    = min(batch_start + self.batch_size, total)
            batch_chunks = all_chunks[batch_start:batch_end]
            batch_ids    = all_node_ids[batch_start:batch_end]

            print(f"\n  📦 Batch [{batch_start+1}–{batch_end}/{total}]")
            extracted_list = self._extract_batch(batch_chunks)

            for tree_node_id, extracted in zip(batch_ids, extracted_list):
                for ent in extracted.get("entities", []):
                    ename = ent.get("name", "").strip()
                    if not ename:
                        continue
                    eid = f"ent:{self.slugify(ename)}"
                    # Giữ description đầy đủ hơn nếu entity đã tồn tại
                    existing = self.entities.get(eid)
                    if not existing or len(ent.get("description","")) > len(existing.get("description","")):
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

                # ← FIX VĐ4: Dedup relations bằng set
                for rel in extracted.get("relations", []):
                    src_name = rel.get("source", "").strip()
                    tgt_name = rel.get("target", "").strip()
                    if src_name and tgt_name:
                        src_id   = f"ent:{self.slugify(src_name)}"
                        tgt_id   = f"ent:{self.slugify(tgt_name)}"
                        rel_type = rel.get("relation", "RELATED_TO").strip()
                        rel_key  = (src_id, tgt_id, rel_type)

                        if rel_key not in self._seen_relations:
                            self._seen_relations.add(rel_key)
                            self.relations.append({
                                "source":   src_id,
                                "target":   tgt_id,
                                "relation": rel_type,
                            })

        elapsed = time.time() - t_start
        print(f"\n✅ Trích xuất xong: {total} chunks trong {elapsed:.1f}s "
              f"({total/elapsed:.1f} chunks/s)")
        print(f"   Entities: {len(self.entities)} | Relations (unique): {len(self.relations)}")

        # ← FIX VĐ7: Chạy Entity Resolution sau khi trích xuất xong
        if self.entity_resolution:
            self._resolve_entities_batch()

    def save(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "bookrag_entity_graph": {
                "metadata": {
                    "model":          self.model_name,
                    "total_entities": len(self.entities),
                    "total_relations": len(self.relations),
                    "total_mappings": len(self.node_entity_links),
                    "entity_resolution_applied": self.entity_resolution,
                },
                "entities": list(self.entities.values()),
                "relations": self.relations,
                "mappings":  self.node_entity_links,
            }
        }
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Đã lưu JSON tại: {self.output_path}")


# =========================================================================================
# CHẠY
# =========================================================================================
if __name__ == "__main__":
    builder = EntityGraphBuilder(
        data_dir    = "/content/drive/MyDrive/GraphRAG/data",
        output_path = "/content/drive/MyDrive/GraphRAG/outputs/entity_graph.json",
        model_name  = "Qwen/Qwen3.5-35B-A3B",

        tensor_parallel_size   = 1,
        gpu_memory_utilization = 0.9,
        max_model_len          = 32768,

        batch_size             = 64,    # ← FIX VĐ6: Tối ưu cho RTX 6000 96GB
        resolution_batch_size  = 200,   # Số entity mỗi lần gọi resolution
        enable_thinking        = False,
        entity_resolution      = True,  # ← FIX VĐ7: Bật entity resolution
    )

    print("\n🚀 Bắt đầu trích xuất thực thể bằng LLM Offline (Batched)...")
    builder.process_all()
    builder.save()
    print("✨ Quá trình hoàn tất.")