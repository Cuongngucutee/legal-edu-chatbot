# HƯỚNG DẪN CHẠY TRÊN GOOGLE COLAB (SỬ DỤNG GPU A100 80GB)
# =========================================================================================
# BƯỚC 1: Chọn Runtime (Mô trường thực thi)
#   Trên menu Colab: Runtime -> Change runtime type -> Hardware accelerator: A100 GPU
#
# BƯỚC 2: Kết nối Google Drive (Để lưu trữ dữ liệu JSON Data đầu vào & Graph đầu ra)
#   Tạo một Code cell mới và chạy:
#   from google.colab import drive
#   drive.mount('/content/drive')
#
# BƯỚC 3: Cài đặt thư viện vLLM (Engine chạy LLM Offline, Không thông qua API Server)
#   Tạo một Code cell mới và chạy:
#   !pip install vllm outlines
#
# BƯỚC 4: Upload thư mục data/ của bạn lên Drive (VD: /content/drive/MyDrive/GraphRAG/data/)
#
# BƯỚC 5: Copy nguyên toàn bộ mã Python bên dưới, paste vào 1 Code cell và chạy "Run".
#   Code này sử dụng trực tiếp engine `vllm.LLM` offline (không gọi API)
#   nhờ vậy bảo mật dữ liệu tuyệt đối 100% trong Colab.
# =========================================================================================

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

# Cấu hình multiprocessing cho vLLM trên môi trường Colab
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Import Engine LLM trực tiếp (Offline Mode)
from vllm import LLM, SamplingParams
try:
    from vllm.sampling_params import GuidedDecodingParams
except ImportError:
    GuidedDecodingParams = None

# =========================================================================================
# CẤU HÌNH (CONFIG)
# =========================================================================================
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# Schema ràng buộc LLM sinh ra đúng định dạng JSON
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
                            "Organization", "Concept", "Role", "Document", 
                            "EducationLevel", "Policy", "Entity"
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

# =========================================================================================
# TRÌNH XÂY DỰNG ĐỒ THỊ (BUILDER)
# =========================================================================================
class EntityGraphBuilder:
    _SYS_EXTRACT = (
        "Bạn là chuyên gia pháp lý Việt Nam và chuyên gia xây dựng Knowledge Graph.\n"
        "Nhiệm vụ: Trích xuất Thực thể (Entity) và Mối quan hệ (Relation) từ văn bản luật\n"
        "Quy tắc BẮT BUỘC:\n"
        "1. Chỉ trích xuất thực thể THỰC SỰ xuất hiện hoặc được đề cập trong văn bản.\n"
        "2. Tên thực thể phải ĐẦY ĐỦ, không viết tắt. VD: 'Bộ Giáo dục và Đào tạo'.\n"
        "3. Quan hệ dùng động từ VIẾT_HOA_GẠCHDưới. VD: QUẢN_LÝ, BAN_HÀNH, QUY_ĐỊNH.\n"
        "4. Trả về JSON hợp lệ theo schema — KHÔNG giải thích thêm.\n"
        "5. GIỚI HẠN: Trích xuất tối đa 40 thực thể cốt lõi nhất và các mối quan hệ quan trọng nhất (để tránh giới hạn token đầu ra)."
    )

    _SYS_RESOLVE = (
        "Bạn là chuyên gia Entity Resolution cho Knowledge Graph pháp lý.\n"
        "Nhiệm vụ: Tìm các nhóm thực thể GIỐNG NHAU (cùng chỉ một đối tượng thực tế) "
        "và gom thành merge_groups để hợp nhất node.\n"
        "Quy tắc:\n"
        "- Chỉ gom khi CHẮC CHẮN là cùng thực thể.\n"
        "- 'canonical' là tên ĐẦY ĐỦ nhất, 'aliases' là các tên viết tắt / khác.\n"
        "- Trả về JSON hợp lệ."
    )

    def __init__(
        self,
        data_dir: str,
        output_path: str,
        model_name: str           = MODEL_NAME,
        enable_thinking: bool     = False,
        entity_resolution: bool   = True,
        min_text_length: int      = 30,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int        = 32768,
    ):
        self.model_name        = model_name
        self.enable_thinking   = enable_thinking
        self.entity_resolution = entity_resolution
        self.min_text_length   = min_text_length

        self.data_dir    = Path(data_dir)
        self.output_path = Path(output_path)

        self.entities: Dict[str, Dict]     = {}
        self.relations: List[Dict]          = []
        self.node_entity_links: List[Dict]  = []

        print(f"⏳ Tải Model Offline {self.model_name} vào VRAM (Sẽ tốn ít phút)...")
        # Khởi tạo trực tiếp Engine của vLLM trên Colab (Không dùng API Http)
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        # Sử dụng tokenizer tích hợp
        self.tokenizer = self.llm.get_tokenizer()
        print("✅ Model LLM Offline đã được load vào A100 VRAM!")

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

    def _chat_offline(self, system_prompt: str, user_prompt: str, guided_json: Optional[dict] = None) -> str:
        import inspect
        # Nếu vLLM phiên bản trên Colab không hỗ trợ guided_decoding trong SamplingParams
        # Ta sẽ nối thẳng JSON Schema vào Prompt để ép Qwen3 trả về đúng định dạng.
        has_guided = "guided_decoding" in inspect.signature(SamplingParams).parameters

        if guided_json and not has_guided:
            system_prompt += (
                f"\n\nBẮT BUỘC TRẢ VỀ CHÍNH XÁC THEO JSON SCHEMA SAU MÀ KHÔNG GIẢI THÍCH (KHÔNG DÙNG ```json):\n"
                f"{json.dumps(guided_json, ensure_ascii=False)}"
            )

        # Áp dụng template Chat vào prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        guided_decoding = None
        if guided_json and has_guided and GuidedDecodingParams is not None:
            guided_decoding = GuidedDecodingParams(json=json.dumps(guided_json))

        if has_guided:
            sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=8192,
                guided_decoding=guided_decoding,
            )
        else:
            sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=8192,
            )

        # Suy luận hoàn toàn offline không gọi API
        outputs = self.llm.generate(
            prompts=[prompt_text],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        return outputs[0].outputs[0].text.strip()

    def extract_from_chunk(self, chunk: dict) -> dict:
        full_text = chunk.get("content", {}).get("full_text", "")
        clauses = chunk.get("content", {}).get("clause", [])

        text = full_text
        if clauses and len(full_text) < 200:
            clause_texts = "\n".join(f"Khoản {c.get('clause_id')}: {c.get('text')}" for c in clauses)
            text = full_text + "\nChi tiết:\n" + clause_texts

        user_prompt = f"=== NỘI DUNG ===\n{text}\n\nHãy trích xuất tất cả thực thể và mối quan hệ quan trọng theo JSON Array."

        try:
            raw = self._chat_offline(self._SYS_EXTRACT, user_prompt, guided_json=ENTITY_GRAPH_SCHEMA)
            # vLLM/Outlines đôi khi sinh text JSON thêm bọc ```json, cần làm sạch
            if raw.startswith("```json"): raw = raw[7:]
            if raw.endswith("```"): raw = raw[:-3]
            return json.loads(raw.strip())
        except Exception as exc:
            print(f"  [WARN] Extract error: {exc}")
            return {"entities": [], "relations": []}

    def process_all(self, limit: Optional[int] = None) -> None:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Thư mục không tồn tại: {self.data_dir}")

        json_files = sorted(self.data_dir.glob("*.json"))
        count = 0
        t_start = time.time()

        for file_path in json_files:
            print(f"\n📄 Đang xử lý: {file_path.name}")
            with open(file_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            for chunk in chunks:
                if limit and count >= limit: break
                
                full_text = chunk.get("content", {}).get("full_text", "")
                if len(full_text.strip()) < self.min_text_length:
                    continue

                tree_node_id = self._build_tree_node_id(chunk)
                print(f"  -> Trích xuất: {tree_node_id} ({len(full_text)} chars)")
                
                extracted = self.extract_from_chunk(chunk)

                for ent in extracted.get("entities", []):
                    ename = ent.get("name", "").strip()
                    if not ename: continue
                    eid = f"ent:{self.slugify(ename)}"
                    self.entities[eid] = {
                        "id": eid, "name": ename,
                        "type": ent.get("type", "Entity"),
                        "description": ent.get("description", "")
                    }
                    self.node_entity_links.append({"tree_node": tree_node_id, "entity": eid, "relation": "MENTIONS"})

                for rel in extracted.get("relations", []):
                    src_name = rel.get("source", "").strip()
                    tgt_name = rel.get("target", "").strip()
                    if src_name and tgt_name:
                        self.relations.append({
                            "source": f"ent:{self.slugify(src_name)}",
                            "target": f"ent:{self.slugify(tgt_name)}",
                            "relation": rel.get("relation", "RELATED_TO").strip(),
                        })

                count += 1
            if limit and count >= limit: break

        print(f"\n✅ Hoàn thành: {count} chunks trong {time.time() - t_start:.1f}s")
        print(f"   Entities: {len(self.entities)} | Relations: {len(self.relations)}")

    def save(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "bookrag_entity_graph": {
                "metadata": {
                    "model": self.model_name,
                    "total_entities": len(self.entities),
                    "total_relations": len(self.relations),
                    "total_mappings": len(self.node_entity_links),
                },
                "entities": list(self.entities.values()),
                "relations": self.relations,
                "mappings": self.node_entity_links,
            }
        }
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Đã lưu JSON tại: {self.output_path}")

# =========================================================================================
# CHẠY (Thực thi trong Colab Cell khi Run)
# =========================================================================================
if __name__ == "__main__":
    # Thay đường dẫn sau thành đúng đường dẫn Drive của bạn trong Colab
    builder = EntityGraphBuilder(
        data_dir    = "/content/drive/MyDrive/GraphRAG/data",
        output_path = "/content/drive/MyDrive/GraphRAG/outputs/entity_graph.json",
        model_name  = "Qwen/Qwen3-30B-A3B-Instruct-2507",
        
        # A100 80GB specs
        tensor_parallel_size   = 1,
        gpu_memory_utilization = 0.9,
        max_model_len          = 32768, 
        
        enable_thinking   = False,
        entity_resolution = False, # Bật nếu muốn gộp entity sau khi trích xuất
    )

    print("\n🚀 Bắt đầu Trích xuất Thực thể Bằng LLM Offline...")
    # Xoá đối số limit=5 để xử lý toàn bộ văn bản
    builder.process_all(limit=5)
    builder.save()
    print("✨ Quá trình hoàn tất.")
