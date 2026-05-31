"""
LawEdu AI — Context Builder.
Formats retrieved documents into context string for LLM generation.
"""
import re
import logging

logger = logging.getLogger(__name__)


def build_context(docs: list, max_chars: int = 16000) -> str:
    """Xây dựng context từ retrieved docs (list[dict] format)."""
    parts = []
    total_chars = 0

    for i, doc in enumerate(docs, 1):
        meta = doc.get("metadata", {})
        breadcrumb = meta.get("breadcrumb", f"{meta.get('doc_id', '')} Điều {meta.get('so_dieu', '')}")

        status_note = ""
        tinh_trang = meta.get("tinh_trang", "con_hieu_luc")
        if tinh_trang == "da_sua_doi":
            status_note = " ⚠️ [Văn bản đã được sửa đổi, bổ sung]"
        elif tinh_trang == "bi_bai_bo":
            status_note = " ❌ [Văn bản đã bị bãi bỏ]"

        content = doc.get("content", "")
        if len(content) > 4000:
            content = content[:4000] + "..."

        ngay_hieu_luc = meta.get("ngay_hieu_luc", "")
        ngay_str = f" (Hiệu lực: {ngay_hieu_luc})" if ngay_hieu_luc else ""

        part = f"[{i}] {breadcrumb}{ngay_str}{status_note}\n{content}"
        part_len = len(part)

        if total_chars + part_len > max_chars:
            break
        parts.append(part)
        total_chars += part_len

    return "\n\n---\n\n".join(parts)


def postprocess_citations(answer: str, docs: list) -> str:
    """Post-process: thay thế trích dẫn [1], [2]... bằng tên VB thực từ sources."""
    if not docs:
        return answer

    # Build mapping: index → document name
    doc_map = {}
    for i, doc in enumerate(docs, 1):
        m = doc.get("metadata", {})
        name = m.get("ten_van_ban") or m.get("so_hieu") or m.get("doc_id", "").replace("_", "/")
        if name:
            doc_map[str(i)] = {"name": name, "dieu": m.get("so_dieu")}

    if not doc_map:
        return answer

    def replace_ref(match):
        inner = match.group(1).strip()
        parts = re.split(r'[,;]', inner)
        first = parts[0].strip()
        if first in doc_map:
            info = doc_map[first]
            rest = inner[len(first):].strip().lstrip(',').strip()
            if rest:
                return f"[{info['name']}, {rest}]"
            elif info["dieu"]:
                return f"[{info['name']}, Điều {info['dieu']}]"
            return f"[{info['name']}]"
        return match.group(0)

    return re.sub(r'\[(\d+(?:\s*,\s*[^\]]*)?)\]', replace_ref, answer)


def fmt_source(d: dict) -> dict:
    """Format a document dict into a source dict for the response."""
    m = d.get("metadata", {})
    return {
        "doc_id": m.get("doc_id", ""),
        "so_hieu": m.get("so_hieu", ""),
        "ten_van_ban": m.get("ten_van_ban", ""),
        "so_dieu": m.get("so_dieu"),
        "breadcrumb": m.get("breadcrumb", ""),
        "tinh_trang": m.get("tinh_trang", ""),
        "content": d.get("content") or d.get("text") or "",
    }
