import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "final"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "knowledge_graph" / "knowledge_graph_full.json"


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str


class KnowledgeGraphBuilder:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.entities: Dict[str, Dict] = {}
        self.relations: Set[Tuple[str, str, str]] = set()
        self.evidence: List[Dict] = []

        self.doc_to_articles: Dict[str, Set[str]] = {}

        self.re_article_ref = re.compile(r"\bĐiều\s+(\d+[a-z]?)\b", re.IGNORECASE)
        self.re_clause_start = re.compile(r"^(\d+)\.\s+(.*)$")
        self.re_requirement = re.compile(
            r"\bco\s+((?:bang|chung\s+chi)[^;\n\.]{3,180}?)\s+doi\s+voi\s+([^;\n\.]{3,120})",
            re.IGNORECASE,
        )
        self.authority_patterns = [
            ("chinh phu quy dinh", "Chinh phu"),
            ("thu tuong chinh phu quy dinh", "Thu tuong Chinh phu"),
            ("bo truong bo giao duc va dao tao quy dinh", "Bo truong Bo Giao duc va Dao tao"),
            ("bo truong bo lao dong thuong binh va xa hoi quy dinh", "Bo truong Bo Lao dong - Thuong binh va Xa hoi"),
            ("uy ban nhan dan cap tinh quy dinh", "Uy ban nhan dan cap tinh"),
            ("chu tich uy ban nhan dan cap tinh quy dinh", "Chu tich Uy ban nhan dan cap tinh"),
        ]

    def normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        return re.sub(r"\s+", " ", text).strip()

    def slug(self, text: str) -> str:
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        text = text.replace("đ", "d").replace("Đ", "D")
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        return text.strip("_")

    def plain_text(self, text: str) -> str:
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        text = text.lower().replace("đ", "d")
        return re.sub(r"\s+", " ", text).strip()

    def doc_slug_from_source(self, source: str) -> str:
        return self.slug(source)

    def article_entity_id(self, doc_slug: str, article_number: str) -> str:
        return f"art:{doc_slug}:d{article_number.lower()}"

    def add_entity(self, entity: Dict) -> None:
        entity_id = entity["id"]
        if entity_id not in self.entities:
            self.entities[entity_id] = entity

    def add_relation(
        self,
        subject: str,
        relation: str,
        object_id: str,
        source_doc: Optional[str] = None,
        source_article: Optional[str] = None,
        source_clause: Optional[str] = None,
        quote: Optional[str] = None,
        confidence: float = 1.0,
    ) -> None:
        key = (subject, relation, object_id)
        if key in self.relations:
            return
        self.relations.add(key)
        if quote and source_doc and source_article:
            self.evidence.append(
                {
                    "triple": {
                        "subject": subject,
                        "relation": relation,
                        "object": object_id,
                    },
                    "source_document": source_doc,
                    "source_article": source_article,
                    "source_clause": source_clause,
                    "quote": quote[:600],
                    "confidence": confidence,
                }
            )

    def split_numbered_clauses(self, full_text: str) -> List[Tuple[str, str]]:
        lines = [ln.strip() for ln in full_text.split("\n") if ln.strip()]
        clauses: List[Tuple[str, str]] = []
        current_num = None
        current_content: List[str] = []

        for line in lines[1:]:
            m = self.re_clause_start.match(line)
            if m:
                if current_num:
                    clauses.append((current_num, " ".join(current_content).strip()))
                current_num = m.group(1)
                current_content = [m.group(2).strip()]
            elif current_num:
                current_content.append(line)

        if current_num:
            clauses.append((current_num, " ".join(current_content).strip()))
        return clauses

    def topic_from_article_title(self, article_title: str) -> Optional[str]:
        title = self.normalize(article_title)
        title = re.sub(r"^Điều\s+\d+[a-z]?\s*[\.:\-]\s*", "", title, flags=re.IGNORECASE)
        if not title:
            return None
        return title

    def parse_document_metadata(self, file_name: str, first_source: str) -> Dict:
        name_clean = file_name.replace(".json", "")
        m = re.search(r"(\d+)_(\d{4})_QH(\d+)", name_clean)
        if m:
            code = m.group(1)
            year = m.group(2)
            session = m.group(3)
            canonical = f"Luat {code}/{year}/QH{session}"
        else:
            canonical = first_source.replace("Luật", "Luat")

        return {
            "name": canonical,
            "status": "chua_xac_dinh",
        }

    def ensure_article_ref_entity(self, ref_text: str) -> str:
        ref_slug = self.slug(ref_text)
        entity_id = f"ref:{ref_slug}"
        self.add_entity(
            {
                "id": entity_id,
                "name": ref_text,
                "type": "ArticleReference",
            }
        )
        return entity_id

    def extract_semantic_relations(
        self,
        doc_slug: str,
        article_id: str,
        article_number: str,
        full_text: str,
        source_doc: str,
    ) -> None:
        lines = [ln.strip() for ln in full_text.split("\n") if ln.strip()]
        title_line = lines[0] if lines else ""
        body = "\n".join(lines[1:]) if len(lines) > 1 else ""
        body_plain = self.plain_text(body)

        for ref in self.re_article_ref.finditer(body):
            target_num = ref.group(1)
            if target_num.lower() == article_number.lower():
                continue
            target_article_id = self.article_entity_id(doc_slug, target_num)
            if target_article_id in self.entities:
                obj = target_article_id
            else:
                obj = self.ensure_article_ref_entity(f"Dieu {target_num}")
            self.add_relation(
                article_id,
                "DAN_CHIEU_DEN",
                obj,
                source_doc=source_doc,
                source_article=title_line,
                quote=ref.group(0),
                confidence=0.95,
            )

        for rm in self.re_requirement.finditer(body_plain):
            requirement_text = rm.group(1).strip()
            role_text = rm.group(2).strip()
            if len(requirement_text) < 5 or len(role_text) < 3:
                continue
            req_id = f"req:{self.slug(requirement_text)}"
            role_id = f"role:{self.slug(role_text)}"
            self.add_entity(
                {
                    "id": req_id,
                    "name": requirement_text,
                    "type": "Requirement",
                }
            )
            self.add_entity(
                {
                    "id": role_id,
                    "name": role_text,
                    "type": "Role",
                }
            )
            self.add_relation(
                role_id,
                "PHAI_CO",
                req_id,
                source_doc=source_doc,
                source_article=title_line,
                quote=rm.group(0),
                confidence=0.72,
            )

        for pattern, auth_name in self.authority_patterns:
            if pattern in body_plain:
                auth_id = f"auth:{self.slug(auth_name)}"
                self.add_entity(
                    {
                        "id": auth_id,
                        "name": auth_name,
                        "type": "Authority",
                    }
                )
                self.add_relation(
                    article_id,
                    "GIAO_THAM_QUYEN_CHO",
                    auth_id,
                    source_doc=source_doc,
                    source_article=title_line,
                    quote=pattern,
                    confidence=0.9,
                )

        for mm in re.finditer(r"(?i)sua\s+doi|bo\s+sung", body_plain):
            for ref in self.re_article_ref.finditer(body):
                target = self.ensure_article_ref_entity(f"Dieu {ref.group(1)}")
                self.add_relation(
                    article_id,
                    "TAC_DONG_DEN_DIEU",
                    target,
                    source_doc=source_doc,
                    source_article=title_line,
                    quote=f"{ref.group(0)}",
                    confidence=0.75,
                )
            break

        if re.search(r"(?i)bai\s+bo", body_plain):
            for ref in self.re_article_ref.finditer(body):
                target = self.ensure_article_ref_entity(f"Dieu {ref.group(1)}")
                self.add_relation(
                    target,
                    "BI_BAI_BO_BOI",
                    article_id,
                    source_doc=source_doc,
                    source_article=title_line,
                    quote=f"Bai bo {ref.group(0)}",
                    confidence=0.78,
                )

    def build(self) -> Dict:
        files = sorted(self.data_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"Khong tim thay du lieu trong {self.data_dir}")

        for file_path in files:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if not data:
                continue

            first_source = data[0].get("metadata", {}).get("source", file_path.stem)
            doc_meta = self.parse_document_metadata(file_path.name, first_source)
            doc_slug = self.doc_slug_from_source(doc_meta["name"])
            doc_id = f"doc:{doc_slug}"

            self.add_entity(
                {
                    "id": doc_id,
                    "name": doc_meta["name"],
                    "type": "LegalDocument",
                    "status": doc_meta["status"],
                    "source_file": file_path.name,
                }
            )

            self.doc_to_articles.setdefault(doc_id, set())

            for chunk in data:
                meta = chunk.get("metadata", {})
                content = chunk.get("content", {})
                article_number = str(meta.get("article_number", "")).strip()
                if not article_number:
                    continue

                article_id = self.article_entity_id(doc_slug, article_number)
                article_title = self.normalize(meta.get("article_title", f"Dieu {article_number}"))
                full_text = content.get("full_text", "")

                article_entity = {
                    "id": article_id,
                    "name": f"Dieu {article_number}",
                    "type": "Article",
                    "title": article_title,
                    "article_number": article_number,
                    "chapter": meta.get("chapter", ""),
                    "section": meta.get("section", ""),
                    "is_sub_split": bool(meta.get("is_sub_split", False)),
                }
                if meta.get("target_article"):
                    article_entity["target_article"] = meta.get("target_article")

                self.add_entity(article_entity)
                self.doc_to_articles[doc_id].add(article_id)

                self.add_relation(article_id, "THUOC_VAN_BAN", doc_id)

                topic = self.topic_from_article_title(article_title)
                if topic:
                    topic_id = f"topic:{self.slug(topic)}"
                    self.add_entity(
                        {
                            "id": topic_id,
                            "name": topic,
                            "type": "Topic",
                        }
                    )
                    self.add_relation(article_id, "QUY_DINH_VE", topic_id)

                clauses = self.split_numbered_clauses(full_text)
                for clause_num, clause_text in clauses:
                    clause_id = f"cl:{doc_slug}:d{article_number.lower()}:k{clause_num}"
                    self.add_entity(
                        {
                            "id": clause_id,
                            "name": f"Dieu {article_number} Khoan {clause_num}",
                            "type": "Clause",
                            "text": clause_text[:1200],
                        }
                    )
                    self.add_relation(clause_id, "THUOC_DIEU", article_id)

                if full_text:
                    self.extract_semantic_relations(
                        doc_slug=doc_slug,
                        article_id=article_id,
                        article_number=article_number,
                        full_text=full_text,
                        source_doc=doc_meta["name"],
                    )

                target_article = meta.get("target_article")
                if target_article and target_article != "Lời dẫn/Mở đầu":
                    target_id = self.ensure_article_ref_entity(self.normalize(target_article))
                    self.add_relation(
                        article_id,
                        "TAC_DONG_DEN_DIEU",
                        target_id,
                        source_doc=doc_meta["name"],
                        source_article=article_title,
                        quote=f"target_article={target_article}",
                        confidence=0.92,
                    )

        graph = {
            "knowledge_graph": {
                "version": "2.0",
                "domain": "phap_luat_giao_duc_viet_nam",
                "generated_from": str(self.data_dir),
                "entity_types": [
                    "LegalDocument",
                    "Article",
                    "Clause",
                    "Topic",
                    "Role",
                    "Requirement",
                    "Authority",
                    "ArticleReference",
                ],
                "relation_types": [
                    "THUOC_VAN_BAN",
                    "THUOC_DIEU",
                    "QUY_DINH_VE",
                    "DAN_CHIEU_DEN",
                    "TAC_DONG_DEN_DIEU",
                    "BI_BAI_BO_BOI",
                    "GIAO_THAM_QUYEN_CHO",
                    "PHAI_CO",
                ],
                "stats": {
                    "entities": len(self.entities),
                    "relations": len(self.relations),
                    "evidence": len(self.evidence),
                    "documents": len([e for e in self.entities.values() if e["type"] == "LegalDocument"]),
                    "articles": len([e for e in self.entities.values() if e["type"] == "Article"]),
                    "clauses": len([e for e in self.entities.values() if e["type"] == "Clause"]),
                },
                "entities": sorted(self.entities.values(), key=lambda x: x["id"]),
                "relations": [
                    {
                        "subject": s,
                        "relation": r,
                        "object": o,
                    }
                    for (s, r, o) in sorted(self.relations)
                ],
                "evidence": self.evidence,
            }
        }
        return graph


if __name__ == "__main__":
    builder = KnowledgeGraphBuilder(DATA_DIR)
    kg = builder.build()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)
    stats = kg["knowledge_graph"]["stats"]
    print(
        "Da tao knowledge graph:",
        f"entities={stats['entities']}",
        f"relations={stats['relations']}",
        f"evidence={stats['evidence']}",
    )
