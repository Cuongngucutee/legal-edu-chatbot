import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


PROJECT_ROOT = Path(__file__).resolve().parent.parent
KG_PATH = PROJECT_ROOT / "outputs" / "knowledge_graph" / "knowledge_graph_full.json"


STOPWORDS = {
    "la",
    "cua",
    "va",
    "voi",
    "cho",
    "duoc",
    "theo",
    "trong",
    "co",
    "cac",
    "nhung",
    "ve",
    "ai",
    "nao",
    "gi",
    "tai",
    "mot",
    "phap",
    "luat",
    "giao",
    "duc",
    "dieu",
    "khoan",
    "muc",
    "chuong",
    "nay",
    "do",
    "tu",
    "toi",
    "o",
    "bi",
    "da",
    "dang",
    "quy",
    "dinh",
    "chi",
    "tiet",
    "huong",
    "dan",
    "thi",
    "hanh",
    "noi",
}


REFERENCE_RELATIONS = {"DAN_CHIEU_DEN", "TAC_DONG_DEN_DIEU", "BI_BAI_BO_BOI"}
RELATION_WEIGHTS = {
    "QUY_DINH_VE": 1.6,
    "GIAO_THAM_QUYEN_CHO": 1.9,
    "PHAI_CO": 1.3,
    "THUOC_DIEU": 0.9,
    "THUOC_VAN_BAN": 0.7,
    "DAN_CHIEU_DEN": 0.25,
    "TAC_DONG_DEN_DIEU": 0.2,
    "BI_BAI_BO_BOI": 0.15,
}


@dataclass
class ArticleHit:
    article_id: str
    score: float
    reasons: List[str]


@dataclass(frozen=True)
class QueryIntent:
    article_numbers: Set[str]
    asks_authority: bool
    asks_reference: bool
    asks_amendment: bool
    asks_detailing: bool


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> Set[str]:
    toks = set(re.findall(r"[a-z0-9]{2,}", normalize(text)))
    return {t for t in toks if t not in STOPWORDS}


def content_tokens(text: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]{2,}", normalize(text))
    return [t for t in toks if t not in STOPWORDS]


def extract_article_numbers(text: str) -> Set[str]:
    return {m.group(1).lower() for m in re.finditer(r"\bdieu\s+(\d+[a-z]?)\b", normalize(text), flags=re.IGNORECASE)}


def get_entity_article_number(entity: Dict) -> Optional[str]:
    num = str(entity.get("article_number", "")).strip().lower()
    if num:
        return num

    name = normalize(entity.get("name", ""))
    m = re.search(r"\bdieu\s+(\d+[a-z]?)\b", name)
    if m:
        return m.group(1).lower()
    return None


def is_reference_only_quote(text: str) -> bool:
    cleaned = normalize(text)
    return bool(re.fullmatch(r"dieu\s+\d+[a-z]?", cleaned))


def infer_intent(question: str) -> QueryIntent:
    q_plain = normalize(question)
    article_numbers = extract_article_numbers(question)

    asks_authority = bool(
        re.search(
            r"tham\s+quyen|ai\s+co\s+tham\s+quyen|co\s+quan\s+nao|quy\s+dinh\s+chi\s+tiet",
            q_plain,
        )
    )
    asks_reference = bool(
        article_numbers
        or re.search(
            r"dan\s+chieu|tham\s+chieu|vien\s+dan|sua\s+doi\s+dieu|bo\s+sung\s+dieu|bai\s+bo\s+dieu",
            q_plain,
        )
    )
    asks_amendment = bool(re.search(r"sua\s+doi|bo\s+sung|bai\s+bo", q_plain))
    asks_detailing = bool(re.search(r"quy\s+dinh\s+chi\s+tiet|huong\s+dan\s+thi\s+hanh", q_plain))

    return QueryIntent(
        article_numbers=article_numbers,
        asks_authority=asks_authority,
        asks_reference=asks_reference,
        asks_amendment=asks_amendment,
        asks_detailing=asks_detailing,
    )


class KGFullGraphRAGTester:
    def __init__(self, kg_path: Path):
        data = json.loads(kg_path.read_text(encoding="utf-8"))
        kg = data["knowledge_graph"]
        self.entities = kg["entities"]
        self.relations = kg["relations"]
        self.evidence = kg["evidence"]

        self.entity_map: Dict[str, Dict] = {e["id"]: e for e in self.entities}
        self.article_ids = [e["id"] for e in self.entities if e.get("type") == "Article"]

        self.relations_by_subject: Dict[str, List[Dict]] = {}
        for rel in self.relations:
            self.relations_by_subject.setdefault(rel["subject"], []).append(rel)

        self.evidence_by_article: Dict[str, List[Dict]] = {}
        for ev in self.evidence:
            triple = ev.get("triple", {})
            subj = triple.get("subject", "")
            if subj.startswith("art:"):
                self.evidence_by_article.setdefault(subj, []).append(ev)

    def score_article(self, article_id: str, q_tokens: Set[str], q_plain: str, intent: QueryIntent) -> ArticleHit:
        article = self.entity_map.get(article_id, {})
        title = article.get("title", article.get("name", ""))
        article_number = str(article.get("article_number", "")).strip().lower()
        score = 0.0
        reasons: List[str] = []

        title_tokens = tokenize(title)
        overlap_title = q_tokens & title_tokens
        if overlap_title:
            score += 2.2 * len(overlap_title)
            reasons.append(f"title_overlap={','.join(sorted(overlap_title))}")

        # Phrase overlap captures specific topic phrases and reduces single-token noise.
        q_content = content_tokens(q_plain)
        title_plain = normalize(title)
        if intent.asks_detailing and (
            "quy dinh chi tiet" in title_plain or "huong dan thi hanh" in title_plain
        ):
            score += 4.2
            reasons.append("detailing_title_boost")

        bigram_hits = 0
        for i in range(len(q_content) - 1):
            bg = f"{q_content[i]} {q_content[i + 1]}"
            if bg in title_plain:
                bigram_hits += 1
        if bigram_hits:
            score += min(2.5, 1.25 * bigram_hits)
            reasons.append(f"title_phrase_hits={bigram_hits}")

        if article_number and article_number in intent.article_numbers:
            score += 5.0
            reasons.append(f"article_number_match={article_number}")

        is_amendment_article = bool(re.search(r"sua doi|bo sung mot so dieu|bai bo", title_plain))
        if is_amendment_article and not intent.asks_amendment:
            score -= 1.8
            reasons.append("penalty=amendment_title")
        elif is_amendment_article and intent.asks_amendment:
            score += 0.8
            reasons.append("bonus=amendment_title")

        # Score by relation object entity names.
        rel_seen = 0
        for rel in self.relations_by_subject.get(article_id, []):
            rel_name = rel["relation"]
            rel_weight = RELATION_WEIGHTS.get(rel_name, 0.6)
            if rel_name in REFERENCE_RELATIONS and not intent.asks_reference:
                continue

            obj = self.entity_map.get(rel["object"], {})
            obj_name = obj.get("name", "")
            obj_tokens = tokenize(obj_name)
            ov = q_tokens & obj_tokens
            if ov:
                score += rel_weight * (0.9 + 0.25 * len(ov))
                reasons.append(f"rel:{rel_name}->{obj_name}")
                rel_seen += 1

                obj_article_num = get_entity_article_number(obj)
                if obj_article_num and obj_article_num in intent.article_numbers:
                    score += 2.2
                    reasons.append(f"citation_target_match={obj_article_num}")

                if "doi tuong ap dung" in q_plain and rel_name == "QUY_DINH_VE":
                    score += 0.6

                if intent.asks_authority and rel_name == "GIAO_THAM_QUYEN_CHO":
                    score += 2.0

            if rel_seen >= 14:
                break

        # Score by quote evidence.
        evidence_seen = 0
        for ev in self.evidence_by_article.get(article_id, []):
            tr = ev.get("triple", {}).get("relation", "")
            if tr in REFERENCE_RELATIONS and not intent.asks_reference:
                continue

            quote = ev.get("quote", "")
            if is_reference_only_quote(quote):
                continue

            qt = tokenize(quote)
            ov = q_tokens & qt
            if ov:
                ev_weight = RELATION_WEIGHTS.get(tr, 0.6)
                score += ev_weight * (0.45 + 0.18 * len(ov))
                reasons.append(f"evidence_overlap={','.join(sorted(ov))}")
                evidence_seen += 1

                if intent.asks_authority and tr == "GIAO_THAM_QUYEN_CHO":
                    score += 1.3

            if evidence_seen >= 6:
                break

        if score < 0:
            score = 0.0

        return ArticleHit(article_id=article_id, score=score, reasons=reasons[:8])

    def ask(self, question: str, top_k: int = 5) -> Dict:
        q_tokens = tokenize(question)
        q_plain = normalize(question)
        intent = infer_intent(question)
        hits = [self.score_article(aid, q_tokens, q_plain, intent) for aid in self.article_ids]
        hits = [h for h in hits if h.score > 0]
        hits.sort(key=lambda x: x.score, reverse=True)

        top = []
        for h in hits[:top_k]:
            article = self.entity_map[h.article_id]
            rels = self.relations_by_subject.get(h.article_id, [])[:5]
            evs = self.evidence_by_article.get(h.article_id, [])[:3]
            top.append(
                {
                    "article_id": h.article_id,
                    "article_number": str(article.get("article_number", "")),
                    "title": article.get("title", article.get("name", "")),
                    "score": round(h.score, 3),
                    "reasons": h.reasons,
                    "sample_relations": rels,
                    "sample_evidence": [
                        {
                            "relation": e.get("triple", {}).get("relation", ""),
                            "quote": e.get("quote", ""),
                            "source_article": e.get("source_article", ""),
                        }
                        for e in evs
                    ],
                }
            )

        return {
            "question": question,
            "top_articles": top,
        }


def default_questions() -> List[str]:
    return [
        "Giáo dục bắt buộc được quy định tại điều mấy?",
        "Ai có thẩm quyền quy định chi tiết về văn bằng, chứng chỉ?",
        "Nhà nước có chính sách gì về học bổng và tín dụng sinh viên?",
        "Cơ sở giáo dục đại học có trách nhiệm giải trình với ai?",
        "Giáo dục bắt buộc bao gồm những cấp học nào?",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphRAG relevance test for knowledge_graph_full.json")
    parser.add_argument("--question", type=str, default="", help="Single question")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top articles")
    parser.add_argument("--json", action="store_true", help="Print as JSON")
    args = parser.parse_args()

    if not KG_PATH.exists():
        raise FileNotFoundError(f"Missing KG file: {KG_PATH}")

    tester = KGFullGraphRAGTester(KG_PATH)
    questions = [args.question] if args.question else default_questions()
    results = [tester.ask(q, top_k=args.top_k) for q in questions]

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    for idx, res in enumerate(results, start=1):
        print(f"\n=== Test {idx}: {res['question']} ===")
        if not res["top_articles"]:
            print("No relevant article found.")
            continue

        for i, item in enumerate(res["top_articles"], start=1):
            print(f"{i}. {item['title']} | score={item['score']}")
            if item["reasons"]:
                print(f"   reasons: {'; '.join(item['reasons'])}")
            for ev in item["sample_evidence"][:2]:
                print(f"   evidence[{ev['relation']}]: {ev['quote'][:170]}")


if __name__ == "__main__":
    main()