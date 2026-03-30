import argparse
import math
import json
import re
import unicodedata
from collections import Counter
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
    "so",
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


INTENT_TEMPLATES = [
    {
        "name": "prohibition",
        "trigger": r"nghiem\s+cam|hanh\s+vi\s+bi\s+cam|vi\s+pham\s+chuan\s+muc",
        "boost": [
            (r"hanh\s+vi\s+bi\s+nghiem\s+cam\s+trong\s+co\s+so\s+giao\s+duc", 9.0),
            (r"hanh\s+vi\s+bi\s+nghiem\s+cam", 4.5),
        ],
        "penalty": [
            (r"xu\s+ly\s+vi\s+pham", -2.0),
        ],
    },
    {
        "name": "student_rights_duties",
        "trigger": r"nguoi\s+hoc.*(quyen|nhiem\s+vu)|(quyen|nhiem\s+vu).*nguoi\s+hoc",
        "boost": [
            (r"quyen\s+cua\s+nguoi\s+hoc", 5.8),
            (r"nhiem\s+vu\s+cua\s+nguoi\s+hoc", 5.8),
        ],
        "penalty": [
            (r"nhiem\s+vu\s+va\s+quyen\s+han\s+cua\s+co\s+so\s+giao\s+duc", -1.7),
        ],
    },
    {
        "name": "teacher_standard",
        "trigger": r"tieu\s+chuan\s+cua\s+nha\s+giao|chuan\s+nghe\s+nghiep\s+nha\s+giao",
        "boost": [
            (r"tieu\s+chuan\s+cua\s+nha\s+giao", 7.0),
            (r"chuan\s+nghe\s+nghiep\s+nha\s+giao", 2.5),
        ],
    },
    {
        "name": "student_finance_support",
        "trigger": r"vay\s+von|tin\s+dung|ho\s+tro\s+tai\s+chinh|hoan\s+canh\s+kho\s+khan",
        "boost": [
            (r"tin\s+dung\s+giao\s+duc", 7.5),
            (r"hoc\s+bong|mien,\s*giam\s+hoc\s+phi", 3.2),
        ],
        "penalty": [
            (r"vung\s+co\s+dieu\s+kien\s+kinh\s+te", -1.8),
        ],
    },
    {
        "name": "visiting_lecturer",
        "trigger": r"thinh\s+giang|moi.*giang\s+day.*ngan\s+han",
        "boost": [
            (r"thinh\s+giang", 8.0),
        ],
        "penalty": [
            (r"tam\s+dinh\s+chi\s+giang\s+day", -2.2),
        ],
    },
    {
        "name": "public_university_governance",
        "trigger": r"hoi\s+dong\s+truong|hieu\s+truong|quan\s+tri",
        "boost": [
            (r"hoi\s+dong\s+truong", 3.8),
            (r"hieu\s+truong", 2.0),
        ],
        "penalty": [
            (r"hoi\s+dong\s+quan\s+tri", -2.5),
        ],
        "only_if_query_has": r"cong\s+lap|dai\s+hoc|quan\s+tri",
    },
]


@dataclass
class ArticleHit:
    article_id: str
    score: float
    base_score: float
    reasons: List[str]
    clause_matches: List[Dict]


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

        self.title_tf: Dict[str, Counter] = {}
        self.title_len: Dict[str, int] = {}
        self.title_df: Counter = Counter()
        for article_id in self.article_ids:
            article = self.entity_map.get(article_id, {})
            title = article.get("title", article.get("name", ""))
            toks = content_tokens(title)
            tf = Counter(toks)
            self.title_tf[article_id] = tf
            self.title_len[article_id] = len(toks)
            for tok in tf:
                self.title_df[tok] += 1

        self.num_titles = max(1, len(self.article_ids))
        total_title_len = sum(self.title_len.values())
        self.avg_title_len = (total_title_len / self.num_titles) if total_title_len else 1.0

        self.clause_map: Dict[str, Dict] = {
            e["id"]: e for e in self.entities if e.get("type") == "Clause"
        }
        self.clauses_by_article: Dict[str, List[str]] = {}
        for rel in self.relations:
            if rel.get("relation") != "THUOC_DIEU":
                continue
            clause_id = rel.get("subject", "")
            article_id = rel.get("object", "")
            if clause_id.startswith("cl:") and article_id.startswith("art:"):
                self.clauses_by_article.setdefault(article_id, []).append(clause_id)

        self.clause_tf: Dict[str, Counter] = {}
        self.clause_len: Dict[str, int] = {}
        self.clause_df: Counter = Counter()
        for clause_id, clause in self.clause_map.items():
            toks = content_tokens(clause.get("text", ""))
            tf = Counter(toks)
            self.clause_tf[clause_id] = tf
            self.clause_len[clause_id] = len(toks)
            for tok in tf:
                self.clause_df[tok] += 1

        self.num_clauses = max(1, len(self.clause_map))
        total_clause_len = sum(self.clause_len.values())
        self.avg_clause_len = (total_clause_len / self.num_clauses) if total_clause_len else 1.0

    def token_idf(self, tok: str) -> float:
        df = self.title_df.get(tok, 0)
        return math.log(1.0 + (self.num_titles - df + 0.5) / (df + 0.5))

    def bm25_title_score(self, q_tokens: Set[str], article_id: str, k1: float = 1.2, b: float = 0.75) -> float:
        tf = self.title_tf.get(article_id)
        if not tf:
            return 0.0
        dl = self.title_len.get(article_id, 0)
        if dl <= 0:
            return 0.0

        score = 0.0
        for tok in q_tokens:
            freq = tf.get(tok, 0)
            if not freq:
                continue
            idf = self.token_idf(tok)
            denom = freq + k1 * (1 - b + b * (dl / self.avg_title_len))
            score += idf * ((freq * (k1 + 1)) / max(1e-9, denom))
        return score

    def apply_intent_templates(self, q_plain: str, title_plain: str) -> List[tuple[str, float]]:
        adjustments: List[tuple[str, float]] = []
        for tmpl in INTENT_TEMPLATES:
            if not re.search(tmpl["trigger"], q_plain):
                continue
            required = tmpl.get("only_if_query_has", "")
            if required and not re.search(required, q_plain):
                continue

            for pat, delta in tmpl.get("boost", []):
                if re.search(pat, title_plain):
                    adjustments.append((f"intent_boost={tmpl['name']}", float(delta)))

            for pat, delta in tmpl.get("penalty", []):
                if re.search(pat, title_plain):
                    adjustments.append((f"intent_penalty={tmpl['name']}", float(delta)))

        return adjustments

    def bm25_clause_score(self, q_tokens: Set[str], clause_id: str, k1: float = 1.5, b: float = 0.75) -> float:
        tf = self.clause_tf.get(clause_id)
        if not tf:
            return 0.0
        dl = self.clause_len.get(clause_id, 0)
        if dl <= 0:
            return 0.0

        score = 0.0
        for tok in q_tokens:
            freq = tf.get(tok, 0)
            if not freq:
                continue
            df = self.clause_df.get(tok, 0)
            idf = math.log(1.0 + (self.num_clauses - df + 0.5) / (df + 0.5))
            denom = freq + k1 * (1 - b + b * (dl / self.avg_clause_len))
            score += idf * ((freq * (k1 + 1)) / max(1e-9, denom))
        return score

    def semantic_clause_score(self, q_plain: str, clause_text: str) -> float:
        q_terms = content_tokens(q_plain)
        c_terms = content_tokens(clause_text)
        if not q_terms or not c_terms:
            return 0.0

        q_set = set(q_terms)
        c_set = set(c_terms)
        overlap = q_set & c_set
        if not overlap:
            return 0.0

        weighted_overlap = sum(1.0 + 0.12 * min(5, c_terms.count(t)) for t in overlap)
        coverage = weighted_overlap / max(1.0, len(q_set))

        clause_plain = normalize(clause_text)
        phrase_hits = 0
        for i in range(len(q_terms) - 1):
            phrase = f"{q_terms[i]} {q_terms[i + 1]}"
            if phrase in clause_plain:
                phrase_hits += 1

        return coverage + min(2.0, 0.35 * phrase_hits)

    def best_clause_matches(self, article_id: str, q_tokens: Set[str], q_plain: str, limit: int = 3) -> List[Dict]:
        matches: List[Dict] = []
        for clause_id in self.clauses_by_article.get(article_id, []):
            clause = self.clause_map.get(clause_id, {})
            clause_text = clause.get("text", "")
            if not clause_text:
                continue

            bm25 = self.bm25_clause_score(q_tokens, clause_id)
            sem = self.semantic_clause_score(q_plain, clause_text)
            hybrid = 1.15 * bm25 + 0.95 * sem

            if hybrid <= 0:
                continue

            matches.append(
                {
                    "clause_id": clause_id,
                    "bm25": round(bm25, 4),
                    "semantic": round(sem, 4),
                    "hybrid": round(hybrid, 4),
                    "text_preview": clause_text[:280],
                }
            )

        matches.sort(key=lambda x: x["hybrid"], reverse=True)
        return matches[:limit]

    def rerank_semantic_bonus(self, article_id: str, q_tokens: Set[str], q_plain: str, clause_matches: List[Dict]) -> float:
        bonus = 0.0

        if clause_matches:
            bonus += min(3.5, clause_matches[0]["semantic"] * 0.85 + clause_matches[0]["bm25"] * 0.3)
            if len(clause_matches) > 1:
                bonus += min(1.2, clause_matches[1]["semantic"] * 0.35)

        ev_hits = 0
        for ev in self.evidence_by_article.get(article_id, []):
            quote = ev.get("quote", "")
            qtoks = tokenize(quote)
            ov = q_tokens & qtoks
            if ov:
                ev_hits += 1
            if ev_hits >= 3:
                break

        bonus += min(1.2, 0.35 * ev_hits)
        return bonus

    def score_article(self, article_id: str, q_tokens: Set[str], q_plain: str, intent: QueryIntent) -> ArticleHit:
        article = self.entity_map.get(article_id, {})
        title = article.get("title", article.get("name", ""))
        article_number = str(article.get("article_number", "")).strip().lower()
        score = 0.0
        reasons: List[str] = []

        title_tokens = tokenize(title)
        overlap_title = q_tokens & title_tokens
        if overlap_title:
            overlap_weight = sum(self.token_idf(tok) for tok in overlap_title)
            score += 1.3 * overlap_weight
            reasons.append(f"title_overlap={','.join(sorted(overlap_title))}")

        title_bm25 = self.bm25_title_score(q_tokens, article_id)
        if title_bm25 > 0:
            title_bm25_boost = min(14.0, 4.0 * title_bm25)
            score += title_bm25_boost
            reasons.append(f"title_bm25={round(title_bm25, 3)}")

        # Phrase overlap captures specific topic phrases and reduces single-token noise.
        q_content = content_tokens(q_plain)
        title_plain = normalize(title)
        if intent.asks_detailing and (
            "quy dinh chi tiet" in title_plain or "huong dan thi hanh" in title_plain
        ):
            score += 4.2
            reasons.append("detailing_title_boost")

        phrase_score = 0.0
        bigram_hits = 0
        for i in range(len(q_content) - 1):
            bg = f"{q_content[i]} {q_content[i + 1]}"
            if bg in title_plain:
                bigram_hits += 1
                phrase_score += (self.token_idf(q_content[i]) + self.token_idf(q_content[i + 1])) / 2.0

        trigram_hits = 0
        for i in range(len(q_content) - 2):
            tg = f"{q_content[i]} {q_content[i + 1]} {q_content[i + 2]}"
            if tg in title_plain:
                trigram_hits += 1
                phrase_score += (
                    self.token_idf(q_content[i])
                    + self.token_idf(q_content[i + 1])
                    + self.token_idf(q_content[i + 2])
                ) / 3.0

        if bigram_hits:
            score += min(4.5, 0.9 * phrase_score)
            reasons.append(f"title_phrase_hits={bigram_hits}")
        if trigram_hits:
            reasons.append(f"title_trigram_hits={trigram_hits}")

        if article_number and article_number in intent.article_numbers:
            score += 5.0
            reasons.append(f"article_number_match={article_number}")

        for reason_tag, delta in self.apply_intent_templates(q_plain, title_plain):
            score += delta
            reasons.append(reason_tag)

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
                ov_weight = sum(self.token_idf(tok) for tok in ov)
                score += rel_weight * (0.6 + 0.35 * ov_weight)
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
                ov_weight = sum(self.token_idf(tok) for tok in ov)
                score += ev_weight * (0.3 + 0.18 * ov_weight)
                reasons.append(f"evidence_overlap={','.join(sorted(ov))}")
                evidence_seen += 1

                if intent.asks_authority and tr == "GIAO_THAM_QUYEN_CHO":
                    score += 1.3

            if evidence_seen >= 6:
                break

        clause_matches = self.best_clause_matches(article_id, q_tokens, q_plain, limit=3)
        if clause_matches:
            clause_hybrid = sum(m["hybrid"] for m in clause_matches[:2])
            query_specificity = (
                sum(self.token_idf(tok) for tok in q_tokens) / max(1, len(q_tokens)) if q_tokens else 1.0
            )
            clause_scale = 2.8 - min(1.2, 0.35 * query_specificity)
            clause_boost = min(8.0, clause_scale * math.log1p(max(0.0, clause_hybrid)))
            score += clause_boost
            reasons.append(f"clause_hybrid={round(clause_hybrid, 3)}")
            reasons.append(f"clause_boost={round(clause_boost, 3)}")

        if score < 0:
            score = 0.0

        return ArticleHit(
            article_id=article_id,
            score=score,
            base_score=score,
            reasons=reasons[:10],
            clause_matches=clause_matches,
        )

    def ask(self, question: str, top_k: int = 5) -> Dict:
        q_tokens = tokenize(question)
        q_plain = normalize(question)
        intent = infer_intent(question)
        hits = [self.score_article(aid, q_tokens, q_plain, intent) for aid in self.article_ids]
        hits = [h for h in hits if h.score > 0]
        hits.sort(key=lambda x: x.score, reverse=True)

        rerank_pool = min(len(hits), max(24, top_k * 6))
        for h in hits[:rerank_pool]:
            bonus = self.rerank_semantic_bonus(h.article_id, q_tokens, q_plain, h.clause_matches)
            if bonus > 0:
                h.score += bonus
                h.reasons.append(f"rerank_semantic={round(bonus, 3)}")

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
                    "base_score": round(h.base_score, 3),
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
                    "sample_clauses": h.clause_matches,
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
        "Hội đồng trường của trường đại học công lập được quy định ở điều nào?",
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