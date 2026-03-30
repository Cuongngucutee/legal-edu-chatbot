import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from test_graphrag_kg_full import KG_PATH, KGFullGraphRAGTester, normalize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "benchmarks" / "kg_query_benchmark.json"


def match_rule(article: Dict, rule: Dict) -> bool:
    article_number = str(article.get("article_number", "")).strip().lower()
    title_plain = normalize(article.get("title", ""))

    rule_number = str(rule.get("article_number", "")).strip().lower()
    if rule_number and article_number != rule_number:
        return False

    title_contains = str(rule.get("title_contains", "")).strip()
    if title_contains:
        if normalize(title_contains) not in title_plain:
            return False

    return True


def first_match_rank(top_articles: List[Dict], gold_rules: List[Dict]) -> Optional[int]:
    for rank, article in enumerate(top_articles, start=1):
        if any(match_rule(article, rule) for rule in gold_rules):
            return rank
    return None


def evaluate(benchmark_path: Path, top_k: int) -> Dict:
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if not cases:
        raise ValueError("Benchmark has no test cases")

    tester = KGFullGraphRAGTester(KG_PATH)

    details: List[Dict] = []
    for case in cases:
        case_top_k = int(case.get("top_k", top_k))
        result = tester.ask(case["question"], top_k=case_top_k)
        rank = first_match_rank(result["top_articles"], case["gold"])
        details.append(
            {
                "id": case["id"],
                "question": case["question"],
                "top_k": case_top_k,
                "best_rank": rank,
                "hit": rank is not None,
                "reciprocal_rank": 0.0 if rank is None else round(1.0 / rank, 6),
                "top_articles": result["top_articles"],
            }
        )

    n = len(details)
    hit_at_1 = sum(1 for d in details if d["best_rank"] == 1) / n
    hit_at_3 = sum(1 for d in details if d["best_rank"] is not None and d["best_rank"] <= 3) / n
    hit_at_5 = sum(1 for d in details if d["best_rank"] is not None and d["best_rank"] <= 5) / n
    mrr = sum(d["reciprocal_rank"] for d in details) / n

    return {
        "benchmark": payload.get("name", benchmark_path.name),
        "num_cases": n,
        "metrics": {
            "hit_at_1": round(hit_at_1, 4),
            "hit_at_3": round(hit_at_3, 4),
            "hit_at_5": round(hit_at_5, 4),
            "mrr": round(mrr, 4),
        },
        "details": details,
    }


def print_report(report: Dict, fail_only: bool = False) -> None:
    metrics = report["metrics"]
    print(f"Benchmark: {report['benchmark']} | cases={report['num_cases']}")
    print(
        "Metrics: "
        f"Hit@1={metrics['hit_at_1']:.4f} "
        f"Hit@3={metrics['hit_at_3']:.4f} "
        f"Hit@5={metrics['hit_at_5']:.4f} "
        f"MRR={metrics['mrr']:.4f}"
    )

    print("\nCase details:")
    for item in report["details"]:
        if fail_only and item["hit"]:
            continue
        status = "PASS" if item["hit"] else "FAIL"
        rank = item["best_rank"] if item["best_rank"] is not None else "-"
        print(f"- {item['id']} | {status} | best_rank={rank} | q={item['question']}")
        for idx, art in enumerate(item["top_articles"][:3], start=1):
            print(f"    top{idx}: D{art.get('article_number', '?')} | {art.get('title', '')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate KG retrieval with benchmark questions")
    parser.add_argument(
        "--benchmark",
        type=str,
        default=str(DEFAULT_BENCHMARK_PATH),
        help="Path to benchmark json file",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Default top-k retrieval")
    parser.add_argument("--fail-only", action="store_true", help="Only print failed cases")
    parser.add_argument("--json", action="store_true", help="Print full report as JSON")
    parser.add_argument("--json-output", type=str, default="", help="Write JSON report to file")
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Missing benchmark file: {benchmark_path}")
    if not KG_PATH.exists():
        raise FileNotFoundError(f"Missing KG file: {KG_PATH}")

    report = evaluate(benchmark_path=benchmark_path, top_k=args.top_k)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report, fail_only=args.fail_only)

    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report: {out_path}")


if __name__ == "__main__":
    main()
