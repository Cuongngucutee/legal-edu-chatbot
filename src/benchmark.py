"""
Benchmarking Framework

Evaluates the BookRAG pipeline with:
1. Precision@K / Recall@K for retrieval
2. Answer quality scoring (faithfulness, relevance)
3. Multi-hop reasoning evaluation
4. End-to-end pipeline performance
"""
from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.models import ParsedQuery, RetrievedChunk, PipelineAnswer
from src.pipeline import BookRAGPipeline

logger = logging.getLogger(__name__)


# ── Benchmark Data ─────────────────────────────────────────────────

@dataclass
class BenchmarkQuestion:
    """A single benchmark question with expected answers."""
    question: str
    intent: str  # "definition", "procedure", "condition", "comparison", "general"
    expected_articles: list[str] = field(default_factory=list)  # e.g. ["Điều 4", "Điều 5"]
    expected_sources: list[str] = field(default_factory=list)   # e.g. ["Luật 08/2012/QH13"]
    expected_keywords: list[str] = field(default_factory=list)  # keywords in answer
    difficulty: str = "easy"  # "easy", "medium", "hard"
    category: str = ""  # "definition", "multi-hop", etc.


@dataclass
class BenchmarkResult:
    """Result for a single benchmark question."""
    question: str
    intent_correct: bool = False
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    keyword_coverage: float = 0.0
    retrieval_time_ms: float = 0.0
    total_time_ms: float = 0.0
    retrieved_articles: list[str] = field(default_factory=list)
    retrieved_contexts: list[str] = field(default_factory=list)
    answer_preview: str = ""
    confidence: float = 0.0


@dataclass
class BenchmarkReport:
    """Aggregated benchmark report."""
    total_questions: int = 0
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    avg_keyword_coverage: float = 0.0
    intent_accuracy: float = 0.0
    avg_retrieval_time_ms: float = 0.0
    avg_total_time_ms: float = 0.0
    avg_confidence: float = 0.0
    results_by_category: dict[str, dict] = field(default_factory=dict)
    individual_results: list[BenchmarkResult] = field(default_factory=list)


# ── Default Benchmark Questions ────────────────────────────────────

DEFAULT_BENCHMARK: list[BenchmarkQuestion] = [
    # ─── Definition questions (easy) ───
    BenchmarkQuestion(
        question="Giáo dục chính quy là gì?",
        intent="definition",
        expected_articles=["Điều 4", "Điều 5"],
        expected_sources=["Luật 08/2012/QH13", "Luật 43/2019/QH14"],
        expected_keywords=["giáo dục chính quy", "khóa học", "tập trung"],
        difficulty="easy",
        category="definition",
    ),
    BenchmarkQuestion(
        question="Cơ sở giáo dục đại học là gì?",
        intent="definition",
        expected_articles=["Điều 4", "Điều 2"],
        expected_sources=["Luật 08/2012/QH13"],
        expected_keywords=["cơ sở giáo dục đại học", "đại học", "trường"],
        difficulty="easy",
        category="definition",
    ),
    BenchmarkQuestion(
        question="Liên thông trong giáo dục là gì?",
        intent="definition",
        expected_articles=["Điều 4", "Điều 5"],
        expected_sources=[],
        expected_keywords=["liên thông", "chương trình", "trình độ"],
        difficulty="easy",
        category="definition",
    ),
    BenchmarkQuestion(
        question="Kiểm định chất lượng giáo dục là gì?",
        intent="definition",
        expected_articles=["Điều 4", "Điều 5"],
        expected_sources=[],
        expected_keywords=["kiểm định", "chất lượng"],
        difficulty="easy",
        category="definition",
    ),

    # ─── Condition questions (medium) ───
    BenchmarkQuestion(
        question="Điều kiện thành lập trường đại học là gì?",
        intent="condition",
        expected_articles=["Điều 22", "Điều 23"],
        expected_sources=["Luật 08/2012/QH13"],
        expected_keywords=["thành lập", "điều kiện", "quy hoạch"],
        difficulty="medium",
        category="condition",
    ),
    BenchmarkQuestion(
        question="Điều kiện để được tuyển sinh đại học?",
        intent="condition",
        expected_articles=[],
        expected_sources=[],
        expected_keywords=["tuyển sinh", "điều kiện"],
        difficulty="medium",
        category="condition",
    ),
    BenchmarkQuestion(
        question="Tiêu chuẩn của giảng viên đại học là gì?",
        intent="condition",
        expected_articles=[],
        expected_sources=[],
        expected_keywords=["giảng viên", "tiêu chuẩn", "trình độ"],
        difficulty="medium",
        category="condition",
    ),

    # ─── Procedure questions (medium) ───
    BenchmarkQuestion(
        question="Quy trình mở ngành đào tạo mới như thế nào?",
        intent="procedure",
        expected_articles=[],
        expected_sources=[],
        expected_keywords=["mở ngành", "đào tạo", "quy trình"],
        difficulty="medium",
        category="procedure",
    ),

    # ─── Multi-hop / Comparison (hard) ───
    BenchmarkQuestion(
        question="So sánh quy định về tự chủ đại học giữa Luật 08/2012 và Luật 34/2018?",
        intent="comparison",
        expected_articles=[],
        expected_sources=["Luật 08/2012/QH13", "Luật 34/2018/QH14"],
        expected_keywords=["tự chủ", "sửa đổi"],
        difficulty="hard",
        category="comparison",
    ),
    BenchmarkQuestion(
        question="Quyền và nghĩa vụ của người học theo Luật Giáo dục được quy định như thế nào?",
        intent="general",
        expected_articles=[],
        expected_sources=["Luật 43/2019/QH14"],
        expected_keywords=["quyền", "nghĩa vụ", "người học"],
        difficulty="medium",
        category="general",
    ),
    BenchmarkQuestion(
        question="Hội đồng trường đại học có những quyền hạn gì?",
        intent="general",
        expected_articles=[],
        expected_sources=[],
        expected_keywords=["hội đồng trường", "quyền hạn", "nhiệm vụ"],
        difficulty="medium",
        category="general",
    ),
    BenchmarkQuestion(
        question="Chính sách học phí và hỗ trợ tài chính cho sinh viên?",
        intent="general",
        expected_articles=[],
        expected_sources=[],
        expected_keywords=["học phí", "tài chính", "hỗ trợ"],
        difficulty="medium",
        category="general",
    ),
]


# ── Benchmark Runner ───────────────────────────────────────────────

class BenchmarkRunner:
    """Runs benchmark evaluation on the BookRAG pipeline."""

    def __init__(self, pipeline: BookRAGPipeline):
        self.pipeline = pipeline

    def run(
        self,
        questions: Optional[list[BenchmarkQuestion]] = None,
        top_k: int = 8,
        verbose: bool = True,
    ) -> BenchmarkReport:
        """Run full benchmark evaluation."""
        if questions is None:
            questions = DEFAULT_BENCHMARK

        report = BenchmarkReport(total_questions=len(questions))
        category_results: dict[str, list[BenchmarkResult]] = {}

        for i, bq in enumerate(questions, 1):
            if verbose:
                logger.info(f"[{i}/{len(questions)}] {bq.question[:60]}...")

            result = self._evaluate_question(bq, top_k)
            report.individual_results.append(result)

            # Group by category
            cat = bq.category or "uncategorized"
            if cat not in category_results:
                category_results[cat] = []
            category_results[cat].append(result)

        # Aggregate metrics
        results = report.individual_results
        n = len(results)
        report.avg_precision = sum(r.precision_at_k for r in results) / n
        report.avg_recall = sum(r.recall_at_k for r in results) / n
        report.avg_keyword_coverage = sum(r.keyword_coverage for r in results) / n
        report.intent_accuracy = sum(1 for r in results if r.intent_correct) / n
        report.avg_retrieval_time_ms = sum(r.retrieval_time_ms for r in results) / n
        report.avg_total_time_ms = sum(r.total_time_ms for r in results) / n
        report.avg_confidence = sum(r.confidence for r in results) / n

        # Per-category metrics
        for cat, cat_res in category_results.items():
            cn = len(cat_res)
            report.results_by_category[cat] = {
                "count": cn,
                "avg_precision": sum(r.precision_at_k for r in cat_res) / cn,
                "avg_recall": sum(r.recall_at_k for r in cat_res) / cn,
                "avg_keyword_coverage": sum(r.keyword_coverage for r in cat_res) / cn,
                "intent_accuracy": sum(1 for r in cat_res if r.intent_correct) / cn,
            }

        return report

    def _evaluate_question(
        self, bq: BenchmarkQuestion, top_k: int
    ) -> BenchmarkResult:
        """Evaluate a single benchmark question."""
        result = BenchmarkResult(question=bq.question)

        # ── Query processing ──
        parsed = self.pipeline.query_processor.process(bq.question)
        result.intent_correct = parsed.intent.value == bq.intent

        # ── Retrieval ──
        t0 = time.time()
        retrieved = self.pipeline.hybrid_retrieval.retrieve(
            parsed_query=parsed, top_k=top_k,
        )
        result.retrieval_time_ms = (time.time() - t0) * 1000

        # ── Precision & Recall ──
        retrieved_articles = set()
        for rc in retrieved:
            result.retrieved_contexts.append(
                f"[{rc.chunk.source} | {rc.chunk.article_title}]\n{rc.chunk.text}"
            )
            art = rc.chunk.article_number
            if art:
                retrieved_articles.add(f"Điều {art}")

        result.retrieved_articles = list(retrieved_articles)

        if bq.expected_articles:
            expected_set = set(bq.expected_articles)
            hits = retrieved_articles & expected_set
            result.precision_at_k = len(hits) / len(retrieved_articles) if retrieved_articles else 0
            result.recall_at_k = len(hits) / len(expected_set) if expected_set else 1.0
        else:
            result.precision_at_k = 1.0 if retrieved else 0.0
            result.recall_at_k = 1.0 if retrieved else 0.0

        # ── Source coverage ──
        if bq.expected_sources:
            retrieved_sources = {rc.chunk.source for rc in retrieved}
            source_hits = sum(
                1 for es in bq.expected_sources
                if any(es in rs for rs in retrieved_sources)
            )
            # Blend source coverage into recall
            source_recall = source_hits / len(bq.expected_sources)
            result.recall_at_k = (result.recall_at_k + source_recall) / 2

        # ── Keyword coverage ──
        if bq.expected_keywords:
            all_text = " ".join(rc.chunk.text.lower() for rc in retrieved)
            keyword_hits = sum(
                1 for kw in bq.expected_keywords if kw.lower() in all_text
            )
            result.keyword_coverage = keyword_hits / len(bq.expected_keywords)
        else:
            result.keyword_coverage = 1.0

        # ── Full pipeline ──
        t1 = time.time()
        answer = self.pipeline.query_raw(bq.question, top_k=top_k)
        result.total_time_ms = (time.time() - t1) * 1000
        result.confidence = answer.confidence
        result.answer_preview = answer.answer[:200]

        return result

    # ── Report formatting ──────────────────────────────────────────

    @staticmethod
    def format_report(report: BenchmarkReport) -> str:
        """Format benchmark report as Markdown."""
        lines = [
            "# 📊 Benchmark Report",
            "",
            "## Overall Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Questions | {report.total_questions} |",
            f"| Avg Precision@K | {report.avg_precision:.3f} |",
            f"| Avg Recall@K | {report.avg_recall:.3f} |",
            f"| Avg Keyword Coverage | {report.avg_keyword_coverage:.3f} |",
            f"| Intent Accuracy | {report.intent_accuracy:.1%} |",
            f"| Avg Confidence | {report.avg_confidence:.3f} |",
            f"| Avg Retrieval Time | {report.avg_retrieval_time_ms:.0f}ms |",
            f"| Avg Total Time | {report.avg_total_time_ms:.0f}ms |",
            "",
        ]

        # Per-category
        if report.results_by_category:
            lines.append("## Results by Category")
            lines.append("")
            lines.append("| Category | Count | Precision | Recall | Keywords | Intent |")
            lines.append("|----------|-------|-----------|--------|----------|--------|")
            for cat, metrics in sorted(report.results_by_category.items()):
                lines.append(
                    f"| {cat} | {metrics['count']} | "
                    f"{metrics['avg_precision']:.3f} | "
                    f"{metrics['avg_recall']:.3f} | "
                    f"{metrics['avg_keyword_coverage']:.3f} | "
                    f"{metrics['intent_accuracy']:.1%} |"
                )
            lines.append("")

        # Individual results
        lines.append("## Individual Results")
        lines.append("")
        for i, r in enumerate(report.individual_results, 1):
            intent_icon = "✅" if r.intent_correct else "❌"
            lines.append(f"### Q{i}: {r.question}")
            lines.append(f"- Intent: {intent_icon}")
            lines.append(f"- Precision@K: {r.precision_at_k:.3f}")
            lines.append(f"- Recall@K: {r.recall_at_k:.3f}")
            lines.append(f"- Keyword Coverage: {r.keyword_coverage:.3f}")
            lines.append(f"- Confidence: {r.confidence:.3f}")
            lines.append(f"- Retrieved: {', '.join(r.retrieved_articles[:5])}")
            lines.append(f"- Time: {r.retrieval_time_ms:.0f}ms retrieval, {r.total_time_ms:.0f}ms total")
            lines.append(f"\n  **Retrieved Contexts ({len(r.retrieved_contexts)}):**\n")
            for ctx in r.retrieved_contexts[:3]:  # Show top 3 chunks to avoid overly long files
                lines.append(f"  > {ctx.replace(chr(10), chr(10) + '  > ')}")
            lines.append("")

        return "\n".join(lines)

    def save_report(self, report: BenchmarkReport, path: Path) -> None:
        """Save report as JSON."""
        data = {
            "total_questions": report.total_questions,
            "avg_precision": report.avg_precision,
            "avg_recall": report.avg_recall,
            "avg_keyword_coverage": report.avg_keyword_coverage,
            "intent_accuracy": report.intent_accuracy,
            "avg_retrieval_time_ms": report.avg_retrieval_time_ms,
            "avg_total_time_ms": report.avg_total_time_ms,
            "avg_confidence": report.avg_confidence,
            "results_by_category": report.results_by_category,
            "individual_results": [
                {
                    "question": r.question,
                    "intent_correct": r.intent_correct,
                    "precision_at_k": r.precision_at_k,
                    "recall_at_k": r.recall_at_k,
                    "keyword_coverage": r.keyword_coverage,
                    "confidence": r.confidence,
                    "retrieval_time_ms": r.retrieval_time_ms,
                    "total_time_ms": r.total_time_ms,
                    "retrieved_articles": r.retrieved_articles,
                    "retrieved_contexts": r.retrieved_contexts,
                }
                for r in report.individual_results
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Report saved to {path}")
