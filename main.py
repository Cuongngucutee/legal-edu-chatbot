#!/usr/bin/env python3
"""
BookRAG – Interactive CLI for Vietnamese Legal Education QA

Usage:
    python main.py                  # Interactive mode
    python main.py --debug          # Show reasoning trace
    python main.py --query "..."    # Single query mode
    python main.py --benchmark      # Run benchmark evaluation
    python main.py --enhance-kg     # Enhance KG with LLM
"""
from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="BookRAG – Hệ thống hỏi đáp Luật Giáo dục Việt Nam"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Single query mode (answer one question and exit)",
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Show reasoning trace in output",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show pipeline statistics",
    )
    parser.add_argument(
        "--tree",
        action="store_true",
        help="Show hierarchical tree structure",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark evaluation on test questions",
    )
    parser.add_argument(
        "--enhance-kg",
        action="store_true",
        help="Enhance knowledge graph using LLM extraction",
    )
    parser.add_argument(
        "--enhance-kg-max",
        type=int,
        default=50,
        help="Max chunks to process for KG enhancement (default: 50)",
    )
    args = parser.parse_args()

    # ── Build pipeline ─────────────────────────────────────────────
    console.print(
        Panel(
            "[bold cyan]BookRAG – Hệ thống hỏi đáp Luật Giáo dục Việt Nam[/]"
            "\n[dim]Đang khởi tạo pipeline...[/]",
            border_style="cyan",
        )
    )

    from src.indexing import build_pipeline
    pipeline = build_pipeline(verbose=True)

    # ── Stats mode ─────────────────────────────────────────────────
    if args.stats:
        stats = pipeline.get_stats()
        console.print("\n[bold]📊 Pipeline Statistics:[/]")
        for k, v in stats.items():
            console.print(f"  {k}: [cyan]{v}[/]")
        return

    # ── Tree mode ──────────────────────────────────────────────────
    if args.tree:
        console.print("\n[bold]🌳 Hierarchical Tree:[/]")
        console.print(pipeline.get_tree_info())
        return

    # ── Benchmark mode ─────────────────────────────────────────────
    if args.benchmark:
        _run_benchmark(pipeline)
        return

    # ── KG Enhancement mode ────────────────────────────────────────
    if args.enhance_kg:
        _enhance_kg(pipeline, max_chunks=args.enhance_kg_max)
        return

    # ── Single query mode ──────────────────────────────────────────
    if args.query:
        result = pipeline.query(args.query, show_reasoning=args.debug)
        console.print(Markdown(result))
        return

    # ── Interactive mode ───────────────────────────────────────────
    console.print(
        Panel(
            "[bold green]✅ Pipeline ready![/]\n\n"
            "Nhập câu hỏi về luật giáo dục Việt Nam.\n"
            "Gõ [bold cyan]/debug[/] để bật/tắt hiển thị suy luận.\n"
            "Gõ [bold cyan]/stats[/] để xem thống kê.\n"
            "Gõ [bold cyan]/tree[/] để xem cấu trúc cây.\n"
            "Gõ [bold cyan]/benchmark[/] để chạy benchmark.\n"
            "Gõ [bold cyan]/quit[/] hoặc Ctrl+C để thoát.",
            title="[bold]Hướng dẫn[/]",
            border_style="green",
        )
    )

    show_debug = args.debug

    while True:
        try:
            console.print()
            question = console.input("[bold yellow]❓ Câu hỏi: [/]").strip()

            if not question:
                continue

            # Commands
            if question.lower() in ("/quit", "/exit", "quit", "exit"):
                console.print("[dim]Tạm biệt! 👋[/]")
                break

            if question.lower() == "/debug":
                show_debug = not show_debug
                state = "BẬT" if show_debug else "TẮT"
                console.print(f"[cyan]🔍 Debug mode: {state}[/]")
                continue

            if question.lower() == "/stats":
                stats = pipeline.get_stats()
                console.print("\n[bold]📊 Pipeline Statistics:[/]")
                for k, v in stats.items():
                    console.print(f"  {k}: [cyan]{v}[/]")
                continue

            if question.lower() == "/tree":
                console.print(pipeline.get_tree_info())
                continue

            if question.lower() == "/benchmark":
                _run_benchmark(pipeline)
                continue

            # Process question
            console.print("[dim]⏳ Đang xử lý...[/]\n")

            result = pipeline.query(question, show_reasoning=show_debug)

            console.print(Panel(
                Markdown(result),
                title="[bold]📋 Kết quả[/]",
                border_style="blue",
                padding=(1, 2),
            ))

        except KeyboardInterrupt:
            console.print("\n[dim]Tạm biệt! 👋[/]")
            break
        except Exception as e:
            console.print(f"[red]❌ Lỗi: {e}[/]")
            import traceback
            if show_debug:
                traceback.print_exc()


def _run_benchmark(pipeline):
    """Run benchmark evaluation and display results."""
    from src.benchmark import BenchmarkRunner
    from pathlib import Path

    console.print(Panel(
        "[bold yellow]📊 Running Benchmark...[/]\n"
        "[dim]Evaluating 12 test questions across 5 categories[/]",
        border_style="yellow",
    ))

    runner = BenchmarkRunner(pipeline)
    report = runner.run(verbose=True)

    # Display results table
    table = Table(title="📊 Benchmark Results", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan", justify="right")

    table.add_row("Total Questions", str(report.total_questions))
    table.add_row("Avg Precision@K", f"{report.avg_precision:.3f}")
    table.add_row("Avg Recall@K", f"{report.avg_recall:.3f}")
    table.add_row("Avg Keyword Coverage", f"{report.avg_keyword_coverage:.3f}")
    table.add_row("Intent Accuracy", f"{report.intent_accuracy:.1%}")
    table.add_row("Avg Confidence", f"{report.avg_confidence:.3f}")
    table.add_row("Avg Retrieval Time", f"{report.avg_retrieval_time_ms:.0f}ms")
    table.add_row("Avg Total Time", f"{report.avg_total_time_ms:.0f}ms")

    console.print(table)

    # Category breakdown
    if report.results_by_category:
        cat_table = Table(title="📋 Results by Category", show_lines=True)
        cat_table.add_column("Category", style="bold")
        cat_table.add_column("Count", justify="right")
        cat_table.add_column("Precision", justify="right")
        cat_table.add_column("Recall", justify="right")
        cat_table.add_column("Keywords", justify="right")
        cat_table.add_column("Intent", justify="right")

        for cat, m in sorted(report.results_by_category.items()):
            cat_table.add_row(
                cat,
                str(m["count"]),
                f"{m['avg_precision']:.3f}",
                f"{m['avg_recall']:.3f}",
                f"{m['avg_keyword_coverage']:.3f}",
                f"{m['intent_accuracy']:.1%}",
            )

        console.print(cat_table)

    # Individual results
    console.print("\n[bold]Individual Results:[/]")
    for i, r in enumerate(report.individual_results, 1):
        intent_icon = "✅" if r.intent_correct else "❌"
        score_color = "green" if r.recall_at_k >= 0.5 else "yellow" if r.recall_at_k > 0 else "red"
        console.print(
            f"  {i:2d}. {intent_icon} P={r.precision_at_k:.2f} "
            f"R=[{score_color}]{r.recall_at_k:.2f}[/] "
            f"KW={r.keyword_coverage:.2f} "
            f"[dim]{r.question[:50]}...[/]"
        )

    # Save report
    report_path = Path("indexes/benchmark_report.json")
    runner.save_report(report, report_path)
    console.print(f"\n[dim]Report saved to {report_path}[/]")

    # Also save Markdown report
    md_report = BenchmarkRunner.format_report(report)
    md_path = Path("indexes/benchmark_report.md")
    md_path.write_text(md_report, encoding="utf-8")
    console.print(f"[dim]Markdown report saved to {md_path}[/]")


def _enhance_kg(pipeline, max_chunks: int = 50):
    """Enhance knowledge graph with LLM extraction."""
    from src.bookindex.kg_enhancer import KGEnhancer

    console.print(Panel(
        f"[bold magenta]🧠 Enhancing Knowledge Graph[/]\n"
        f"[dim]Processing up to {max_chunks} chunks with LLM...[/]",
        border_style="magenta",
    ))

    enhancer = KGEnhancer(pipeline.kg)
    chunks = pipeline.tree.get_leaf_chunks()

    stats = enhancer.enhance_from_chunks(chunks, max_chunks=max_chunks)

    table = Table(title="🧠 KG Enhancement Results", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan", justify="right")

    for k, v in stats.items():
        table.add_row(k, str(v))

    console.print(table)


if __name__ == "__main__":
    main()
