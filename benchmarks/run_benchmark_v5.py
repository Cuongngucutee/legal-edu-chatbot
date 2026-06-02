"""
Education Benchmark v5 — New 320B Pipeline
=====================================================
Chạy CÙNG bộ 12 câu hỏi từ education_benchmark_v2.json
trên pipeline mới (app.main.PIPELINE) với Pro LLM API.
"""
import os
import sys
import json
import time
import re
from collections import defaultdict

# Setup paths — benchmarks/ is a subfolder, project root is parent
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from app.config import settings
from app.logging_monitor.logger import setup_logging
setup_logging()

# Load the new pipeline
from app.main import init_pipeline, PIPELINE

# ══════════════════════════════════════════════════════════════════
# Judge Prompt — GIỐNG HỆT TestRAG benchmark v2
# ══════════════════════════════════════════════════════════════════

JUDGE_PROMPT = """Bạn là giám khảo đánh giá chatbot tư vấn pháp luật giáo dục. Hãy chấm NGHIÊM KHẮC.

## Câu hỏi
{question}

## Căn cứ pháp lý cần trích dẫn
{legal_basis}

## Đáp án tham khảo (Ground Truth)
{reference_answer}

## Từ khóa bắt buộc phải có trong câu trả lời
{keywords}

## Câu trả lời của chatbot
---
{answer}
---

## Tiêu chí chấm điểm (0-5 mỗi tiêu chí)

### 1. VERDICT (Khẳng định rõ ràng)
- 5: Khẳng định chính xác, rõ ràng ngay từ đầu
- 3: Có khẳng định nhưng chưa rõ ràng
- 0: Không có hoặc sai

### 2. CITATION (Dẫn chiếu văn bản pháp lý)
- 5: Trích dẫn đúng và đầy đủ [Tên VB, Điều X, Khoản Y]
- 3: Có nhưng thiếu hoặc không chính xác
- 0: Không trích dẫn

### 3. EXPLANATION (Giải thích chi tiết)
- 5: Giải thích đầy đủ, chính xác
- 3: Cơ bản đúng nhưng thiếu
- 0: Sai hoặc không có

### 4. ACCURACY (Độ chính xác nội dung)
- 5: Mọi thông tin đều chính xác
- 3: Phần lớn đúng
- 0: Sai hoàn toàn

### 5. KEYWORD_COVERAGE (Bao phủ từ khóa)
- 5: Đề cập đầy đủ tất cả từ khóa bắt buộc
- 3: 50-80%
- 0: Không đề cập

Trả về JSON (chỉ JSON):
{{"verdict": ?, "citation": ?, "explanation": ?, "accuracy": ?, "keyword_coverage": ?, "comment": "nhận xét ngắn"}}"""

def check_retrieval_from_context(context_text: str, ground_truth: dict) -> float:
    """Check retrieval quality by matching ground truth docs/articles in context text."""
    gt_docs = set(ground_truth.get("relevant_docs", []))
    gt_articles = set(ground_truth.get("relevant_articles", []))
    if not gt_docs:
        return 0.0

    found_docs = set()
    found_articles = set()
    ctx_lower = context_text.lower()

    for gt_doc in gt_docs:
        gt_nums = set(re.findall(r'\d+', gt_doc))
        matches = sum(1 for n in gt_nums if n in ctx_lower)
        if gt_nums and matches >= min(2, len(gt_nums)):
            found_docs.add(gt_doc)

    for gt_art in gt_articles:
        art_num = re.search(r'\d+', gt_art)
        if art_num:
            pattern = rf'[Đđ]iều\s+{art_num.group()}\b'
            if re.search(pattern, context_text):
                found_articles.add(gt_art)

    doc_score = len(found_docs) / len(gt_docs) if gt_docs else 1.0
    art_score = len(found_articles) / len(gt_articles) if gt_articles else 1.0
    return round(0.6 * doc_score + 0.4 * art_score, 2)


def run_benchmark():
    print("=" * 70)
    print("📊 EDUCATION LAW BENCHMARK v5 — New 320B Pipeline")
    print("=" * 70)

    bench_path = os.path.join(os.path.dirname(__file__), "education_benchmark_v2.json")
    if not os.path.exists(bench_path):
        bench_path = os.path.join(PROJECT_ROOT, "..", "TestRAG", "benchmark_results", "education_benchmark_v2.json")
        if not os.path.exists(bench_path):
            print(f"❌ Benchmark file not found.")
            sys.exit(1)

    with open(bench_path, encoding="utf-8") as f:
        bench_data = json.load(f)
    questions = bench_data["questions"]
    print(f"📋 Loaded {len(questions)} benchmark questions\n")

    init_pipeline()
    import app.main
    pipeline = app.main.PIPELINE
    
    # We will use the same Pro LLM API for grading
    llm_judge = pipeline.llm

    all_results = []
    total_start = time.time()

    for q in questions:
        qid = q["id"]
        question = q["question"]
        difficulty = q["difficulty"]
        legal_basis = q["legal_basis"]
        gt = q["ground_truth"]

        print(f"{'─' * 70}")
        print(f"🔹 {qid} [{difficulty.upper()}] Q: {question[:80]}...")

        # Run pipeline
        q_start = time.time()
        result = pipeline.run(question)
        total_q_time = time.time() - q_start
        
        answer = result["answer"]
        intent = result["intent"]
        
        # Build raw context block to check hit rate
        raw_context = "\n".join([src.get("breadcrumb", "") for src in result.get("sources", [])])
        retrieval_score = check_retrieval_from_context(raw_context, gt)

        # Judge
        judge_scores = {
            "verdict": 3, "citation": 3, "explanation": 3,
            "accuracy": 3, "keyword_coverage": 3, "comment": "default"
        }
        
        try:
            keywords_str = ", ".join(gt.get("expected_answer_keywords", []))
            jr = llm_judge.generate(JUDGE_PROMPT.format(
                question=question,
                legal_basis=legal_basis,
                reference_answer=gt.get("reference_answer", "N/A"),
                keywords=keywords_str,
                answer=answer[:1500],
            ), temperature=0.0)
            
            match = re.search(r'\{[^}]+\}', jr, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                for k in ["verdict", "citation", "explanation", "accuracy", "keyword_coverage"]:
                    if k in parsed and isinstance(parsed[k], (int, float)) and 0 <= parsed[k] <= 5:
                        judge_scores[k] = parsed[k]
                if "comment" in parsed:
                    judge_scores["comment"] = str(parsed["comment"])
        except Exception as e:
            print(f"   ⚠️ Judge error: {e}")

        composite = (
            judge_scores["verdict"] * 0.25 +
            judge_scores["citation"] * 0.25 +
            judge_scores["explanation"] * 0.20 +
            judge_scores["accuracy"] * 0.20 +
            judge_scores["keyword_coverage"] * 0.10
        )

        print(f"   Intent={intent} | ⏱️ {total_q_time:.1f}s | Ret={retrieval_score:.0%}")
        print(f"   RAW ANSWER: {answer[:300]}...")
        print(f"   V={judge_scores['verdict']} C={judge_scores['citation']} E={judge_scores['explanation']} A={judge_scores['accuracy']} K={judge_scores['keyword_coverage']} → {composite:.2f}/5")
        print(f"   💬 {judge_scores.get('comment', '')[:80]}")

        break # STOP AFTER 1 QUESTION FOR DEBUGGING

        all_results.append({
            "qid": qid,
            "difficulty": difficulty,
            "question": question,
            "intent_detected": intent,
            "retrieval_score": retrieval_score,
            "total_time_s": round(total_q_time, 2),
            "judge_scores": judge_scores,
            "composite_score": round(composite, 2),
            "answer_preview": answer[:500],
        })

    n = len(all_results)
    avg_ret = sum(r["retrieval_score"] for r in all_results) / n
    avg_composite = sum(r["composite_score"] for r in all_results) / n
    
    print(f"\n{'=' * 70}")
    print(f"📊 BENCHMARK v5 REPORT (Model: {settings.llm.model})")
    print(f"{'=' * 70}")
    print(f"📈 Overall Retrieval Hit Rate: {avg_ret:.1%}")
    print(f"📈 Overall Composite Score:    {avg_composite:.2f}/5\n")

if __name__ == "__main__":
    run_benchmark()
