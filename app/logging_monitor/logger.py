"""
LawEdu AI — Structured Logging & Monitoring.
Provides request tracing, metrics, and debug logging.
"""
import logging
import time
import uuid
from functools import wraps
from typing import Optional

# Configure structured logging
def setup_logging(level: int = logging.INFO):
    """Setup application-wide logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Suppress noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


logger = logging.getLogger("lawedu")


class RequestTracer:
    """Traces a single request through the pipeline with timing."""

    def __init__(self, question: str, session_id: str = "default"):
        self.request_id = str(uuid.uuid4())[:8]
        self.question = question
        self.session_id = session_id
        self.start_time = time.time()
        self.steps = []

    def log_step(self, step_name: str, detail: str = "", **kwargs):
        """Log a pipeline step with timing."""
        elapsed = (time.time() - self.start_time) * 1000
        entry = {
            "step": step_name,
            "elapsed_ms": round(elapsed, 1),
            "detail": detail,
            **kwargs,
        }
        self.steps.append(entry)
        logger.info(f"[{self.request_id}] {step_name}: {detail} ({elapsed:.0f}ms)")

    @property
    def total_ms(self) -> float:
        return (time.time() - self.start_time) * 1000

    def summary(self) -> dict:
        return {
            "request_id": self.request_id,
            "question": self.question[:100],
            "session_id": self.session_id,
            "total_ms": round(self.total_ms, 1),
            "steps": self.steps,
        }


class PipelineMetrics:
    """Tracks aggregate pipeline metrics."""

    def __init__(self):
        self.total_requests = 0
        self.total_cache_hits = 0
        self.total_errors = 0
        self.intent_counts = {}
        self.avg_latency_ms = 0.0
        self._latency_sum = 0.0

    def record_request(self, intent: str, latency_ms: float, cache_hit: bool = False):
        self.total_requests += 1
        self._latency_sum += latency_ms
        self.avg_latency_ms = self._latency_sum / self.total_requests
        self.intent_counts[intent] = self.intent_counts.get(intent, 0) + 1
        if cache_hit:
            self.total_cache_hits += 1

    def record_error(self):
        self.total_errors += 1

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_cache_hits": self.total_cache_hits,
            "total_errors": self.total_errors,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "intent_distribution": self.intent_counts,
        }


# Global metrics instance
metrics = PipelineMetrics()
