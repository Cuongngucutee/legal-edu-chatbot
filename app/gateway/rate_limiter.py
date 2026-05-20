"""
LawEdu AI — API Gateway: Rate Limiter.
Simple in-memory rate limiting per IP/session.
"""
import time
import logging
from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Token bucket rate limiter.
    Limits requests per IP per minute.
    """

    SKIP_PATHS = {"/", "/api/health", "/api/stats", "/docs", "/openapi.json"}

    def __init__(self, app, requests_per_minute: int = 30):
        super().__init__(app)
        self.rpm = requests_per_minute
        self._buckets = defaultdict(lambda: {"tokens": requests_per_minute, "last_refill": time.time()})

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier (IP or forwarded IP)."""
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _check_rate(self, client_id: str) -> bool:
        """Check and consume a rate limit token. Returns True if allowed."""
        bucket = self._buckets[client_id]
        now = time.time()

        # Refill tokens based on elapsed time
        elapsed = now - bucket["last_refill"]
        refill = int(elapsed * self.rpm / 60)
        if refill > 0:
            bucket["tokens"] = min(self.rpm, bucket["tokens"] + refill)
            bucket["last_refill"] = now

        if bucket["tokens"] > 0:
            bucket["tokens"] -= 1
            return True
        return False

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip rate limiting for certain paths
        if path in self.SKIP_PATHS or path.startswith("/static"):
            return await call_next(request)

        client_id = self._get_client_id(request)

        if not self._check_rate(client_id):
            logger.warning(f"Rate limit exceeded for {client_id}")
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
            )

        return await call_next(request)
