"""
LawEdu AI — API Gateway: Authentication Middleware.
"""
import logging
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Simple API key authentication middleware.
    Skips auth for:
    - Health check endpoints
    - Static files
    - Root page
    """

    SKIP_PATHS = {"/", "/api/health", "/docs", "/openapi.json", "/redoc"}

    def __init__(self, app, api_key: str = ""):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip auth for certain paths
        if path in self.SKIP_PATHS or path.startswith("/static"):
            return await call_next(request)

        # Skip if no API key is configured (open access)
        if not self.api_key:
            return await call_next(request)

        # Check Authorization header
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
        else:
            token = request.headers.get("X-API-Key", "")

        if token != self.api_key:
            # Also allow requests from the web UI (same-origin, no auth header)
            referer = request.headers.get("Referer", "")
            origin = request.headers.get("Origin", "")
            if referer or origin:
                # Likely from web UI, allow
                return await call_next(request)

            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        return await call_next(request)
