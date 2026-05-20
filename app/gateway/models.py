"""
LawEdu AI — API Gateway: Pydantic Models.
Request/Response schemas for the chat API.
"""
from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    """Incoming chat request."""
    question: str
    session_id: Optional[str] = None
    include_sources: bool = True
    stream: bool = False


class ChatResponse(BaseModel):
    """Chat response (non-streaming)."""
    answer: str
    sources: list = []
    intent: Optional[str] = None
    skill_used: Optional[str] = None
    warning: Optional[str] = None
    iterations: int = 1
    latency_ms: float = 0.0


class SessionResponse(BaseModel):
    """New session response."""
    session_id: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    pipeline: bool
    llm_model: Optional[str] = None
    cache_backend: Optional[str] = None


class StatsResponse(BaseModel):
    """Pipeline statistics response."""
    sessions: int = 0
    nodes: int = 0
    model: str = ""
    cache: dict = {}
    metrics: dict = {}
