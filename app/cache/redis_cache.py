"""
LawEdu AI — Cache Layer.
Redis-backed cache with in-memory fallback.
Handles query caching, session storage, and conversation history.
"""
import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Optional, Any

logger = logging.getLogger(__name__)


class CacheBackend:
    """Abstract cache backend interface."""

    def get(self, key: str) -> Optional[str]:
        raise NotImplementedError

    def set(self, key: str, value: str, ttl: int = 1800) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError


class RedisCacheBackend(CacheBackend):
    """Redis-backed cache. Falls back to InMemory if Redis is unavailable."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self._client = None
        self._redis_url = redis_url
        self._available = False
        self._try_connect()

    def _try_connect(self):
        try:
            import redis
            self._client = redis.from_url(self._redis_url, decode_responses=True)
            self._client.ping()
            self._available = True
            logger.info("✅ Redis connected successfully")
        except Exception as e:
            logger.warning(f"⚠️ Redis unavailable ({e}), using in-memory fallback")
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def get(self, key: str) -> Optional[str]:
        if not self._available:
            return None
        try:
            return self._client.get(key)
        except Exception:
            return None

    def set(self, key: str, value: str, ttl: int = 1800) -> None:
        if not self._available:
            return
        try:
            self._client.setex(key, ttl, value)
        except Exception:
            pass

    def delete(self, key: str) -> None:
        if not self._available:
            return
        try:
            self._client.delete(key)
        except Exception:
            pass

    def exists(self, key: str) -> bool:
        if not self._available:
            return False
        try:
            return self._client.exists(key) > 0
        except Exception:
            return False


class InMemoryCacheBackend(CacheBackend):
    """Simple in-memory LRU cache with TTL support."""

    def __init__(self, maxsize: int = 500, ttl: int = 1800):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl

    def get(self, key: str) -> Optional[str]:
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["time"] < self._ttl:
                self._cache.move_to_end(key)
                return entry["value"]
            del self._cache[key]
        return None

    def set(self, key: str, value: str, ttl: int = None) -> None:
        self._cache[key] = {"value": value, "time": time.time()}
        self._cache.move_to_end(key)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def delete(self, key: str) -> None:
        self._cache.pop(key, None)

    def exists(self, key: str) -> bool:
        return self.get(key) is not None


class QueryCache:
    """
    High-level query cache for the RAG pipeline.
    Maps user questions → full response dicts.
    Uses Redis if available, falls back to in-memory.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0", maxsize: int = 500, ttl: int = 1800):
        self._ttl = ttl
        # Try Redis first
        self._redis = RedisCacheBackend(redis_url)
        if self._redis.is_available:
            self._backend = self._redis
            logger.info("📦 QueryCache using Redis backend")
        else:
            self._backend = InMemoryCacheBackend(maxsize=maxsize, ttl=ttl)
            logger.info("📦 QueryCache using in-memory backend")

        # Stats tracking
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(question: str) -> str:
        """Generate cache key from question."""
        return f"qcache:{hashlib.md5(question.strip().lower().encode()).hexdigest()}"

    def get(self, question: str) -> Optional[dict]:
        """Look up cached response for a question."""
        key = self._key(question)
        raw = self._backend.get(key)
        if raw:
            self.hits += 1
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return None
        self.misses += 1
        return None

    def put(self, question: str, result: dict) -> None:
        """Cache a response for a question."""
        key = self._key(question)
        try:
            self._backend.set(key, json.dumps(result, ensure_ascii=False), ttl=self._ttl)
        except (TypeError, ValueError):
            pass

    def stats(self) -> dict:
        """Return cache statistics."""
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hits / total:.0%}" if total > 0 else "0%",
            "backend": "redis" if isinstance(self._backend, RedisCacheBackend) else "in-memory",
        }
