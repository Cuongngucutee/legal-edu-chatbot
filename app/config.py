"""
LawEdu AI — Centralized Configuration.
Loads from .env + provides typed config objects.
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env from project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(ROOT_DIR, ".env"))


@dataclass
class LLMConfig:
    """Configuration for the Primary LLM API (120B for Generation)."""
    api_base: str = os.getenv("LLM_API_BASE", "https://api.groq.com/openai/v1")
    api_key: str = os.getenv("LLM_API_KEY", "")
    model: str = os.getenv("LLM_MODEL_NAME", "openai/gpt-oss-120b")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "8192"))

@dataclass
class GeneratorLLMConfig:
    """Configuration for the Secondary LLM (20B for intermediate reasoning)."""
    api_base: str = os.getenv("GEN_LLM_API_BASE", "https://api.groq.com/openai/v1")
    api_key: str = os.getenv("GEN_LLM_API_KEY", "")
    model: str = os.getenv("GEN_LLM_MODEL_NAME", "openai/gpt-oss-20b")
    temperature: float = float(os.getenv("GEN_LLM_TEMPERATURE", "0.0"))
    max_tokens: int = int(os.getenv("GEN_LLM_MAX_TOKENS", "8192"))


@dataclass
class ProLLMConfig:
    """Configuration for the Pro LLM API (GPT 120B)."""
    api_base: str = os.getenv("PRO_LLM_API_BASE", "https://api.int2.net/v1")
    api_key: str = os.getenv("PRO_LLM_API_KEY", "")
    model: str = os.getenv("PRO_LLM_MODEL_NAME", "glm-4.7")
    temperature: float = float(os.getenv("PRO_LLM_TEMPERATURE", "0.0"))
    max_tokens: int = int(os.getenv("PRO_LLM_MAX_TOKENS", "8192"))


@dataclass
class RedisConfig:
    """Configuration for Redis cache."""
    url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    ttl: int = int(os.getenv("REDIS_TTL", "1800"))  # 30 minutes
    max_cache_size: int = int(os.getenv("REDIS_MAX_CACHE", "500"))


@dataclass
class APIConfig:
    """Configuration for the API Gateway."""
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    api_key: str = os.getenv("API_KEY", "")
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT", "30"))


@dataclass
class DataConfig:
    """Configuration for data paths."""
    data_dir: str = os.path.join(ROOT_DIR, os.getenv("DATA_DIR", "data/final"))
    kg_path: str = os.path.join(ROOT_DIR, os.getenv("KG_PATH", "outputs/knowledge_graph/entity_graph.json"))


@dataclass
class AppConfig:
    """Master configuration object."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    generator_llm: GeneratorLLMConfig = field(default_factory=GeneratorLLMConfig)
    pro_llm: ProLLMConfig = field(default_factory=ProLLMConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    api: APIConfig = field(default_factory=APIConfig)
    data: DataConfig = field(default_factory=DataConfig)
    root_dir: str = ROOT_DIR


# Singleton
settings = AppConfig()
