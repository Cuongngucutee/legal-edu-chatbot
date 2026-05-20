"""
LawEdu AI — LLM Client.
Unified 320B LLM API client replacing all local models (1.5B + 7B).
Uses OpenAI-compatible API via LiteLLM gateway.
"""
import logging
from typing import Optional, Generator

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified LLM client for the 320B model API.
    Replaces:
    - LLMWrapper (Ollama qwen2.5:7b)
    - QwenIntentClassifier (local 1.5B transformers)
    - BookRAGGenerator
    - RAGAgents (self-check, judge)
    """

    def __init__(
        self,
        api_base: str = "https://api.int2.net/v1",
        api_key: str = "",
        model: str = "deepseek-chat",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        """Lazy-initialize OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
            )
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text from prompt (non-streaming)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
            raw = response.choices[0].message.content
            return (raw or "").strip()
        except Exception as e:
            logger.error(f"LLM generate error: {e}")
            return f"[LLM Error: {e}]"

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """Generate text with streaming — yields tokens one by one."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=True,
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"LLM stream error: {e}")
            yield f"[LLM Error: {e}]"

    @classmethod
    def from_config(cls, config) -> "LLMClient":
        """Create LLMClient from AppConfig.llm."""
        return cls(
            api_base=config.api_base,
            api_key=config.api_key,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
