"""
LLM provider abstraction.

Supported out of the box:
  - ollama   (local, free — mistral, llama3, gemma, phi, …)
  - openai   (GPT-4o, GPT-3.5; also works with LM Studio's OpenAI-compat API)

To add your own provider, subclass BaseLLMProvider and register it in get_llm_provider().
"""
from __future__ import annotations
from abc import ABC, abstractmethod

from doc_pipeline.core.config import LLMConfig


class BaseLLMProvider(ABC):
    @abstractmethod
    async def complete(self, system: str, user: str) -> str:
        """Send a system + user turn and return the assistant's reply as a string."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
#  Ollama  (pip install ollama)
# ─────────────────────────────────────────────────────────────────────────────

class OllamaProvider(BaseLLMProvider):
    def __init__(self, config: LLMConfig) -> None:
        import ollama
        self._client = ollama.AsyncClient(host=config.base_url)
        self._model = config.model
        self._temperature = config.temperature

    async def complete(self, system: str, user: str) -> str:
        response = await self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            options={"temperature": self._temperature},
        )
        return response["message"]["content"]


# ─────────────────────────────────────────────────────────────────────────────
#  OpenAI / LM Studio  (pip install openai)
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIProvider(BaseLLMProvider):
    """
    Works with:
      - OpenAI API  (model="gpt-4o",  api_key="sk-…", base_url omitted)
      - LM Studio   (model="<any>",   api_key="lm-studio", base_url="http://localhost:1234/v1")
    """
    def __init__(self, config: LLMConfig) -> None:
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            api_key=config.api_key or "sk-no-key",
            base_url=config.base_url if config.base_url != "http://localhost:11434" else None,
        )
        self._model = config.model
        self._temperature = config.temperature
        self._max_tokens = config.max_tokens

    async def complete(self, system: str, user: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return resp.choices[0].message.content or ""


# ─────────────────────────────────────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_llm_provider(config: LLMConfig) -> BaseLLMProvider:
    match config.provider:
        case "ollama":
            return OllamaProvider(config)
        case "openai" | "lmstudio":
            return OpenAIProvider(config)
        case _:
            raise ValueError(
                f"Unknown LLM provider '{config.provider}'. "
                "Supported: 'ollama', 'openai', 'lmstudio'."
            )
