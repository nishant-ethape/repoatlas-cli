"""
Pluggable LLM provider abstraction.

Supports:
  - OllamaProvider    (local, free, default — no API key needed)
  - OpenAIProvider    (requires OPENAI_API_KEY)
  - AnthropicProvider (requires ANTHROPIC_API_KEY)
  - GroqProvider      (requires GROQ_API_KEY)

All providers implement the same interface:
    provider.chat(system: str, user: str) -> str

Factory:
    get_provider(config: dict) -> LLMProvider
"""
from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

import httpx


# --------------------------------------------------------------------------- #
# Protocol (interface)
# --------------------------------------------------------------------------- #

@runtime_checkable
class LLMProvider(Protocol):
    def chat(self, system: str, user: str) -> str:
        """Send a system + user message pair and return the assistant response."""
        ...


# --------------------------------------------------------------------------- #
# Ollama (local)
# --------------------------------------------------------------------------- #

class OllamaProvider:
    """
    Chat via Ollama's /api/chat endpoint.
    Default model: qwen2.5-coder:3b (fits in 4 GB VRAM).
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder:3b",
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self._client = httpx.Client(base_url=self.base_url, timeout=120.0)

    def chat(self, system: str, user: str) -> str:
        try:
            resp = self._client.post(
                "/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Run `ollama serve` first, or run `repoatlas init` to switch to a cloud provider.\n"
                f"Details: {e}"
            ) from e

        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama chat error: HTTP {resp.status_code}\n{resp.text[:500]}"
            )
        return resp.json()["message"]["content"]

    def __repr__(self) -> str:
        return f"OllamaProvider(model={self.model!r}, base_url={self.base_url!r})"


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #

class OpenAIProvider:
    """Chat via the OpenAI API (uses httpx or openai client)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY or run `repoatlas init`."
            )
        self._client = httpx.Client(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120.0,
        )

    def chat(self, system: str, user: str) -> str:
        try:
            resp = self._client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to OpenAI API: {e}") from e

        if resp.status_code != 200:
            raise RuntimeError(
                f"OpenAI API error: HTTP {resp.status_code}\n{resp.text[:500]}"
            )
        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""

    def __repr__(self) -> str:
        return f"OpenAIProvider(model={self.model!r})"


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #

class AnthropicProvider:
    """Chat via the Anthropic API."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-latest",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY or run `repoatlas init`."
            )
        self._client = httpx.Client(
            base_url="https://api.anthropic.com/v1",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            timeout=120.0,
        )

    def chat(self, system: str, user: str) -> str:
        try:
            resp = self._client.post(
                "/messages",
                json={
                    "model": self.model,
                    "max_tokens": 2048,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
            )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Anthropic API: {e}") from e

        if resp.status_code != 200:
            raise RuntimeError(
                f"Anthropic API error: HTTP {resp.status_code}\n{resp.text[:500]}"
            )
        data = resp.json()
        return data["content"][0]["text"]

    def __repr__(self) -> str:
        return f"AnthropicProvider(model={self.model!r})"


# --------------------------------------------------------------------------- #
# Groq
# --------------------------------------------------------------------------- #

class GroqProvider:
    """Chat via the Groq API (ultra-fast inference)."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY or run `repoatlas init`."
            )
        self._client = httpx.Client(
            base_url="https://api.groq.com/openai/v1",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120.0,
        )

    def chat(self, system: str, user: str) -> str:
        try:
            resp = self._client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Groq API: {e}") from e

        if resp.status_code != 200:
            raise RuntimeError(
                f"Groq API error: HTTP {resp.status_code}\n{resp.text[:500]}"
            )
        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""

    def __repr__(self) -> str:
        return f"GroqProvider(model={self.model!r})"


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

_PROVIDERS = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "groq": GroqProvider,
}


def get_provider(config: dict | None = None) -> LLMProvider:
    """
    Build an LLMProvider from a config dict.

    Config keys (all optional):
        provider  : one of "ollama", "openai", "anthropic", "groq"
                    (env: REPOATLAS_PROVIDER, default: "ollama")
        model     : model name override (chat_model or model)
                    (env: REPOATLAS_MODEL)
        api_key   : API key (only for cloud providers)
        base_url  : Ollama base URL override
    """
    cfg = config or {}

    provider_name = (
        cfg.get("provider")
        or os.environ.get("REPOATLAS_PROVIDER", "ollama")
    ).lower()

    model = cfg.get("chat_model") or cfg.get("model") or os.environ.get("REPOATLAS_MODEL")
    api_key = (
        cfg.get("api_key")
        or cfg.get(f"{provider_name}_api_key")
        or os.environ.get(f"{provider_name.upper()}_API_KEY")
    )
    base_url = cfg.get("ollama_base_url") or cfg.get("base_url")

    cls = _PROVIDERS.get(provider_name)
    if cls is None:
        raise ValueError(
            f"Unknown provider {provider_name!r}. "
            f"Choose from: {', '.join(_PROVIDERS)}"
        )

    kwargs: dict = {}
    if model:
        kwargs["model"] = model
    if api_key:
        kwargs["api_key"] = api_key
    if base_url and provider_name == "ollama":
        kwargs["base_url"] = base_url

    return cls(**kwargs)
