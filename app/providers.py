"""
Pluggable LLM provider abstraction.

Supports:
  - OllamaProvider  (local, free, default — no API key needed)
  - OpenAIProvider  (requires OPENAI_API_KEY, optional dep: openai)
  - AnthropicProvider (requires ANTHROPIC_API_KEY, optional dep: anthropic)
  - GroqProvider    (requires GROQ_API_KEY, optional dep: groq)

All providers implement the same interface:
    provider.chat(system: str, user: str) -> str

Factory:
    get_provider(config: dict) -> LLMProvider

Config dict keys:
    provider  : "ollama" | "openai" | "anthropic" | "groq"  (default: "ollama")
    model     : model name string
    api_key   : API key (falls back to env var if omitted)
    base_url  : override for Ollama base URL
"""
import os
from typing import Protocol, runtime_checkable


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
        import httpx  # always available

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
                "Run `ollama serve` first.\n"
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
    """Chat via the OpenAI API. Requires `pip install openai`."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "The 'openai' package is required for OpenAI provider.\n"
                "Install it: pip install 'repoatlas[openai]'"
            ) from e

        self.model = model
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY or pass api_key=."
            )
        self._client = OpenAI(api_key=key)

    def chat(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    def __repr__(self) -> str:
        return f"OpenAIProvider(model={self.model!r})"


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #

class AnthropicProvider:
    """Chat via the Anthropic API. Requires `pip install anthropic`."""

    def __init__(
        self,
        model: str = "claude-3-5-haiku-latest",
        api_key: str | None = None,
    ) -> None:
        try:
            import anthropic as _anthropic  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' package is required for Anthropic provider.\n"
                "Install it: pip install 'repoatlas[anthropic]'"
            ) from e

        self.model = model
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY or pass api_key=."
            )
        self._client = _anthropic.Anthropic(api_key=key)

    def chat(self, system: str, user: str) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text  # type: ignore[index]

    def __repr__(self) -> str:
        return f"AnthropicProvider(model={self.model!r})"


# --------------------------------------------------------------------------- #
# Groq
# --------------------------------------------------------------------------- #

class GroqProvider:
    """Chat via the Groq API. Requires `pip install groq`."""

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        api_key: str | None = None,
    ) -> None:
        try:
            from groq import Groq  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "The 'groq' package is required for Groq provider.\n"
                "Install it: pip install 'repoatlas[groq]'"
            ) from e

        self.model = model
        key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY or pass api_key=."
            )
        self._client = Groq(api_key=key)

    def chat(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

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
        model     : model name override
                    (env: REPOATLAS_MODEL)
        api_key   : API key (only for cloud providers)
        base_url  : Ollama base URL override

    Examples
    --------
    get_provider()                              # → OllamaProvider defaults
    get_provider({"provider": "openai"})        # → OpenAIProvider(gpt-4o-mini)
    get_provider({"provider": "groq", "model": "mixtral-8x7b-32768"})
    """
    cfg = config or {}

    provider_name = (
        cfg.get("provider")
        or os.environ.get("REPOATLAS_PROVIDER", "ollama")
    ).lower()

    model = cfg.get("model") or os.environ.get("REPOATLAS_MODEL")
    api_key = cfg.get("api_key")
    base_url = cfg.get("base_url")

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
