"""
Embedding helper — supports FastEmbed, Ollama, OpenAI, and local in-process embeddings.

Backends:
  - FastEmbed: BAAI/bge-small-en-v1.5 (universal in-process neural model, 384-dim, zero server setup)
  - Ollama:    nomic-embed-text (local background server)
  - OpenAI:    text-embedding-3-small (cloud API)
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Optional

import httpx

from app.config import load as load_config

# Persistent clients and model cache
_ollama_client: Optional[httpx.Client] = None
_openai_client: Optional[httpx.Client] = None
_fastembed_model = None


def _get_ollama_client(base_url: str) -> httpx.Client:
    global _ollama_client
    if _ollama_client is None or str(_ollama_client.base_url) != base_url:
        _ollama_client = httpx.Client(base_url=base_url, timeout=60.0)
    return _ollama_client


def _get_openai_client() -> httpx.Client:
    global _openai_client
    if _openai_client is None:
        _openai_client = httpx.Client(base_url="https://api.openai.com/v1", timeout=60.0)
    return _openai_client


def _get_fastembed(model_name: str = "BAAI/bge-small-en-v1.5"):
    global _fastembed_model
    if _fastembed_model is None:
        try:
            from fastembed import TextEmbedding
            _fastembed_model = TextEmbedding(model_name=model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FastEmbed: {e}") from e
    return _fastembed_model


def _embed_fastembed(text: str, model_name: str = "BAAI/bge-small-en-v1.5") -> list[float]:
    try:
        model = _get_fastembed(model_name)
        embeddings = list(model.embed([text]))
        return [float(x) for x in embeddings[0]]
    except Exception:
        # Fallback to in-memory hashing if ONNX/FastEmbed fails
        return _embed_hash(text)


def _embed_ollama(text: str, model: str, base_url: str) -> list[float]:
    client = _get_ollama_client(base_url)
    # Ensure text does not exceed typical 2048 token limits
    safe_text = text[:3500] if len(text) > 3500 else text
    try:
        response = client.post(
            "/api/embeddings",
            json={"model": model, "prompt": safe_text},
        )
    except httpx.ConnectError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {base_url}.\n"
            "Make sure Ollama is running (`ollama serve`) or switch to a cloud provider with `repoatlas init`.\n"
            f"Original error: {e}"
        ) from e

    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama embedding request failed: HTTP {response.status_code}\n"
            f"Body: {response.text[:500]}"
        )

    data = response.json()
    embedding: list[float] = data.get("embedding", [])
    if not embedding:
        raise RuntimeError(
            f"Ollama returned an empty embedding for model {model!r}.\n"
            f"Is the model pulled? Run: ollama pull {model}"
        )
    return embedding


def _embed_openai(text: str, model: str, api_key: str) -> list[float]:
    if not api_key:
        raise ValueError(
            "OpenAI API key missing. Set OPENAI_API_KEY environment variable or run `repoatlas init`."
        )
    client = _get_openai_client()
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = client.post(
            "/embeddings",
            json={"model": model, "input": text},
            headers=headers,
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI embedding request failed: {e}") from e

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenAI embedding error: HTTP {response.status_code}\n{response.text[:500]}"
        )

    data = response.json()
    return data["data"][0]["embedding"]


def _embed_hash(text: str, dim: int = 384) -> list[float]:
    """Deterministic in-memory vectorizer (384 dimensions) used as instant fallback."""
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    vec = [0.0] * dim
    if not tokens:
        return vec

    for i, t in enumerate(tokens):
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 16) % 2 == 0 else -1.0
        vec[idx] += sign * 1.0
        if i > 0:
            bg = f"{tokens[i-1]}_{t}"
            h2 = int(hashlib.sha256(bg.encode("utf-8")).hexdigest(), 16)
            idx2 = h2 % dim
            sign2 = 1.0 if (h2 >> 16) % 2 == 0 else -1.0
            vec[idx2] += sign2 * 1.5

    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def get_embedding(
    text: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> list[float]:
    """
    Return the embedding vector for *text* using the configured provider.

    Parameters
    ----------
    text     : str — snippet or question to embed
    model    : str, optional — model name override
    provider : str, optional — "fastembed", "local", "openai", or "ollama"
    api_key  : str, optional — API key for cloud providers
    base_url : str, optional — base URL for Ollama
    """
    cfg = load_config()

    active_provider = (
        provider
        or cfg.get("embed_provider")
        or ("openai" if cfg.get("provider") == "openai" else None)
        or ("fastembed" if cfg.get("provider") in ("groq", "anthropic") else None)
        or cfg.get("provider", "fastembed")
    ).lower()

    if active_provider in ("fastembed", "local", "in-process", "fast"):
        model_name = model or cfg.get("embed_model") or "BAAI/bge-small-en-v1.5"
        if model_name == "nomic-embed-text":
            model_name = "BAAI/bge-small-en-v1.5"
        return _embed_fastembed(text, model_name=model_name)

    elif active_provider == "openai":
        active_model = model or cfg.get("embed_model") or "text-embedding-3-small"
        if active_model in ("nomic-embed-text", "BAAI/bge-small-en-v1.5"):
            active_model = "text-embedding-3-small"
        key = api_key or cfg.get("openai_api_key") or cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
        return _embed_openai(text, active_model, key or "")

    else:
        # Default: Ollama
        active_model = model or cfg.get("embed_model") or "nomic-embed-text"
        url = base_url or cfg.get("ollama_base_url") or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return _embed_ollama(text, active_model, url)
