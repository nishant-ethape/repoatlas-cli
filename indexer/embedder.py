"""
Embedding helper — wraps Ollama's /api/embeddings endpoint.

Default model: nomic-embed-text (768-dim output, matches schema.EMBEDDING_DIM).
"""
import os
from typing import Optional

import httpx

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_EMBED_MODEL = os.environ.get("REPOATLAS_EMBED_MODEL", "nomic-embed-text")

# Re-use a single httpx client across calls — avoids repeated TCP handshakes.
_client: Optional[httpx.Client] = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(base_url=OLLAMA_BASE_URL, timeout=60.0)
    return _client


def get_embedding(text: str, model: str = DEFAULT_EMBED_MODEL) -> list[float]:
    """
    Return the embedding vector for *text* using Ollama's local API.

    Raises RuntimeError if Ollama is unreachable or returns an error status.
    """
    client = _get_client()
    try:
        response = client.post(
            "/api/embeddings",
            json={"model": model, "prompt": text},
        )
    except httpx.ConnectError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}.\n"
            "Make sure Ollama is running (`ollama serve`).\n"
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
            "Is the model pulled? Run: ollama pull nomic-embed-text"
        )
    return embedding
