"""
Vector retrieval — embeds a question and finds the most similar code chunks
using ChromaDB cosine similarity. No SQL, no server.
"""
from typing import Optional

import chromadb

from indexer.embedder import get_embedding

# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


def retrieve(
    question: str,
    collection: chromadb.Collection,
    repo_path: Optional[str] = None,
    top_k: int = 5,
    embed_model: Optional[str] = None,
    embed_provider: Optional[str] = None,
    api_key: Optional[str] = None,
) -> list[dict]:
    """
    Embed *question* and return the *top_k* most similar chunks.

    Parameters
    ----------
    question       : str — natural-language question from the user
    collection     : chromadb.Collection — the RepoAtlas chunks collection
    repo_path      : str, optional — restrict results to a specific indexed repo
    top_k          : int — number of chunks to return (default 5)
    embed_model    : str, optional — override the embedding model
    embed_provider : str, optional — override embedding provider ("ollama", "openai", "local")
    api_key        : str, optional — API key for cloud embedding provider

    Returns
    -------
    list of dicts, each with keys:
        repo_path, file_path, chunk_type, name,
        start_line, end_line, code, similarity
    """
    kwargs: dict = {}
    if embed_model:
        kwargs["model"] = embed_model
    if embed_provider:
        kwargs["provider"] = embed_provider
    if api_key:
        kwargs["api_key"] = api_key

    q_embedding = get_embedding(question, **kwargs)

    query_kwargs: dict = {
        "query_embeddings": [q_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }

    if repo_path:
        query_kwargs["where"] = {"repo_path": repo_path}

    try:
        results = collection.query(**query_kwargs)
    except Exception as exc:
        # ChromaDB raises if collection is empty or top_k > collection size
        msg = str(exc).lower()
        if any(k in msg for k in ("no elements", "cannot query", "index not found", "greater than")):
            return []
        raise

    docs  = results.get("documents",  [[]])[0]
    metas = results.get("metadatas",  [[]])[0]
    dists = results.get("distances",  [[]])[0]

    chunks = []
    for doc, meta, dist in zip(docs, metas, dists):
        # cosine space: distance ∈ [0, 2], but in practice [0, 1] for normalised
        # vectors. similarity = 1 − distance gives a clean 0–1 score.
        chunks.append({
            "repo_path":  meta.get("repo_path", ""),
            "file_path":  meta.get("file_path", ""),
            "chunk_type": meta.get("chunk_type", ""),
            "name":       meta.get("name", ""),
            "start_line": meta.get("start_line", 0),
            "end_line":   meta.get("end_line", 0),
            "code":       doc,
            "similarity": round(1.0 - dist, 4),
        })

    return chunks
