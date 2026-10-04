"""
Vector retrieval — embeds a question and finds the most similar code chunks
using ChromaDB cosine similarity. No SQL, no server.

On top of the raw vector search, results are post-processed so the limited
top-k slots are spent well:

  1. Over-fetch a larger candidate pool than top_k.
  2. Path-aware keyword boost — chunks whose file path / symbol name contains
     words from the question are ranked a little higher (hybrid search).
  3. De-duplicate — the same file indexed from two different roots no longer
     eats two slots.
  4. Diversity — at most ``max_per_file`` chunks per file, so one big file
     cannot crowd out the rest of the repository.
"""
import re
from typing import Optional

import chromadb

from indexer.embedder import get_embedding
from indexer.repomap import REPO_MAP_CHUNK_TYPE

_POOL_FACTOR = 4          # candidates fetched per requested chunk
_MIN_POOL = 20
_BOOST_PER_MATCH = 0.03   # ranking bonus per question word found in the path/name
_MAX_BOOST = 0.10

_STOPWORDS = {
    "the", "and", "for", "how", "does", "what", "where", "which", "that", "this",
    "with", "are", "all", "inside", "name", "list", "show", "work", "works", "into",
    "from", "about", "code", "file", "files", "can", "you", "me", "tell", "explain",
}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _question_words(question: str) -> set[str]:
    words = {w for w in re.findall(r"[a-z0-9]{3,}", question.lower()) if w not in _STOPWORDS}
    return words | {w[:-1] for w in words if w.endswith("s") and len(w) > 3}


def _lexical_boost(words: set[str], file_path: str, name: str) -> float:
    """Small ranking bonus for each question word that appears in the path or symbol name."""
    if not words:
        return 0.0
    haystack = set(re.findall(r"[a-z0-9]+", f"{file_path} {name}".lower()))
    return min(_MAX_BOOST, _BOOST_PER_MATCH * len(words & haystack))


def _norm_path(path: str) -> str:
    return path.replace("\\", "/").lower()


def _is_duplicate(a: dict, b: dict) -> bool:
    """Same content at the same place, even if indexed from different repo roots."""
    if a["code"] != b["code"] or a["name"] != b["name"] or a["start_line"] != b["start_line"]:
        return False
    pa, pb = _norm_path(a["file_path"]), _norm_path(b["file_path"])
    return pa == pb or pa.endswith("/" + pb) or pb.endswith("/" + pa)


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
    max_per_file: int = 2,
) -> list[dict]:
    """
    Embed *question* and return the *top_k* most relevant, de-duplicated chunks.

    Parameters
    ----------
    question       : str — natural-language question from the user
    collection     : chromadb.Collection — the RepoAtlas chunks collection
    repo_path      : str, optional — restrict results to a specific indexed repo
    top_k          : int — number of chunks to return (default 5)
    embed_model    : str, optional — override the embedding model
    embed_provider : str, optional — override embedding provider ("ollama", "openai", "local")
    api_key        : str, optional — API key for cloud embedding provider
    max_per_file   : int — maximum chunks returned from any single file (default 2)

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

    pool = max(top_k * _POOL_FACTOR, _MIN_POOL)

    # Repo-map chunks are for structural questions only (see app/rag.py);
    # they must not compete with real code in similarity search.
    conditions: list[dict] = [{"chunk_type": {"$ne": REPO_MAP_CHUNK_TYPE}}]
    if repo_path:
        conditions.append({"repo_path": repo_path})
    where = conditions[0] if len(conditions) == 1 else {"$and": conditions}

    query_kwargs: dict = {
        "query_embeddings": [q_embedding],
        "n_results": pool,
        "include": ["documents", "metadatas", "distances"],
        "where": where,
    }

    try:
        results = collection.query(**query_kwargs)
    except Exception as exc:
        msg = str(exc).lower()
        if "dimension" in msg:
            raise RuntimeError(
                "This index was built with a different embedding model than the one "
                "now configured (vector sizes differ).\n"
                "Re-index with the current settings:  repoatlas index <repo_path> --reset"
            ) from exc
        # ChromaDB raises if collection is empty or top_k > collection size
        if any(k in msg for k in ("no elements", "cannot query", "index not found", "greater than")):
            return []
        raise

    docs  = results.get("documents",  [[]])[0]
    metas = results.get("metadatas",  [[]])[0]
    dists = results.get("distances",  [[]])[0]

    words = _question_words(question)
    candidates = []
    for doc, meta, dist in zip(docs, metas, dists):
        # cosine space: distance ∈ [0, 2], but in practice [0, 1] for normalised
        # vectors. similarity = 1 − distance gives a clean 0–1 score.
        similarity = round(1.0 - dist, 4)
        file_path = meta.get("file_path", "")
        name = meta.get("name", "")
        candidates.append({
            "repo_path":  meta.get("repo_path", ""),
            "file_path":  file_path,
            "chunk_type": meta.get("chunk_type", ""),
            "name":       name,
            "start_line": meta.get("start_line", 0),
            "end_line":   meta.get("end_line", 0),
            "code":       doc,
            "similarity": similarity,
            "_score":     similarity + _lexical_boost(words, file_path, name),
        })

    candidates.sort(key=lambda c: c["_score"], reverse=True)

    chunks: list[dict] = []
    per_file: dict[str, int] = {}
    for cand in candidates:
        if len(chunks) >= top_k:
            break
        if any(_is_duplicate(cand, kept) for kept in chunks):
            continue
        key = _norm_path(cand["file_path"])
        if per_file.get(key, 0) >= max_per_file:
            continue
        per_file[key] = per_file.get(key, 0) + 1
        chunks.append(cand)

    for c in chunks:
        c.pop("_score", None)
    return chunks
