"""
ChromaDB client and collection management for RepoAtlas.

Zero server setup — ChromaDB persists to a local directory.

Default path : ~/.repoatlas/chroma_db
Override     : REPOATLAS_CHROMA_PATH env var  or  --chroma-path CLI flag
"""
import os
from pathlib import Path
from typing import Optional

import chromadb

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

COLLECTION_NAME = "chunks"
_DEFAULT_PATH = str(Path.home() / ".repoatlas" / "chroma_db")


def get_chroma_path() -> str:
    """Return the configured ChromaDB directory (env var or default)."""
    return os.environ.get("REPOATLAS_CHROMA_PATH", _DEFAULT_PATH)


# --------------------------------------------------------------------------- #
# Client / collection helpers
# --------------------------------------------------------------------------- #

def get_client(chroma_path: Optional[str] = None) -> chromadb.PersistentClient:
    """
    Create a PersistentClient at *chroma_path* (or the configured default).

    The directory is created automatically if it doesn't exist.
    """
    path = chroma_path or get_chroma_path()
    Path(path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=path)


def get_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """
    Get (or create) the chunks collection, configured for cosine similarity.

    Using cosine space means similarity = 1 − distance, giving a clean 0–1
    score where 1.0 = identical.
    """
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """Drop and recreate the collection — erases ALL indexed repos."""
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return get_collection(client)


def clear_repo(collection: chromadb.Collection, repo_path: str) -> int:
    """
    Delete all chunks belonging to *repo_path*.

    Returns the number of deleted chunks (0 if none were present).
    Used by the non-reset indexing path so multiple repos can coexist.
    """
    try:
        existing = collection.get(where={"repo_path": repo_path})
        ids = existing.get("ids", [])
        if ids:
            collection.delete(ids=ids)
        return len(ids)
    except Exception:
        return 0
