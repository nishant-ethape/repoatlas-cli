"""
Ingestion pipeline — walks a repository, chunks every file, embeds each chunk,
and stores results in ChromaDB (no SQL, no server).

Usage (programmatic):
    from db.chroma import get_client, get_collection
    from indexer.ingestion import index_repo

    client     = get_client()            # uses ~/.repoatlas/chroma_db
    collection = get_collection(client)
    summary    = index_repo("/path/to/repo", collection)
    print(summary)
"""
import hashlib
import os
from pathlib import Path
from typing import Optional

import chromadb

from db.chroma import clear_repo
from indexer.chunker import chunk_file

from indexer.embedder import get_embedding

# --------------------------------------------------------------------------- #
# Skip lists  (unchanged from pgvector version)
# --------------------------------------------------------------------------- #

_SKIP_DIRS = {
    ".git", ".hg", ".svn",
    ".venv", "venv", "env", ".env",
    "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", ".eggs",
    ".idea", ".vscode",
    ".repoatlas",           # don't index our own ChromaDB data
}

_SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".so", ".dylib", ".dll", ".exe", ".obj", ".o",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".pdf", ".docx", ".xlsx", ".pptx", ".ttf", ".woff", ".woff2",
    ".lock",
}

_MAX_FILE_BYTES = 500_000  # 500 KB


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _chunk_id(repo_path: str, rel_path: str, name: str, start_line: int) -> str:
    """Stable, unique string ID for a chunk (MD5 of its key fields)."""
    raw = f"{repo_path}|{rel_path}|{name}|{start_line}"
    return hashlib.md5(raw.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def index_repo(
    repo_path: str,
    collection: chromadb.Collection,
    reset: bool = False,
    embed_model: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Walk *repo_path*, chunk and embed every eligible file, store in ChromaDB.

    Parameters
    ----------
    repo_path  : str — absolute path to the repository root to index
    collection : chromadb.Collection — target collection (already open)
    reset      : bool — True means the collection was already wiped by the
                 caller (via reset_collection); False means we clear only
                 this repo's existing chunks before re-indexing
    embed_model: str, optional — override the Ollama embedding model
    verbose    : bool — print per-file progress lines

    Returns
    -------
    dict with keys: files_indexed, chunks_stored, files_skipped, errors
    """
    repo_path = str(Path(repo_path).resolve())

    # In non-reset mode, remove only this repo's stale chunks first
    if not reset:
        deleted = clear_repo(collection, repo_path)
        if verbose and deleted:
            print(f"🗑  Removed {deleted} existing chunks for this repo.")

    files_indexed  = 0
    chunks_stored  = 0
    files_skipped  = 0
    errors: list[str] = []

    kwargs = {"model": embed_model} if embed_model else {}

    for dirpath, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]

        for filename in sorted(filenames):
            file_path = os.path.join(dirpath, filename)
            _, ext = os.path.splitext(filename)

            if ext.lower() in _SKIP_EXTENSIONS:
                files_skipped += 1
                continue
            try:
                if os.path.getsize(file_path) > _MAX_FILE_BYTES:
                    files_skipped += 1
                    continue
            except OSError:
                files_skipped += 1
                continue

            chunks = chunk_file(file_path)
            if not chunks:
                files_skipped += 1
                continue

            try:
                rel_path = str(Path(file_path).relative_to(repo_path))
            except ValueError:
                rel_path = file_path

            # ---- embed all chunks for this file, then batch-add ---- #
            ids:        list[str]        = []
            embeddings: list[list[float]] = []
            documents:  list[str]        = []
            metadatas:  list[dict]       = []
            file_had_error = False

            for chunk in chunks:
                try:
                    emb = get_embedding(chunk.code, **kwargs)
                    ids.append(_chunk_id(repo_path, rel_path, chunk.name, chunk.start_line))
                    embeddings.append(emb)
                    documents.append(chunk.code)
                    metadatas.append({
                        "repo_path":  repo_path,
                        "file_path":  rel_path,
                        "chunk_type": chunk.chunk_type,
                        "name":       chunk.name,
                        "start_line": chunk.start_line,
                        "end_line":   chunk.end_line,
                    })
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{rel_path}: {exc}")
                    file_had_error = True
                    break

            if not file_had_error and ids:
                try:
                    collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas,
                    )
                    files_indexed += 1
                    chunks_stored += len(ids)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{rel_path}: {exc}")
                    file_had_error = True

            if file_had_error:
                files_skipped += 1

            if verbose:
                status = "✓" if not file_had_error else "✗"
                n = len(ids) if not file_had_error else 0
                print(f"  {status} {rel_path}  ({n} chunks)")

    return {
        "files_indexed": files_indexed,
        "chunks_stored": chunks_stored,
        "files_skipped": files_skipped,
        "errors": errors,
    }
