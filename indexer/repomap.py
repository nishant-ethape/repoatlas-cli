"""
Repository map — a compact directory tree stored next to the code chunks.

Why it exists
-------------
Vector search finds the few chunks *closest* to a question. That is perfect for
"how does X work?", but it cannot answer structural questions such as
"list all modules" or "how is the project organised?", because no single chunk
(or top-k set of chunks) contains the whole picture.

At index time we therefore record the directory tree (one line per directory)
as special ``repo_map`` chunks. At question time, ``select_tree`` trims that
tree to the part that is relevant to the question and fits a prompt budget.

Line format (two spaces of indentation per level)::

    myproject/  (3 files, 4 subdirs)
      app/  (2 files, 1 subdirs)
        modules/  (1 files, 19 subdirs)
          billing/  (4 files, 2 subdirs)
"""
from __future__ import annotations

import os
import re
from typing import Iterable, Optional

import chromadb

REPO_MAP_CHUNK_TYPE = "repo_map"
REPO_MAP_FILE_PATH = "<repo map>"

_LINES_PER_CHUNK = 40     # keeps every stored chunk well inside embedding limits
_MAX_DIRS = 5000          # safety cap for gigantic monorepos
_INDENT = "  "


# --------------------------------------------------------------------------- #
# Build (index time)
# --------------------------------------------------------------------------- #

def build_repo_map(
    repo_path: str,
    skip_dirs: Iterable[str] = (),
    skip_extensions: Iterable[str] = (),
) -> list[str]:
    """Return one tree line per directory under *repo_path* (pre-order, sorted)."""
    skip_dirs = set(skip_dirs)
    skip_extensions = set(skip_extensions)
    root = os.path.abspath(repo_path)
    root_name = os.path.basename(root) or root

    lines: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in skip_dirs)
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        name = root_name if depth == 0 else os.path.basename(dirpath)
        n_files = sum(
            1 for f in filenames
            if os.path.splitext(f)[1].lower() not in skip_extensions
        )
        lines.append(f"{_INDENT * depth}{name}/  ({n_files} files, {len(dirnames)} subdirs)")
        if len(lines) >= _MAX_DIRS:
            break
    return lines


def split_into_chunks(lines: list[str]) -> list[tuple[int, int, str]]:
    """Group tree lines into (start_line, end_line, text) pieces (1-indexed)."""
    pieces = []
    for start in range(0, len(lines), _LINES_PER_CHUNK):
        part = lines[start:start + _LINES_PER_CHUNK]
        pieces.append((start + 1, start + len(part), "\n".join(part)))
    return pieces


# --------------------------------------------------------------------------- #
# Load + select (query time)
# --------------------------------------------------------------------------- #

def load_repo_maps(
    collection: chromadb.Collection,
    repo_path: Optional[str] = None,
) -> dict[str, list[str]]:
    """Return ``{repo_root: tree_lines}`` for every stored repo map."""
    where: dict = {"chunk_type": REPO_MAP_CHUNK_TYPE}
    if repo_path:
        where = {"$and": [where, {"repo_path": repo_path}]}
    try:
        res = collection.get(where=where, include=["documents", "metadatas"])
    except Exception:  # noqa: BLE001 — a missing map must never break `ask`
        return {}

    pieces: dict[str, list[tuple[int, str]]] = {}
    for doc, meta in zip(res.get("documents") or [], res.get("metadatas") or []):
        pieces.setdefault(meta.get("repo_path", ""), []).append(
            (int(meta.get("start_line", 0)), doc)
        )

    maps: dict[str, list[str]] = {}
    for root, parts in pieces.items():
        parts.sort(key=lambda p: p[0])
        maps[root] = "\n".join(text for _, text in parts).split("\n")
    return maps


def drop_nested_roots(roots: Iterable[str]) -> list[str]:
    """Drop roots that live inside another root (same files indexed twice)."""
    roots = sorted(set(roots), key=len)
    kept: list[str] = []
    for r in roots:
        norm = os.path.normcase(os.path.abspath(r))
        if any(norm.startswith(os.path.normcase(os.path.abspath(k)) + os.sep) for k in kept):
            continue
        kept.append(r)
    return kept


def _depth(line: str) -> int:
    return (len(line) - len(line.lstrip(" "))) // len(_INDENT)


def _name(line: str) -> str:
    return line.strip().split("/  (", 1)[0].lower()


def select_tree(lines: list[str], question: str, max_chars: int) -> str:
    """
    Trim the tree to what matters for *question* and to *max_chars*.

    * If the question mentions a directory name (e.g. "modules"), keep that
      directory, its ancestors, and its descendants (as deep as the budget allows).
    * Otherwise return a whole-tree overview, as deep as the budget allows.
    """
    if not lines:
        return ""

    words = set(re.findall(r"[a-z0-9_]{3,}", question.lower()))
    words |= {w[:-1] for w in words if w.endswith("s") and len(w) > 3}
    focus = [i for i, ln in enumerate(lines) if _name(ln) in words]

    if focus:
        text = ""
        for extra_depth in (3, 2, 1, 0):
            keep: set[int] = set()
            for i in focus:
                d = _depth(lines[i])
                need, j = d - 1, i - 1            # ancestors, nearest first
                while need >= 0 and j >= 0:
                    if _depth(lines[j]) == need:
                        keep.add(j)
                        need -= 1
                    j -= 1
                keep.add(i)
                j = i + 1                         # descendants
                while j < len(lines) and _depth(lines[j]) > d:
                    if _depth(lines[j]) <= d + extra_depth:
                        keep.add(j)
                    j += 1
            text = "\n".join(lines[k] for k in sorted(keep))
            if len(text) <= max_chars:
                return text
        return text[:max_chars]

    max_depth = max(_depth(ln) for ln in lines)
    text = ""
    for limit in range(max_depth, -1, -1):
        text = "\n".join(ln for ln in lines if _depth(ln) <= limit)
        if len(text) <= max_chars:
            return text
    return text[:max_chars]
