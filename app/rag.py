"""
RAG generation — assembles retrieved code chunks into a prompt and calls
the active LLM provider.
"""
import re
from typing import Optional

import chromadb

from api.retriever import retrieve
from app.providers import LLMProvider
from indexer.repomap import (
    REPO_MAP_CHUNK_TYPE,
    REPO_MAP_FILE_PATH,
    drop_nested_roots,
    load_repo_maps,
    select_tree,
)

# --------------------------------------------------------------------------- #
# Prompt templates  (unchanged)
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = """\
You are RepoAtlas, an expert code-intelligence assistant.
You answer questions about a software repository strictly based on the code \
snippets provided in the context below.

Rules:
1. Ground every claim in the provided code. If the answer is not evident from \
   the snippets, say so honestly — do not hallucinate.
2. When citing code, reference the file path and function/class name.
3. Be concise but complete. Use markdown formatting where helpful.
4. If asked to explain how something works, walk through the relevant code \
   step-by-step.
5. If a "Repository structure" tree is provided, use it for questions about \
   folders, modules, packages or layout, and list EVERY relevant entry in the \
   tree — do not limit the answer to the code snippets.
"""

_USER_TEMPLATE = """\
## Retrieved code context

{context}

---

## Question

{question}
"""

_NO_RESULTS_MSG = (
    "I couldn't find any relevant code for that question in the indexed repo.\n"
    "Try running `repoatlas index <repo_path>` first, "
    "or rephrasing your question."
)


def _format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks as a readable block for the prompt."""
    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        header = (
            f"### [{i}] {c['file_path']} — {c['chunk_type']} `{c['name']}` "
            f"(lines {c['start_line']}–{c['end_line']}, "
            f"similarity {c['similarity']:.3f})"
        )
        block = f"```\n{c['code']}\n```"
        parts.append(f"{header}\n{block}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Structural questions ("list all modules", "how is the repo organised?")
# --------------------------------------------------------------------------- #

_STRONG_STRUCTURE = re.compile(
    r"\b(structure|layout|architecture|overview|tree|organi[sz]ed|how many)\b", re.I
)
_STRUCTURE_NOUN = re.compile(
    r"\b(folders?|director(?:y|ies)|modules?|packages?|components?|apps?|services?)\b", re.I
)
_LISTING_WORD = re.compile(
    r"\b(list|name|show|enumerate|all|every|each|which|what are|inside|contains?|under)\b", re.I
)

_MAP_BUDGET_CHARS = 6000   # ~1.5k tokens: fits small local models' context


def is_structural_question(question: str) -> bool:
    """True for questions that need the whole directory layout, not a few snippets."""
    if _STRONG_STRUCTURE.search(question):
        return True
    return bool(_STRUCTURE_NOUN.search(question) and _LISTING_WORD.search(question))


def _repo_structure_context(
    question: str,
    collection: chromadb.Collection,
    repo_path: Optional[str],
) -> Optional[tuple[str, dict]]:
    """Build the prompt section + a pseudo-source entry for the directory tree."""
    maps = load_repo_maps(collection, repo_path)
    if not maps:
        return None

    roots = drop_nested_roots(maps)          # same files indexed twice -> keep outer root
    budget = _MAP_BUDGET_CHARS // len(roots)
    sections = []
    for root in roots:
        tree = select_tree(maps[root], question, budget)
        if tree:
            sections.append(f"Root: {root}\n{tree}")
    if not sections:
        return None

    text = (
        "## Repository structure (directory tree - each line: folder, files, subfolders)\n\n"
        + "\n\n".join(sections)
    )
    source = {
        "repo_path":  roots[0],
        "file_path":  REPO_MAP_FILE_PATH,
        "chunk_type": REPO_MAP_CHUNK_TYPE,
        "name":       "directory_tree",
        "start_line": 0,
        "end_line":   0,
        "code":       text,
        "similarity": 1.0,
    }
    return text, source


# --------------------------------------------------------------------------- #
# Main RAG function
# --------------------------------------------------------------------------- #

def ask(
    question: str,
    collection: chromadb.Collection,
    provider: LLMProvider,
    repo_path: Optional[str] = None,
    top_k: int = 5,
    embed_model: Optional[str] = None,
    embed_provider: Optional[str] = None,
    embed_api_key: Optional[str] = None,
) -> dict:
    """
    Answer *question* using RAG over the indexed code.

    Parameters
    ----------
    question       : str — the user's natural-language question
    collection     : chromadb.Collection — the open RepoAtlas collection
    provider       : LLMProvider — active chat LLM backend
    repo_path      : str, optional — restrict retrieval to a specific repo
    top_k          : int — number of chunks to retrieve (default 5)
    embed_model    : str, optional — override embedding model
    embed_provider : str, optional — override embedding provider ("ollama", "openai", "local")
    embed_api_key  : str, optional — API key for cloud embedding provider

    Returns
    -------
    dict with keys:
        answer  : str — the LLM's response
        sources : list[dict] — retrieved chunks used as context
    """
    chunks = retrieve(
        question=question,
        collection=collection,
        repo_path=repo_path,
        top_k=top_k,
        embed_model=embed_model,
        embed_provider=embed_provider,
        api_key=embed_api_key,
    )

    structure = None
    if is_structural_question(question):
        structure = _repo_structure_context(question, collection, repo_path)

    if not chunks and not structure:
        return {"answer": _NO_RESULTS_MSG, "sources": []}

    context = _format_context(chunks) if chunks else ""
    if structure:
        context = structure[0] + ("\n\n## Relevant code\n\n" + context if context else "")
    user_message = _USER_TEMPLATE.format(context=context, question=question)
    answer       = provider.chat(system=_SYSTEM_PROMPT, user=user_message)

    sources = ([structure[1]] if structure else []) + chunks
    return {"answer": answer, "sources": sources}
