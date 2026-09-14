"""
RAG generation — assembles retrieved code chunks into a prompt and calls
the active LLM provider.
"""
from typing import Optional

import chromadb

from api.retriever import retrieve
from app.providers import LLMProvider

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
# Main RAG function
# --------------------------------------------------------------------------- #

def ask(
    question: str,
    collection: chromadb.Collection,
    provider: LLMProvider,
    repo_path: Optional[str] = None,
    top_k: int = 5,
    embed_model: Optional[str] = None,
) -> dict:
    """
    Answer *question* using RAG over the indexed code.

    Parameters
    ----------
    question   : str — the user's natural-language question
    collection : chromadb.Collection — the open RepoAtlas collection
    provider   : LLMProvider — active chat LLM backend
    repo_path  : str, optional — restrict retrieval to a specific repo
    top_k      : int — number of chunks to retrieve (default 5)
    embed_model: str, optional — override embedding model

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
    )

    if not chunks:
        return {"answer": _NO_RESULTS_MSG, "sources": []}

    context      = _format_context(chunks)
    user_message = _USER_TEMPLATE.format(context=context, question=question)
    answer       = provider.chat(system=_SYSTEM_PROMPT, user=user_message)

    return {"answer": answer, "sources": chunks}
