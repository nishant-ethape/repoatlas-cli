"""
Integration tests for the retrieval pipeline (ChromaDB version).

These tests use a real ChromaDB in a temp directory and require a running
Ollama server. They are automatically skipped if Ollama is unavailable.
"""
import os
import tempfile
import pytest


# --------------------------------------------------------------------------- #
# Availability check
# --------------------------------------------------------------------------- #

def _ollama_available() -> bool:
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not available (run `ollama serve`)",
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def tmp_chroma():
    """A temporary ChromaDB directory that is deleted after the test module."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(scope="module")
def chroma_collection(tmp_chroma):
    from db.chroma import get_client, reset_collection
    client = get_client(tmp_chroma)
    collection = reset_collection(client)
    yield collection


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

@requires_ollama
class TestRetrievalPipeline:
    """End-to-end: index a synthetic snippet, retrieve it."""

    def test_index_and_retrieve(self, chroma_collection):
        from indexer.embedder import get_embedding
        from api.retriever import retrieve

        # Manually add a known chunk directly
        code = "def add(a, b): return a + b"
        embedding = get_embedding(code)
        chroma_collection.add(
            ids=["test-add-fn"],
            embeddings=[embedding],
            documents=[code],
            metadatas=[{
                "repo_path": "/test/repo",
                "file_path": "math_utils.py",
                "chunk_type": "function",
                "name": "add",
                "start_line": 1,
                "end_line": 1,
            }],
        )

        results = retrieve(
            question="how do I add two numbers?",
            collection=chroma_collection,
            repo_path="/test/repo",
            top_k=3,
        )
        assert len(results) >= 1
        top = results[0]
        assert top["name"] == "add"
        assert top["similarity"] > 0.5

    def test_retrieve_returns_similarity_score(self, chroma_collection):
        from api.retriever import retrieve
        results = retrieve(
            question="arithmetic",
            collection=chroma_collection,
            repo_path="/test/repo",
            top_k=1,
        )
        if results:
            assert "similarity" in results[0]
            assert 0.0 <= results[0]["similarity"] <= 1.0

    def test_unknown_repo_returns_empty(self, chroma_collection):
        from api.retriever import retrieve
        results = retrieve(
            question="anything at all",
            collection=chroma_collection,
            repo_path="/nonexistent/repo/never/indexed",
            top_k=5,
        )
        assert results == []
