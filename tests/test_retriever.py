"""
Integration tests for the retrieval and embedding pipeline (ChromaDB version).
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
        r = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
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
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
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

class TestLocalRetrievalPipeline:
    """Test retrieval pipeline using in-process local embeddings (no Ollama needed)."""

    def test_in_process_embed_and_retrieve(self, chroma_collection):
        from indexer.embedder import get_embedding
        from api.retriever import retrieve

        code = "def multiply(x, y): return x * y"
        embedding = get_embedding(code, provider="local")
        assert len(embedding) > 0

        chroma_collection.add(
            ids=["test-mult-fn"],
            embeddings=[embedding],
            documents=[code],
            metadatas=[{
                "repo_path": "/test/local_repo",
                "file_path": "calc.py",
                "chunk_type": "function",
                "name": "multiply",
                "start_line": 1,
                "end_line": 1,
            }],
        )

        results = retrieve(
            question="how to multiply two numbers?",
            collection=chroma_collection,
            repo_path="/test/local_repo",
            embed_provider="local",
            top_k=3,
        )
        assert len(results) >= 1
        top = results[0]
        assert top["name"] == "multiply"
        assert "similarity" in top
        assert 0.0 <= top["similarity"] <= 1.0

    def test_unknown_repo_returns_empty(self, chroma_collection):
        from api.retriever import retrieve
        results = retrieve(
            question="anything at all",
            collection=chroma_collection,
            repo_path="/nonexistent/repo/never/indexed",
            embed_provider="local",
            top_k=5,
        )
        assert results == []


@requires_ollama
class TestOllamaRetrievalPipeline:
    """End-to-end: index a synthetic snippet, retrieve it using Ollama."""

    def test_index_and_retrieve_ollama(self, tmp_chroma):
        from db.chroma import get_client, reset_collection
        from indexer.embedder import get_embedding
        from api.retriever import retrieve

        client = get_client(tmp_chroma + "_ollama")
        collection = reset_collection(client)

        code = "def add(a, b): return a + b"
        embedding = get_embedding(code, provider="ollama")
        collection.add(
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
            collection=collection,
            repo_path="/test/repo",
            embed_provider="ollama",
            top_k=3,
        )
        assert len(results) >= 1
        top = results[0]
        assert top["name"] == "add"
        assert top["similarity"] > 0.5
