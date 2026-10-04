"""
Retrieval-quality tests: de-duplication, per-file diversity, path-aware boost
and repo-map exclusion. A fake embedder is used, so no Ollama / model needed.
"""
import pytest

from api import retriever
from db.chroma import get_client, reset_collection

Q = [1.0, 0.0, 0.0, 0.0]


@pytest.fixture()
def collection(tmp_path, monkeypatch):
    monkeypatch.setattr(retriever, "get_embedding", lambda text, **kw: Q)
    return reset_collection(get_client(str(tmp_path)))


def _add(col, cid, emb, file_path, name="chunk", code="x = 1", repo="/repo",
         chunk_type="text", start=1):
    col.add(
        ids=[cid], embeddings=[emb], documents=[code],
        metadatas=[{"repo_path": repo, "file_path": file_path, "chunk_type": chunk_type,
                    "name": name, "start_line": start, "end_line": start}],
    )


def test_same_file_indexed_from_two_roots_is_returned_once(collection):
    _add(collection, "a", Q, "SCT_Backend\\app\\modules\\billing\\__init__.py",
         repo="/root", code="# billing")
    _add(collection, "b", Q, "modules\\billing\\__init__.py",
         repo="/root/SCT_Backend/app", code="# billing")
    results = retriever.retrieve("anything", collection, top_k=5)
    assert len(results) == 1


def test_identical_content_in_different_modules_is_kept(collection):
    # Two *different* empty __init__.py files must not collapse into one.
    _add(collection, "a", Q, "modules/crm/__init__.py", code="")
    _add(collection, "b", Q, "modules/billing/__init__.py", code="")
    results = retriever.retrieve("anything", collection, top_k=5)
    assert {r["file_path"] for r in results} == {
        "modules/crm/__init__.py", "modules/billing/__init__.py"}


def test_one_file_cannot_crowd_out_the_rest(collection):
    for i in range(4):
        _add(collection, f"big{i}", Q, "big.py", name=f"part{i}", code=f"v{i}", start=i * 10 + 1)
    _add(collection, "other", [0.9, 0.43589, 0, 0], "other.py", code="o")
    results = retriever.retrieve("anything", collection, top_k=3, max_per_file=2)
    files = [r["file_path"] for r in results]
    assert files.count("big.py") == 2
    assert "other.py" in files


def test_path_match_boosts_ranking(collection):
    _add(collection, "plain", Q, "utils/helpers.py", code="a")                    # sim 1.00
    _add(collection, "billing", [0.98, 0.19899, 0, 0], "app/billing/service.py", code="b")  # ~0.98
    results = retriever.retrieve("how does billing work", collection, top_k=2)
    assert results[0]["file_path"] == "app/billing/service.py"


def test_repo_map_chunks_never_appear_in_vector_results(collection):
    _add(collection, "map", Q, "<repo map>", chunk_type="repo_map", code="tree")
    _add(collection, "real", Q, "real.py", code="code")
    results = retriever.retrieve("anything", collection, top_k=5)
    assert [r["file_path"] for r in results] == ["real.py"]


def test_similarity_is_reported_unboosted(collection):
    _add(collection, "billing", Q, "app/billing/service.py")
    result = retriever.retrieve("billing", collection, top_k=1)[0]
    assert 0.0 <= result["similarity"] <= 1.0
    assert "_score" not in result
