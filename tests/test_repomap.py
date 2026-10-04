"""Tests for the repo map (directory tree) and structural-question handling."""
import os

from app.rag import is_structural_question
from indexer.repomap import (
    build_repo_map,
    drop_nested_roots,
    select_tree,
    split_into_chunks,
)


def _make_tree(root):
    for module in ("accessories", "billing", "crm", "inventory"):
        os.makedirs(root / "backend" / "app" / "modules" / module)
        (root / "backend" / "app" / "modules" / module / "__init__.py").write_text("")
    os.makedirs(root / "frontend" / "lib")
    os.makedirs(root / "node_modules" / "left_pad")
    os.makedirs(root / ".git")
    (root / "README.md").write_text("hi")


class TestBuildRepoMap:
    def test_lists_directories_and_skips_ignored(self, tmp_path):
        _make_tree(tmp_path)
        text = "\n".join(build_repo_map(str(tmp_path), {"node_modules", ".git"}))
        assert "billing/" in text and "frontend/" in text
        assert "node_modules" not in text and ".git" not in text

    def test_indentation_reflects_depth(self, tmp_path):
        _make_tree(tmp_path)
        lines = build_repo_map(str(tmp_path), {"node_modules", ".git"})
        billing = next(ln for ln in lines if ln.strip().startswith("billing/"))
        modules = next(ln for ln in lines if ln.strip().startswith("modules/"))
        assert len(billing) - len(billing.lstrip()) > len(modules) - len(modules.lstrip())

    def test_split_into_chunks_covers_every_line(self):
        lines = [f"dir{i}/  (0 files, 0 subdirs)" for i in range(95)]
        pieces = split_into_chunks(lines)
        assert len(pieces) == 3
        assert "\n".join(p[2] for p in pieces).split("\n") == lines
        assert pieces[0][0] == 1 and pieces[-1][1] == 95


class TestSelectTree:
    def test_focus_on_named_directory_returns_all_its_children(self, tmp_path):
        _make_tree(tmp_path)
        lines = build_repo_map(str(tmp_path), {"node_modules", ".git"})
        text = select_tree(lines, "list all the modules inside the modules folder", 6000)
        for module in ("accessories", "billing", "crm", "inventory"):
            assert f"{module}/" in text
        assert "frontend/" not in text      # unrelated branch is trimmed away

    def test_overview_without_focus_is_depth_trimmed_to_budget(self, tmp_path):
        _make_tree(tmp_path)
        lines = build_repo_map(str(tmp_path), {"node_modules", ".git"})
        text = select_tree(lines, "give me a project overview", 120)
        assert len(text) <= 120
        assert text  # never empty for a non-empty tree

    def test_empty_tree(self):
        assert select_tree([], "anything", 100) == ""


class TestDropNestedRoots:
    def test_inner_root_is_dropped(self):
        outer = os.path.abspath("proj")
        inner = os.path.join(outer, "backend", "app")
        other = os.path.abspath("other")
        assert set(drop_nested_roots([inner, outer, other])) == {outer, other}


class TestStructuralQuestion:
    def test_listing_questions_are_structural(self):
        assert is_structural_question("name all the modules inside the modules folder")
        assert is_structural_question("how is the project organized?")
        assert is_structural_question("how many services are there")

    def test_specific_questions_are_not(self):
        assert not is_structural_question("how does the billing module calculate tax?")
        assert not is_structural_question("where is the JWT token validated")
