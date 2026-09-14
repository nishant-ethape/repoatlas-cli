"""
Unit tests for indexer/chunker.py.

No database or network required — pure Python.
"""
import ast
import os
import textwrap
import tempfile
from pathlib import Path

import pytest

from indexer.chunker import (
    chunk_python_file,
    chunk_text_file,
    chunk_file,
    CodeChunk,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _write_temp(content: str, suffix: str = ".py") -> str:
    """Write content to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    )
    f.write(textwrap.dedent(content))
    f.close()
    return f.name


# --------------------------------------------------------------------------- #
# AST chunker — Python files
# --------------------------------------------------------------------------- #

class TestChunkPythonFile:
    def test_extracts_top_level_function(self):
        path = _write_temp("""\
            def greet(name):
                return f"Hello, {name}"
        """)
        try:
            chunks = chunk_python_file(path)
            assert len(chunks) == 1
            assert chunks[0].name == "greet"
            assert chunks[0].chunk_type == "function"
            assert "greet" in chunks[0].code
        finally:
            os.unlink(path)

    def test_extracts_top_level_class(self):
        path = _write_temp("""\
            class Foo:
                def bar(self):
                    pass

                def baz(self):
                    pass
        """)
        try:
            chunks = chunk_python_file(path)
            # Only the class — methods are NOT emitted separately
            names = [c.name for c in chunks]
            assert "Foo" in names
            assert "bar" not in names
            assert "baz" not in names
        finally:
            os.unlink(path)

    def test_no_double_counting_methods(self):
        path = _write_temp("""\
            class MyClass:
                def method_a(self):
                    pass

                def method_b(self):
                    pass

            def standalone():
                pass
        """)
        try:
            chunks = chunk_python_file(path)
            names = [c.name for c in chunks]
            assert names == ["MyClass", "standalone"]
        finally:
            os.unlink(path)

    def test_async_function(self):
        path = _write_temp("""\
            async def fetch(url):
                pass
        """)
        try:
            chunks = chunk_python_file(path)
            assert chunks[0].name == "fetch"
            assert chunks[0].chunk_type == "function"
        finally:
            os.unlink(path)

    def test_empty_file_returns_empty(self):
        path = _write_temp("")
        try:
            chunks = chunk_python_file(path)
            assert chunks == []
        finally:
            os.unlink(path)

    def test_script_with_no_defs_falls_back_to_line_window(self):
        path = _write_temp("""\
            x = 1
            y = 2
            print(x + y)
        """)
        try:
            chunks = chunk_python_file(path)
            # No top-level defs → falls back to text chunker
            assert len(chunks) >= 1
            assert all(c.chunk_type == "text" for c in chunks)
        finally:
            os.unlink(path)

    def test_syntax_error_falls_back_to_text_chunker(self):
        path = _write_temp("def broken(\n  # unclosed\n")
        try:
            chunks = chunk_python_file(path)
            assert all(c.chunk_type == "text" for c in chunks)
        finally:
            os.unlink(path)

    def test_line_numbers_are_correct(self):
        path = _write_temp("""\
            def first():
                pass

            def second():
                pass
        """)
        try:
            chunks = chunk_python_file(path)
            assert chunks[0].start_line == 1
            assert chunks[1].start_line > chunks[0].end_line
        finally:
            os.unlink(path)


# --------------------------------------------------------------------------- #
# Text / line-window chunker
# --------------------------------------------------------------------------- #

class TestChunkTextFile:
    def test_single_chunk_for_short_file(self):
        lines = "\n".join(f"line {i}" for i in range(10))
        path = _write_temp(lines, suffix=".txt")
        try:
            chunks = chunk_text_file(path, chunk_size=50)
            assert len(chunks) == 1
            assert chunks[0].chunk_type == "text"
        finally:
            os.unlink(path)

    def test_multiple_chunks_with_overlap(self):
        # 100 lines, chunk_size=50, overlap=10 → step=40
        # windows: 1-50, 41-90, 81-100 → 3 chunks
        lines = "\n".join(f"line {i}" for i in range(100))
        path = _write_temp(lines, suffix=".js")
        try:
            chunks = chunk_text_file(path, chunk_size=50, overlap=10)
            assert len(chunks) == 3
            assert chunks[0].start_line == 1
            assert chunks[0].end_line == 50
            assert chunks[1].start_line == 41  # overlap
        finally:
            os.unlink(path)

    def test_empty_file_returns_empty(self):
        path = _write_temp("", suffix=".txt")
        try:
            chunks = chunk_text_file(path)
            assert chunks == []
        finally:
            os.unlink(path)

    def test_chunk_names_reflect_line_range(self):
        lines = "\n".join(f"x" for _ in range(60))
        path = _write_temp(lines, suffix=".md")
        try:
            chunks = chunk_text_file(path, chunk_size=50, overlap=0)
            assert chunks[0].name == "lines_1-50"
            assert chunks[1].name == "lines_51-60"
        finally:
            os.unlink(path)


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #

class TestChunkFile:
    def test_py_extension_uses_ast_chunker(self):
        path = _write_temp("def foo(): pass", suffix=".py")
        try:
            chunks = chunk_file(path)
            assert any(c.chunk_type in ("function", "class") for c in chunks)
        finally:
            os.unlink(path)

    def test_non_py_extension_uses_text_chunker(self):
        path = _write_temp("const x = 1;\nconsole.log(x);\n", suffix=".js")
        try:
            chunks = chunk_file(path)
            assert all(c.chunk_type == "text" for c in chunks)
        finally:
            os.unlink(path)

    def test_go_file_uses_text_chunker(self):
        path = _write_temp("package main\nfunc main() {}\n", suffix=".go")
        try:
            chunks = chunk_file(path)
            assert all(c.chunk_type == "text" for c in chunks)
        finally:
            os.unlink(path)
