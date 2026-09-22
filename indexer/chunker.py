"""
AST-based chunking for Python files + line-window fallback for all other file types.

Strategy:
  .py files  → ast.walk, emit only top-level FunctionDef / AsyncFunctionDef /
                ClassDef nodes (direct children of the Module). This keeps each
                chunk logically whole and avoids double-counting methods that live
                inside a class.
  other files → sliding 50-line window with 10-line overlap, giving every chunk
                a name like "lines_1-50", "lines_41-90", etc.
"""
import ast
import os
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class CodeChunk:
    file_path: str
    chunk_type: str     # "function" | "class" | "text"
    name: str           # symbol name or "lines_N-M"
    start_line: int
    end_line: int
    code: str           # raw source text of this chunk


# --------------------------------------------------------------------------- #
# Python AST chunker (top-level nodes only)
# --------------------------------------------------------------------------- #

def chunk_python_file(file_path: str) -> list[CodeChunk]:
    """
    Parse a Python file and return one chunk per top-level function/class.

    Methods nested inside a class are *not* emitted separately — they are
    included as part of their parent class chunk.  This avoids two problems:
      1. Double-counting (method appears in both the class chunk and alone).
      2. Orphaned method chunks with no context about the class they belong to.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
    except OSError:
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Unparseable .py file — fall back to line-window
        return chunk_text_file(file_path)

    source_lines = source.splitlines()
    chunks: list[CodeChunk] = []

    # Only iterate direct children of the module (top-level nodes).
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            end = node.end_lineno  # type: ignore[attr-defined]
            lines = source_lines[start - 1 : end]
            chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"

            # If node is small to medium (<= 60 lines), emit as single chunk
            if len(lines) <= 60:
                code_text = "\n".join(lines)
                chunks.append(CodeChunk(
                    file_path=file_path,
                    chunk_type=chunk_type,
                    name=node.name,
                    start_line=start,
                    end_line=end,
                    code=code_text,
                ))
            else:
                # Large function/class: split into sub-windows with 10-line overlap
                step = 50
                overlap = 10
                for i in range(0, len(lines), step - overlap):
                    sub_lines = lines[i : i + step]
                    sub_start = start + i
                    sub_end = min(start + i + len(sub_lines) - 1, end)
                    sub_code = "\n".join(sub_lines)
                    part_num = (i // (step - overlap)) + 1
                    chunks.append(CodeChunk(
                        file_path=file_path,
                        chunk_type=chunk_type,
                        name=f"{node.name}_p{part_num}",
                        start_line=sub_start,
                        end_line=sub_end,
                        code=sub_code,
                    ))
                    if i + step >= len(lines):
                        break

    # If the file has no top-level symbols (e.g. a pure script with no defs),
    # treat it as a text file so we don't silently skip it.
    if not chunks:
        return chunk_text_file(file_path)

    return chunks


# --------------------------------------------------------------------------- #
# Plain-text / generic line-window chunker (all other file types)
# --------------------------------------------------------------------------- #

def chunk_text_file(
    file_path: str,
    chunk_size: int = 50,
    overlap: int = 10,
) -> list[CodeChunk]:
    """
    Split any text file into overlapping line-window chunks.

    chunk_size : number of lines per window (default 50)
    overlap    : how many lines the next window shares with the previous (default 10)
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return []

    if not lines:
        return []

    step = max(1, chunk_size - overlap)
    chunks: list[CodeChunk] = []
    total = len(lines)

    start = 0
    while start < total:
        end = min(start + chunk_size, total)
        code_text = "".join(lines[start:end])
        # 1-indexed line numbers
        chunk_name = f"lines_{start + 1}-{end}"
        chunks.append(CodeChunk(
            file_path=file_path,
            chunk_type="text",
            name=chunk_name,
            start_line=start + 1,
            end_line=end,
            code=code_text,
        ))
        if end == total:
            break
        start += step

    return chunks


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #

def chunk_file(file_path: str) -> list[CodeChunk]:
    """
    Route a file to the right chunker based on its extension.

    .py → AST chunker (top-level symbols)
    everything else → line-window text chunker
    """
    _, ext = os.path.splitext(file_path)
    if ext.lower() == ".py":
        return chunk_python_file(file_path)
    return chunk_text_file(file_path)


# --------------------------------------------------------------------------- #
# Quick self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    result = chunk_python_file(__file__)
    for c in result:
        print(f"[{c.chunk_type}] {c.name}  (lines {c.start_line}-{c.end_line})")
        print("-" * 40)
        print(c.code[:200])
        print("=" * 40)