# RepoAtlas

> Ask natural-language questions about any code repository — powered by local LLMs (Ollama) or cloud APIs (OpenAI, Groq, Anthropic) and ChromaDB. Zero server setup.

```bash
pip install repoatlas-cli
repoatlas init
repoatlas index /path/to/myproject
repoatlas ask "how does authentication work here?"
repoatlas ask "name all the modules inside the modules folder"
```

---

## What it is

RepoAtlas turns a code repository into a searchable knowledge base. Index it once, then ask questions in plain English and get answers grounded in the actual code.

It combines **AST/line-window chunking**, **vector embeddings**, **hybrid (path-aware) retrieval**, a **repository structure map** and a **language model** in a RAG pipeline.

**No Postgres. No server.** ChromaDB stores everything in a local directory (`~/.repoatlas/chroma_db`).

### What's new in 0.4.0

- **Repository structure map** — a directory tree is stored at index time and injected for structural questions ("list all modules", "how is the repo organised?"), so answers list *every* folder, not just the few that matched.
- **De-duplication** — the same file indexed from different roots no longer wastes retrieval slots.
- **Per-file diversity cap** — one big file can't crowd out the rest.
- **Path-aware keyword boost** — questions naming a folder/file favour matching paths.
- **Live indexing progress** (files, chunks, current path) and smarter skip lists (`.dart_tool`, `Pods`, lockfiles, binaries, `*.min.js`, ...).
- **Embedding safety** — no silent fallback to incompatible vectors; clear error if the embedder changed (see [Switching embedders](#switching-embedders)).

---

## Local or Cloud — you choose

| Mode | Chat LLM | Embeddings | Network |
|---|---|---|---|
| **Local (Ollama)** | e.g. `qwen2.5-coder:3b` | `nomic-embed-text` (768-dim) | None after models are pulled |
| **Groq / Anthropic** | Cloud API | FastEmbed `BAAI/bge-small-en-v1.5` (384-dim, in-process) | Chat only |
| **OpenAI** | `gpt-4o-mini` | `text-embedding-3-small` (1536-dim) | Chat + embeddings |

### Works offline?
Yes, with Ollama: after `repoatlas init` has pulled the models, indexing and asking need no internet. FastEmbed downloads its small model once on first use, then also works offline.

---

## Requirements

- Python 3.11+
- *(Optional)* [Ollama](https://ollama.com/) running locally for offline mode.

## Installation

```bash
python -m pip install repoatlas-cli          # Windows-safe form
python -m pip install --upgrade repoatlas-cli
```

If `repoatlas` isn't on your PATH, use `python -m cli.main ...` from a source checkout, or install into the Python whose `Scripts` folder is on PATH.

---

## Quickstart

### 1. Setup (first time)

```bash
repoatlas init
```

Interactive choices: **1 Ollama**, **2 OpenAI**, **3 Groq**, **4 Anthropic**.
Non-interactive: `repoatlas init --provider openai --api-key sk-... -y`

### 2. Index

```bash
repoatlas index /path/to/myproject
```

Live progress shows files and chunks processed. The path must be a **directory** (no trailing `>` etc.).

### 3. Ask

```bash
repoatlas ask "where is the database connection handled?"
repoatlas ask "list all folders inside the modules directory" --top-k 10
```

---

## Usage

### `repoatlas init`

```bash
repoatlas init [--provider PROVIDER] [--api-key KEY] [--chroma-path PATH] [-y]
```

### `repoatlas index`

```bash
repoatlas index <repo_path> [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--reset` | off | Wipe the entire collection before indexing |
| `--chroma-path` | `~/.repoatlas/chroma_db` | ChromaDB storage directory |
| `--embed-model` | provider default | Embedding model override |
| `--embed-provider` | provider default | `ollama`, `openai`, `local` |

### `repoatlas ask`

```bash
repoatlas ask <question> [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--repo` | all repos | Restrict retrieval to a repo path |
| `--top-k` | 5 | Chunks to retrieve (1–20). Use 8–15 for broad questions or small repos of many files |
| `--provider` | configured | `ollama` \| `openai` \| `anthropic` \| `groq` |
| `--model` | provider default | Chat model override |
| `--chroma-path` | `~/.repoatlas/chroma_db` | ChromaDB storage directory |
| `--no-sources` | off | Hide source citations |

For structural questions the sources list begins with `<repo map>` — the directory tree that was used.

---

## Configuration

Config is stored by `repoatlas init` (YAML in `~/.repoatlas/`). Environment variables override it:

| Variable | Purpose |
|---|---|
| `REPOATLAS_EMBED_PROVIDER` | Embedding provider (`ollama`, `openai`, `local`) |
| `REPOATLAS_API_KEY` | Generic API key |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY` | Provider keys |

---

## How retrieval works

1. **Chunking** — Python is split by function/class (large ones sub-chunked at 60+ lines); other files use line windows.
2. **Embedding** — chunks are embedded with the configured embedder (long text truncated for Ollama's context).
3. **Search** — an over-fetched candidate pool is re-ranked with a small path/name keyword boost, de-duplicated, and capped at 2 chunks per file.
4. **Structure map** — a per-directory tree (`name/ (N files, M subdirs)`) is stored with the index; structural questions get the relevant part of it in the prompt (budgeted to ~6000 chars for small local models).

### Skipped automatically
Directories like `.git`, `node_modules`, `.venv`, `.dart_tool`, `Pods`, `.gradle`, `build`, `dist`, `target`, `.next`, `coverage`; binaries/archives/databases/models; lockfiles; `*.min.js`.

---

## Tips for large repositories

- Index only your **source folders** (e.g. `repoatlas index ./lib`, `./app`) rather than the whole monorepo — much faster.
- Don't index the same code from two roots; if you did, re-run with `--reset`. (Duplicates are also filtered at query time.)
- Use `--repo <path>` to scope questions when several repos are indexed.
- Small local models (3B) give shorter, less complete answers; try a larger model or a cloud provider for complex questions.

## Switching embedders

Vector dimensions differ per embedder (384 FastEmbed, 768 Ollama, 1536 OpenAI). After changing embedding provider/model you **must** re-index:

```bash
repoatlas index <path> --reset
```

Upgrading from < 0.4.0 also requires re-indexing to create the structure map.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `repoatlas` not recognized | Install with `python -m pip install repoatlas-cli`; ensure Python's `Scripts` dir is on PATH |
| Ollama connection error | Start the Ollama app; run `repoatlas init` to pull models |
| `dimension` mismatch error | Re-run `repoatlas index <path> --reset` |
| Answer lists only some modules | Re-index with 0.4.0+, and increase `--top-k` |
| Indexing slow | Index source subfolders; Ollama embeds one chunk at a time |

---

## Project structure

```
api/       retriever (hybrid search, dedupe, diversity)
app/       rag pipeline, LLM providers, config
cli/       Typer CLI (init, index, ask)
db/        ChromaDB client
indexer/   ingestion, chunker, embedder, repomap
tests/     pytest suite
```

## Running tests

```bash
python -m pytest
```
(The Ollama test requires Ollama running with `nomic-embed-text`.)

---

## License

MIT — see [LICENSE](LICENSE)
