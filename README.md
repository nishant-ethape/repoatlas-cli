# RepoAtlas

> Ask natural-language questions about any code repository — powered by local LLMs and ChromaDB. Runs fully offline. Zero server setup.

```
repoatlas init
repoatlas index /path/to/myproject
repoatlas ask "how does authentication work here?"
```

---

## What it is

RepoAtlas is a CLI tool that turns any code repository into a searchable knowledge base. Point it at a repo, index it once, then ask questions in plain English and get answers grounded in the actual code — not generic guesses.

It combines **AST-based chunking**, **vector embeddings**, and a **local LLM** into a standard RAG (Retrieval-Augmented Generation) pipeline — running entirely on your own machine, with no data leaving your network.

**No Postgres. No server. No connection strings.** ChromaDB stores everything in a local directory (`~/.repoatlas/chroma_db`) that just works.

---

## Why local-first?

- **Privacy / compliance** — banking, defense, legal, and other regulated environments forbid sending proprietary code to cloud AI APIs. RepoAtlas works with zero network access by default.
- **Air-gapped environments** — cloud-based agents don't work here at all.
- **No API costs** — the default stack (Ollama + `nomic-embed-text` + `qwen2.5-coder:3b`) is entirely free.
- Cloud providers (OpenAI, Anthropic, Groq) are available as opt-in backends for users with weaker hardware.

---

## How it works

```
Repo files
    │
    ▼
[Chunker]        AST-based for .py (top-level functions & classes)
                 Line-window (50 lines, 10-line overlap) for all other files
    │
    ▼
[Embedder]       nomic-embed-text via Ollama → 768-dim vectors
    │
    ▼
[ChromaDB]       Local embedded vector store — no server, just a directory
    │
    ▼  (at query time)
[Retriever]      Embeds the question, cosine-similarity search → top-k chunks
    │
    ▼
[LLM]            Chunks + question → prompt → grounded answer
```

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com/) (for local mode)

That's it. ChromaDB is installed automatically as a Python dependency — no server setup needed.

---

## Installation

```bash
pip install repoatlas-cli
```

For cloud provider support (optional):
```bash
pip install "repoatlas-cli[openai]"
pip install "repoatlas-cli[anthropic]"
pip install "repoatlas-cli[groq]"
pip install "repoatlas-cli[all]"     # all cloud providers
```

---

## Quickstart

### 1. Run setup (first time only)

```bash
repoatlas init
```

This will:
- Check that Ollama is installed and running
- Pull the required models (`nomic-embed-text`, `qwen2.5-coder:3b`)
- Confirm the ChromaDB storage path
- Write `~/.repoatlas/config.yaml` so you never have to configure again

### 2. Index a repository

```bash
repoatlas index /path/to/myproject
```

### 3. Ask questions

```bash
repoatlas ask "how does authentication work?"
repoatlas ask "where is the database connection handled?"
repoatlas ask "how do I add a new API endpoint?"
```

---

## Usage

### `repoatlas init`

Interactive first-time setup. Checks Ollama, pulls models, writes config.

```bash
repoatlas init [--chroma-path PATH] [--chat-model MODEL] [--embed-model MODEL]
```

### `repoatlas index`

```bash
repoatlas index <repo_path> [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--reset` | off | Wipe entire ChromaDB collection before indexing |
| `--chroma-path` | `~/.repoatlas/chroma_db` | ChromaDB storage directory |
| `--embed-model` | `nomic-embed-text` | Ollama embedding model |

### `repoatlas ask`

```bash
repoatlas ask <question> [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--repo` | (all repos) | Restrict retrieval to a specific repo path |
| `--top-k` | 5 | Number of chunks to retrieve (1–20) |
| `--provider` | `ollama` | `ollama` \| `openai` \| `anthropic` \| `groq` |
| `--model` | provider default | Chat model override |
| `--chroma-path` | `~/.repoatlas/chroma_db` | ChromaDB storage directory |
| `--no-sources` | off | Hide source citations |

---

## Using cloud providers

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
repoatlas ask "explain the auth flow" --provider openai --model gpt-4o

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
repoatlas ask "explain the auth flow" --provider anthropic

# Groq (fast inference, free tier available)
export GROQ_API_KEY=gsk_...
repoatlas ask "explain the auth flow" --provider groq
```

---

## Configuration

### Environment variables

| Variable | Description |
|---|---|
| `REPOATLAS_CHROMA_PATH` | ChromaDB storage path (default: `~/.repoatlas/chroma_db`) |
| `REPOATLAS_PROVIDER` | Default LLM provider (`ollama`) |
| `REPOATLAS_MODEL` | Default chat model |
| `REPOATLAS_EMBED_MODEL` | Embedding model (default: `nomic-embed-text`) |
| `OLLAMA_BASE_URL` | Ollama server URL (default: `http://localhost:11434`) |
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GROQ_API_KEY` | Groq API key |

### Config file (`~/.repoatlas/config.yaml`)

After running `repoatlas init`, settings are persisted here. Environment variables override the config file.

```yaml
chroma_path: ~/.repoatlas/chroma_db
provider: ollama
chat_model: qwen2.5-coder:3b
embed_model: nomic-embed-text
ollama_base_url: http://localhost:11434
```

---

## Multiple repositories

RepoAtlas can hold multiple repos in the same ChromaDB — use `--repo` when querying to restrict to one:

```bash
repoatlas index /path/to/project-a
repoatlas index /path/to/project-b

repoatlas ask "how does auth work?" --repo /path/to/project-a
repoatlas ask "how does auth work?" --repo /path/to/project-b
```

---

## Project structure

```
RepoAtlas/
├── db/
│   └── chroma.py       — ChromaDB client + collection helpers
├── indexer/
│   ├── chunker.py      — AST chunker (Python) + line-window fallback (all files)
│   ├── embedder.py     — Ollama /api/embeddings wrapper
│   └── ingestion.py    — Repo walker + indexing pipeline
├── api/
│   └── retriever.py    — ChromaDB cosine-similarity retrieval
├── app/
│   ├── providers.py    — Pluggable LLM backends (Ollama / OpenAI / Anthropic / Groq)
│   └── rag.py          — RAG orchestration (retrieve → prompt → generate)
├── cli/
│   └── main.py         — Typer CLI (init, index, ask)
└── tests/
    ├── test_chunker.py  — Unit tests for chunking logic
    └── test_retriever.py — Integration tests for retrieval
```

---

## Running tests

```bash
# Unit tests (no Ollama required)
pytest tests/test_chunker.py -v

# Integration tests (requires Ollama running)
pytest tests/ -v
```

---

## Roadmap

- [ ] Tree-sitter chunking for JS / Go / Rust / Java
- [ ] `repoatlas search` — raw chunk retrieval without LLM
- [ ] `repoatlas watch` — incremental re-indexing on file change
- [ ] Web UI

---

## License

MIT — see [LICENSE](LICENSE)
