# RepoAtlas

> Ask natural-language questions about any code repository — powered by Local LLMs (Ollama) or Cloud APIs (OpenAI, Groq, Anthropic) and ChromaDB. Zero server setup.

```bash
repoatlas init
repoatlas index /path/to/myproject
repoatlas ask "how does authentication work here?"
```

---

## What it is

RepoAtlas is a CLI tool that turns any code repository into a searchable knowledge base. Point it at a repo, index it once, then ask questions in plain English and get answers grounded in the actual code — not generic guesses.

It combines **AST-based chunking**, **vector embeddings**, and a **language model** into a standard RAG (Retrieval-Augmented Generation) pipeline.

**No Postgres. No server. No connection strings.** ChromaDB stores everything in a local directory (`~/.repoatlas/chroma_db`) that just works.

---

## Local or Cloud — You Choose

- **Local / Offline Mode (Ollama)**: 100% private, zero network traffic, zero API bills.
- **Cloud Mode (OpenAI, Groq, Anthropic)**: Zero heavy model downloads, works instantly with your API key.

---

## Requirements

- Python 3.11+
- *(Optional)* [Ollama](https://ollama.com/) if you want to use local offline models.

---

## Installation

```bash
pip install repoatlas-cli
```

---

## Quickstart

### 1. Run setup (first time only)

```bash
repoatlas init
```

Choose your provider interactively:
- **1: Ollama** — Local & offline (auto-checks Ollama and pulls models)
- **2: OpenAI** — Cloud (`gpt-4o-mini` + `text-embedding-3-small`)
- **3: Groq** — Cloud (`llama-3.3-70b` + in-process local embeddings)
- **4: Anthropic** — Cloud (`claude-3-5-sonnet` + in-process local embeddings)

*(Non-interactive setup also supported: `repoatlas init --provider openai --api-key sk-... -y`)*

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

```bash
repoatlas init [--provider PROVIDER] [--api-key KEY] [--chroma-path PATH]
```

### `repoatlas index`

```bash
repoatlas index <repo_path> [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--reset` | off | Wipe entire ChromaDB collection before indexing |
| `--chroma-path` | `~/.repoatlas/chroma_db` | ChromaDB storage directory |
| `--embed-model` | provider default | Embedding model override |
| `--embed-provider` | provider default | Embedding provider (`ollama`, `openai`, `local`) |

### `repoatlas ask`

```bash
repoatlas ask <question> [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--repo` | (all repos) | Restrict retrieval to a specific repo path |
| `--top-k` | 5 | Number of chunks to retrieve (1–20) |
| `--provider` | configured default | `ollama` \| `openai` \| `anthropic` \| `groq` |
| `--model` | provider default | Chat model override |
| `--chroma-path` | `~/.repoatlas/chroma_db` | ChromaDB storage directory |
| `--no-sources` | off | Hide source citations |

---

## Running tests

```bash
python -m pytest
```

---

## License

MIT — see [LICENSE](LICENSE)
