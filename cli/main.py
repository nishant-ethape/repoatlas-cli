"""
RepoAtlas CLI — built with Typer.

Commands:
    repoatlas init   — first-time setup: check Ollama, pull models, write config
    repoatlas index <repo_path>  [--reset] [--chroma-path PATH] [--embed-model MODEL]
    repoatlas ask   <question>   [--repo PATH] [--top-k N]
                                 [--provider PROVIDER] [--model MODEL]
                                 [--chroma-path PATH]

Environment variables (all optional, override config file):
    REPOATLAS_CHROMA_PATH  local path for ChromaDB data (default: ~/.repoatlas/chroma_db)
    REPOATLAS_PROVIDER     ollama | openai | anthropic | groq
    REPOATLAS_MODEL        chat model name
    REPOATLAS_EMBED_MODEL  embedding model name
    OLLAMA_BASE_URL        Ollama server URL (default: http://localhost:11434)
    OPENAI_API_KEY
    ANTHROPIC_API_KEY
    GROQ_API_KEY
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich import print as rprint
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich import box

from app.config import load as load_config, save as save_config
from db.chroma import get_client, get_collection, reset_collection, get_chroma_path
from indexer.ingestion import index_repo
from app.providers import get_provider
from app.rag import ask as rag_ask

app = typer.Typer(
    name="repoatlas",
    help="Ask natural-language questions about any code repository.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
console     = Console()
err_console = Console(stderr=True)


# --------------------------------------------------------------------------- #
# init command
# --------------------------------------------------------------------------- #

@app.command()
def init(
    chroma_path: Optional[str] = typer.Option(
        None, "--chroma-path",
        help="Where to store ChromaDB data (default: ~/.repoatlas/chroma_db).",
    ),
    chat_model: Optional[str] = typer.Option(
        None, "--chat-model",
        help="Ollama chat model to use (default: qwen2.5-coder:3b).",
    ),
    embed_model: Optional[str] = typer.Option(
        None, "--embed-model",
        help="Ollama embedding model (default: nomic-embed-text).",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y",
        help="Accept all defaults non-interactively.",
    ),
) -> None:
    """
    First-time setup — check Ollama, pull required models, write config.

    Run this once after installing RepoAtlas. It will:

    \b
    1. Verify Ollama is installed and running
    2. Pull nomic-embed-text (embeddings) and qwen2.5-coder:3b (chat)
    3. Confirm your ChromaDB storage path
    4. Write ~/.repoatlas/config.yaml so future commands need no flags
    """
    console.print(Panel(
        "[bold cyan]RepoAtlas Setup[/bold cyan]\n"
        "[dim]First-time configuration wizard[/dim]",
        expand=False,
    ))

    cfg = load_config()

    # ------------------------------------------------------------------ #
    # Step 1 — Ollama check
    # ------------------------------------------------------------------ #
    console.print("\n[bold]Step 1/3[/bold] — Checking Ollama …")
    ollama_url = os.environ.get("OLLAMA_BASE_URL", cfg.get("ollama_base_url", "http://localhost:11434"))

    ollama_ok = _check_ollama(ollama_url)
    if not ollama_ok:
        err_console.print(
            f"\n[red]✗ Cannot reach Ollama at {ollama_url}[/red]\n"
            "[yellow]Please install and start Ollama first:[/yellow]\n"
            "  → https://ollama.com/download\n"
            "  → Then run: [bold]ollama serve[/bold]\n"
        )
        raise typer.Exit(1)

    console.print(f"  [green]✓[/green] Ollama is running at [cyan]{ollama_url}[/cyan]")

    # ------------------------------------------------------------------ #
    # Step 2 — Model selection and pulling
    # ------------------------------------------------------------------ #
    console.print("\n[bold]Step 2/3[/bold] — Models …")

    default_embed = embed_model or cfg.get("embed_model", "nomic-embed-text")
    default_chat  = chat_model  or cfg.get("chat_model",  "qwen2.5-coder:3b")

    if not yes:
        default_embed = Prompt.ask(
            "  Embedding model",
            default=default_embed,
            console=console,
        )
        default_chat = Prompt.ask(
            "  Chat model    ",
            default=default_chat,
            console=console,
        )

    console.print()
    for model_name in [default_embed, default_chat]:
        if _model_exists(ollama_url, model_name):
            console.print(f"  [green]✓[/green] {model_name} already available")
        else:
            console.print(f"  [yellow]↓[/yellow] Pulling [bold]{model_name}[/bold] …")
            ok = _pull_model(model_name)
            if ok:
                console.print(f"  [green]✓[/green] {model_name} pulled successfully")
            else:
                err_console.print(f"  [red]✗ Failed to pull {model_name}[/red]")
                err_console.print(f"    Run manually: [bold]ollama pull {model_name}[/bold]")

    # ------------------------------------------------------------------ #
    # Step 3 — ChromaDB path
    # ------------------------------------------------------------------ #
    console.print("\n[bold]Step 3/3[/bold] — Storage …")
    default_path = chroma_path or cfg.get("chroma_path", str(Path.home() / ".repoatlas" / "chroma_db"))

    if not yes:
        default_path = Prompt.ask(
            "  ChromaDB path ",
            default=default_path,
            console=console,
        )

    # Resolve ~ in path
    resolved_path = str(Path(default_path).expanduser().resolve())
    Path(resolved_path).mkdir(parents=True, exist_ok=True)
    console.print(f"  [green]✓[/green] ChromaDB will be stored at [cyan]{resolved_path}[/cyan]")

    # ------------------------------------------------------------------ #
    # Write config
    # ------------------------------------------------------------------ #
    new_cfg = {
        "chroma_path":     resolved_path,
        "provider":        cfg.get("provider", "ollama"),
        "chat_model":      default_chat,
        "embed_model":     default_embed,
        "ollama_base_url": ollama_url,
    }
    save_config(new_cfg)

    console.print(Panel(
        "[bold green]✓ Setup complete![/bold green]\n\n"
        "Next steps:\n"
        "  [cyan]repoatlas index /path/to/repo[/cyan]   — index a repository\n"
        "  [cyan]repoatlas ask \"your question\"[/cyan]   — ask anything about it\n\n"
        f"[dim]Config saved to ~/.repoatlas/config.yaml[/dim]",
        expand=False,
    ))


# --------------------------------------------------------------------------- #
# index command
# --------------------------------------------------------------------------- #

@app.command()
def index(
    repo_path: str = typer.Argument(..., help="Path to the repository root to index."),
    reset: bool = typer.Option(
        False, "--reset",
        help="Wipe the entire ChromaDB collection before indexing (erases ALL repos).",
    ),
    chroma_path: Optional[str] = typer.Option(
        None, "--chroma-path", envvar="REPOATLAS_CHROMA_PATH",
        help="ChromaDB storage path (default: from config or ~/.repoatlas/chroma_db).",
    ),
    embed_model: Optional[str] = typer.Option(
        None, "--embed-model", envvar="REPOATLAS_EMBED_MODEL",
        help="Ollama embedding model.",
    ),
) -> None:
    """
    Index a repository — chunk, embed, and store all code in ChromaDB.

    RepoAtlas walks every file in REPO_PATH, embeds each chunk with
    nomic-embed-text (via Ollama), and stores results locally in ChromaDB
    — no server, no connection string, just a local directory.

    Example:

        repoatlas index /home/user/myproject

        repoatlas index /home/user/myproject --reset
    """
    resolved = Path(repo_path).resolve()
    if not resolved.is_dir():
        err_console.print(f"[red]Error:[/red] {repo_path!r} is not a directory.")
        raise typer.Exit(1)

    # Merge CLI flags over config file
    cfg = load_config()
    effective_chroma = chroma_path or cfg.get("chroma_path", get_chroma_path())
    effective_embed  = embed_model  or cfg.get("embed_model")

    console.print(Panel(
        f"[bold cyan]RepoAtlas Indexer[/bold cyan]\n"
        f"Repo    : [green]{resolved}[/green]\n"
        f"ChromaDB: [dim]{effective_chroma}[/dim]\n"
        f"Reset   : {'[yellow]yes — all existing data will be erased[/yellow]' if reset else 'no'}",
        expand=False,
    ))

    client = get_client(effective_chroma)
    if reset:
        collection = reset_collection(client)
    else:
        collection = get_collection(client)

    with console.status("[bold green]Indexing…[/bold green]", spinner="dots"):
        try:
            summary = index_repo(
                repo_path=str(resolved),
                collection=collection,
                reset=reset,
                embed_model=effective_embed,
                verbose=False,
            )
        except RuntimeError as e:
            err_console.print(f"[red]Indexing failed:[/red] {e}")
            raise typer.Exit(1)

    table = Table(box=box.ROUNDED, show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan")
    table.add_row("Files indexed", str(summary["files_indexed"]))
    table.add_row("Chunks stored", str(summary["chunks_stored"]))
    table.add_row("Files skipped", str(summary["files_skipped"]))
    if summary["errors"]:
        table.add_row("[red]Errors[/red]", str(len(summary["errors"])))

    console.print("\n[bold green]✓ Indexing complete[/bold green]")
    console.print(table)

    if summary["errors"]:
        console.print("\n[yellow]Errors encountered:[/yellow]")
        for err in summary["errors"][:10]:
            console.print(f"  • {err}")
        if len(summary["errors"]) > 10:
            console.print(f"  … and {len(summary['errors']) - 10} more")


# --------------------------------------------------------------------------- #
# ask command
# --------------------------------------------------------------------------- #

@app.command()
def ask(
    question: str = typer.Argument(..., help="Natural-language question about the codebase."),
    repo: Optional[str] = typer.Option(
        None, "--repo",
        help="Restrict retrieval to this repo path (when multiple repos are indexed).",
    ),
    top_k: int = typer.Option(5, "--top-k", min=1, max=20, help="Chunks to retrieve (default: 5)."),
    provider: Optional[str] = typer.Option(
        None, "--provider", envvar="REPOATLAS_PROVIDER",
        help="LLM provider: ollama | openai | anthropic | groq",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", envvar="REPOATLAS_MODEL",
        help="Chat model name override.",
    ),
    chroma_path: Optional[str] = typer.Option(
        None, "--chroma-path", envvar="REPOATLAS_CHROMA_PATH",
        help="ChromaDB storage path.",
    ),
    show_sources: bool = typer.Option(
        True, "--sources/--no-sources",
        help="Show retrieved source chunks after the answer.",
    ),
) -> None:
    """
    Ask a natural-language question about an indexed codebase.

    RepoAtlas retrieves the most relevant code snippets and asks the active
    LLM to answer your question — grounded in the actual code.

    Example:

        repoatlas ask "how does authentication work?"

        repoatlas ask "where is the DB connection?" --provider openai
    """
    # Merge CLI flags over config file
    cfg = load_config()
    effective_chroma   = chroma_path or cfg.get("chroma_path", get_chroma_path())
    effective_provider = provider    or cfg.get("provider", "ollama")
    effective_model    = model       or cfg.get("chat_model")

    provider_cfg: dict = {"provider": effective_provider}
    if effective_model:
        provider_cfg["model"] = effective_model

    try:
        llm = get_provider(provider_cfg)
    except (ValueError, ImportError) as e:
        err_console.print(f"[red]Provider error:[/red] {e}")
        raise typer.Exit(1)

    client     = get_client(effective_chroma)
    collection = get_collection(client)

    repo_resolved = str(Path(repo).resolve()) if repo else None

    console.print(Panel(
        f"[bold cyan]RepoAtlas[/bold cyan] · [dim]{llm!r}[/dim]",
        expand=False,
    ))
    console.print(f"[bold]Question:[/bold] {question}\n")

    with console.status("[bold green]Thinking…[/bold green]", spinner="dots"):
        try:
            result = rag_ask(
                question=question,
                collection=collection,
                provider=llm,
                repo_path=repo_resolved,
                top_k=top_k,
            )
        except RuntimeError as e:
            err_console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    console.rule("[bold green]Answer[/bold green]")
    console.print(Markdown(result["answer"]))

    if show_sources and result["sources"]:
        console.rule("[dim]Sources[/dim]")
        for i, src in enumerate(result["sources"], 1):
            console.print(
                f"  [dim][{i}][/dim] "
                f"[cyan]{src['file_path']}[/cyan]"
                f"  [dim]{src['chunk_type']} [bold]{src['name']}[/bold] "
                f"L{src['start_line']}–{src['end_line']} "
                f"(sim={src['similarity']:.3f})[/dim]"
            )


# --------------------------------------------------------------------------- #
# Helpers for init
# --------------------------------------------------------------------------- #

def _check_ollama(base_url: str) -> bool:
    """Return True if Ollama is reachable."""
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False


def _model_exists(base_url: str, model_name: str) -> bool:
    """Return True if *model_name* is already pulled in Ollama."""
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        if r.status_code != 200:
            return False
        models = [m.get("name", "") for m in r.json().get("models", [])]
        # Ollama may return "qwen2.5-coder:3b" or "qwen2.5-coder:3b-instruct"
        return any(m.startswith(model_name.split(":")[0]) for m in models)
    except Exception:
        return False


def _pull_model(model_name: str) -> bool:
    """Run `ollama pull <model>` and stream output. Returns True on success."""
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    app()


if __name__ == "__main__":
    main()
