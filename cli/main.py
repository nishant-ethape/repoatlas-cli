"""
RepoAtlas CLI — built with Typer.

Commands:
    repoatlas init   — first-time setup: choose provider (Ollama, OpenAI, Groq, Anthropic), write config
    repoatlas index <repo_path>  [--reset] [--chroma-path PATH] [--embed-model MODEL]
    repoatlas ask   <question>   [--repo PATH] [--top-k N]
                                 [--provider PROVIDER] [--model MODEL]
                                 [--chroma-path PATH]

Environment variables (all optional, override config file):
    REPOATLAS_CHROMA_PATH      local path for ChromaDB data (default: ~/.repoatlas/chroma_db)
    REPOATLAS_PROVIDER         ollama | openai | anthropic | groq
    REPOATLAS_MODEL            chat model name
    REPOATLAS_EMBED_PROVIDER   ollama | openai | local
    REPOATLAS_EMBED_MODEL      embedding model name
    OLLAMA_BASE_URL            Ollama server URL (default: http://localhost:11434)
    OPENAI_API_KEY
    ANTHROPIC_API_KEY
    GROQ_API_KEY
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from app.config import load as load_config, save as save_config
from app.providers import get_provider
from app.rag import ask as rag_ask
from db.chroma import get_chroma_path, get_client, get_collection, reset_collection
from indexer.ingestion import index_repo

app = typer.Typer(
    name="repoatlas",
    help="Ask natural-language questions about any code repository.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
console = Console(safe_box=True)
err_console = Console(stderr=True, safe_box=True)


# --------------------------------------------------------------------------- #
# init command
# --------------------------------------------------------------------------- #

@app.command()
def init(
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p",
        help="Provider: ollama | openai | groq | anthropic",
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k",
        help="API key for chosen cloud provider (OpenAI, Groq, Anthropic).",
    ),
    chroma_path: Optional[str] = typer.Option(
        None, "--chroma-path",
        help="Where to store ChromaDB data (default: ~/.repoatlas/chroma_db).",
    ),
    chat_model: Optional[str] = typer.Option(
        None, "--chat-model",
        help="Chat model override.",
    ),
    embed_model: Optional[str] = typer.Option(
        None, "--embed-model",
        help="Embedding model override.",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y",
        help="Accept all defaults non-interactively.",
    ),
) -> None:
    """
    First-time setup - choose your AI provider, pull models (if Ollama), and write config.
    """
    console.print(Panel(
        "[bold cyan]RepoAtlas Setup[/bold cyan]\n"
        "[dim]First-time configuration wizard (Local or Cloud API)[/dim]",
        expand=False,
    ))

    cfg = load_config()

    # ------------------------------------------------------------------ #
    # Step 1 — Select Provider
    # ------------------------------------------------------------------ #
    console.print("\n[bold]Step 1/3[/bold] - AI Provider ...")

    chosen_provider = (provider or cfg.get("provider", "ollama")).lower()

    if not yes and not provider:
        console.print("  [cyan]1[/cyan] [bold]Ollama[/bold]    - Local, free & 100% offline (requires Ollama running)")
        console.print("  [cyan]2[/cyan] [bold]OpenAI[/bold]    - Cloud (gpt-4o-mini + text-embedding-3-small, zero setup)")
        console.print("  [cyan]3[/cyan] [bold]Groq[/bold]      - Cloud (ultra-fast Llama 3.3 + local embeddings)")
        console.print("  [cyan]4[/cyan] [bold]Anthropic[/bold] - Cloud (Claude 3.5 Sonnet + local embeddings)")
        choice = Prompt.ask(
            "  Select provider",
            choices=["1", "2", "3", "4", "ollama", "openai", "groq", "anthropic"],
            default="1" if chosen_provider == "ollama" else chosen_provider,
            console=console,
        )
        mapping = {
            "1": "ollama",
            "2": "openai",
            "3": "groq",
            "4": "anthropic",
        }
        chosen_provider = mapping.get(choice, choice)

    console.print(f"  [green][OK][/green] Provider set to: [bold cyan]{chosen_provider}[/bold cyan]")

    # ------------------------------------------------------------------ #
    # Step 2 — Provider Specific Setup (API Key or Ollama Models)
    # ------------------------------------------------------------------ #
    console.print("\n[bold]Step 2/3[/bold] - Models & Credentials ...")

    active_api_key = api_key
    embed_provider = "ollama"
    default_chat = chat_model
    default_embed = embed_model
    ollama_url = cfg.get("ollama_base_url", "http://localhost:11434")

    if chosen_provider == "ollama":
        embed_provider = "ollama"
        ollama_url = os.environ.get("OLLAMA_BASE_URL", ollama_url)
        console.print(f"  Checking Ollama at {ollama_url} ...")
        if not _check_ollama(ollama_url):
            err_console.print(
                f"\n[red][X] Cannot reach Ollama at {ollama_url}[/red]\n"
                "[yellow]Please install and start Ollama:[/yellow]\n"
                "  -> https://ollama.com/download\n"
                "  -> Run: [bold]ollama serve[/bold]\n"
                "[dim](Or run `repoatlas init --provider openai` / `groq` to use Cloud APIs without Ollama)[/dim]\n"
            )
            raise typer.Exit(1)
        console.print(f"  [green][OK][/green] Ollama is running at [cyan]{ollama_url}[/cyan]")

        default_embed = default_embed or cfg.get("embed_model", "nomic-embed-text")
        default_chat = default_chat or cfg.get("chat_model", "qwen2.5-coder:3b")

        if not yes:
            default_embed = Prompt.ask("  Embedding model", default=default_embed, console=console)
            default_chat = Prompt.ask("  Chat model    ", default=default_chat, console=console)

        for model_name in [default_embed, default_chat]:
            if _model_exists(ollama_url, model_name):
                console.print(f"  [green][OK][/green] {model_name} already available")
            else:
                console.print(f"  [yellow][+][/yellow] Pulling [bold]{model_name}[/bold] ...")
                ok = _pull_model(model_name, base_url=ollama_url)
                if ok:
                    console.print(f"  [green][OK][/green] {model_name} pulled successfully")
                else:
                    err_console.print(f"  [red][X] Failed to pull {model_name}[/red]")
                    err_console.print(f"    Run manually: [bold]ollama pull {model_name}[/bold]")

    elif chosen_provider == "openai":
        embed_provider = "openai"
        default_chat = default_chat or "gpt-4o-mini"
        default_embed = default_embed or "text-embedding-3-small"
        env_key = os.environ.get("OPENAI_API_KEY", "")
        if not active_api_key:
            if not yes:
                active_api_key = Prompt.ask(
                    "  OpenAI API Key (press Enter to keep existing)",
                    default=env_key or cfg.get("openai_api_key") or cfg.get("api_key") or "",
                    password=True,
                    console=console,
                )
            else:
                active_api_key = env_key or cfg.get("openai_api_key") or cfg.get("api_key") or ""
        console.print(f"  [green][OK][/green] Chat model: [cyan]{default_chat}[/cyan], Embeddings: [cyan]{default_embed}[/cyan]")

    elif chosen_provider == "groq":
        embed_provider = "local"
        default_chat = default_chat or "llama-3.3-70b-versatile"
        default_embed = default_embed or "all-MiniLM-L6-v2"
        env_key = os.environ.get("GROQ_API_KEY", "")
        if not active_api_key:
            if not yes:
                active_api_key = Prompt.ask(
                    "  Groq API Key (press Enter to keep existing)",
                    default=env_key or cfg.get("groq_api_key") or cfg.get("api_key") or "",
                    password=True,
                    console=console,
                )
            else:
                active_api_key = env_key or cfg.get("groq_api_key") or cfg.get("api_key") or ""
        console.print(f"  [green][OK][/green] Chat model: [cyan]{default_chat}[/cyan], Embeddings: [cyan]in-process local ({default_embed})[/cyan]")

    elif chosen_provider == "anthropic":
        embed_provider = "local"
        default_chat = default_chat or "claude-3-5-sonnet-latest"
        default_embed = default_embed or "all-MiniLM-L6-v2"
        env_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not active_api_key:
            if not yes:
                active_api_key = Prompt.ask(
                    "  Anthropic API Key (press Enter to keep existing)",
                    default=env_key or cfg.get("anthropic_api_key") or cfg.get("api_key") or "",
                    password=True,
                    console=console,
                )
            else:
                active_api_key = env_key or cfg.get("anthropic_api_key") or cfg.get("api_key") or ""
        console.print(f"  [green][OK][/green] Chat model: [cyan]{default_chat}[/cyan], Embeddings: [cyan]in-process local ({default_embed})[/cyan]")

    # ------------------------------------------------------------------ #
    # Step 3 — ChromaDB storage path
    # ------------------------------------------------------------------ #
    console.print("\n[bold]Step 3/3[/bold] - Storage ...")
    default_path = chroma_path or cfg.get("chroma_path", str(Path.home() / ".repoatlas" / "chroma_db"))

    if not yes:
        default_path = Prompt.ask(
            "  ChromaDB path ",
            default=default_path,
            console=console,
        )

    resolved_path = str(Path(default_path).expanduser().resolve())
    Path(resolved_path).mkdir(parents=True, exist_ok=True)
    console.print(f"  [green][OK][/green] ChromaDB will be stored at [cyan]{resolved_path}[/cyan]")

    # ------------------------------------------------------------------ #
    # Write config
    # ------------------------------------------------------------------ #
    new_cfg = {
        "chroma_path": resolved_path,
        "provider": chosen_provider,
        "chat_model": default_chat,
        "embed_provider": embed_provider,
        "embed_model": default_embed,
        "ollama_base_url": ollama_url,
    }
    if active_api_key:
        new_cfg[f"{chosen_provider}_api_key"] = active_api_key
        new_cfg["api_key"] = active_api_key

    save_config(new_cfg)

    console.print(Panel(
        "[bold green][OK] Setup complete![/bold green]\n\n"
        "Next steps:\n"
        "  [cyan]repoatlas index /path/to/repo[/cyan]   - index a repository\n"
        "  [cyan]repoatlas ask \"your question\"[/cyan]   - ask anything about it\n\n"
        "[dim]Config saved to ~/.repoatlas/config.yaml[/dim]",
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
        help="Embedding model override.",
    ),
    embed_provider: Optional[str] = typer.Option(
        None, "--embed-provider", envvar="REPOATLAS_EMBED_PROVIDER",
        help="Embedding provider override (ollama | openai | local).",
    ),
) -> None:
    """
    Index a repository - chunk, embed, and store all code in ChromaDB.
    """
    resolved = Path(repo_path).resolve()
    if not resolved.is_dir():
        err_console.print(f"[red]Error:[/red] {repo_path!r} is not a directory.")
        raise typer.Exit(1)

    cfg = load_config()
    effective_chroma = chroma_path or cfg.get("chroma_path", get_chroma_path())
    effective_embed = embed_model or cfg.get("embed_model")
    effective_embed_provider = embed_provider or cfg.get("embed_provider")
    api_key = cfg.get("openai_api_key") or cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")

    console.print(Panel(
        f"[bold cyan]RepoAtlas Indexer[/bold cyan]\n"
        f"Repo     : [green]{resolved}[/green]\n"
        f"ChromaDB : [dim]{effective_chroma}[/dim]\n"
        f"Embedder : [cyan]{effective_embed_provider or 'default'}[/cyan] ({effective_embed or 'default'})\n"
        f"Reset    : {'[yellow]yes - all existing data will be erased[/yellow]' if reset else 'no'}",
        expand=False,
    ))

    client = get_client(effective_chroma)
    if reset:
        collection = reset_collection(client)
    else:
        collection = get_collection(client)

    with console.status("[bold green]Indexing...[/bold green]", spinner="dots"):
        try:
            summary = index_repo(
                repo_path=str(resolved),
                collection=collection,
                reset=reset,
                embed_model=effective_embed,
                embed_provider=effective_embed_provider,
                api_key=api_key,
                verbose=False,
            )
        except Exception as e:
            err_console.print(f"[red]Indexing failed:[/red] {e}")
            raise typer.Exit(1)

    table = Table(box=box.ASCII, show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan")
    table.add_row("Files indexed", str(summary["files_indexed"]))
    table.add_row("Chunks stored", str(summary["chunks_stored"]))
    table.add_row("Files skipped", str(summary["files_skipped"]))
    if summary["errors"]:
        table.add_row("[red]Errors[/red]", str(len(summary["errors"])))

    console.print("\n[bold green][OK] Indexing complete[/bold green]")
    console.print(table)

    if summary["errors"]:
        console.print("\n[yellow]Errors encountered:[/yellow]")
        for err in summary["errors"][:10]:
            console.print(f"  * {err}")
        if len(summary["errors"]) > 10:
            console.print(f"  ... and {len(summary['errors']) - 10} more")


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
    """
    cfg = load_config()
    effective_chroma = chroma_path or cfg.get("chroma_path", get_chroma_path())
    effective_provider = provider or cfg.get("provider", "ollama")
    effective_model = model or cfg.get("chat_model")
    effective_embed_provider = cfg.get("embed_provider")
    effective_embed_model = cfg.get("embed_model")
    embed_api_key = cfg.get("openai_api_key") or cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")

    provider_cfg: dict = {"provider": effective_provider}
    if effective_model:
        provider_cfg["model"] = effective_model
    if cfg.get(f"{effective_provider}_api_key"):
        provider_cfg["api_key"] = cfg.get(f"{effective_provider}_api_key")
    elif cfg.get("api_key"):
        provider_cfg["api_key"] = cfg.get("api_key")

    try:
        llm = get_provider(provider_cfg)
    except (ValueError, ImportError) as e:
        err_console.print(f"[red]Provider error:[/red] {e}")
        raise typer.Exit(1)

    client = get_client(effective_chroma)
    collection = get_collection(client)

    repo_resolved = str(Path(repo).resolve()) if repo else None

    console.print(Panel(
        f"[bold cyan]RepoAtlas[/bold cyan] - [dim]{llm!r}[/dim]",
        expand=False,
    ))
    console.print(f"[bold]Question:[/bold] {question}\n")

    with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
        try:
            result = rag_ask(
                question=question,
                collection=collection,
                provider=llm,
                repo_path=repo_resolved,
                top_k=top_k,
                embed_model=effective_embed_model,
                embed_provider=effective_embed_provider,
                embed_api_key=embed_api_key,
            )
        except Exception as e:
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
                f"L{src['start_line']}-{src['end_line']} "
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
        return any(m.startswith(model_name.split(":")[0]) for m in models)
    except Exception:
        return False


def _pull_model(model_name: str, base_url: str = "http://localhost:11434") -> bool:
    """Pull model via Ollama HTTP API or subprocess."""
    try:
        with httpx.Client(base_url=base_url, timeout=600.0) as client:
            resp = client.post("/api/pull", json={"name": model_name, "stream": False})
            if resp.status_code == 200:
                return True
    except Exception:
        pass

    # Subprocess fallback
    for exe in ["ollama", os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"), r"C:\Program Files\Ollama\ollama.exe"]:
        try:
            result = subprocess.run([exe, "pull", model_name], check=False)
            if result.returncode == 0:
                return True
        except FileNotFoundError:
            continue
    return False


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    app()


if __name__ == "__main__":
    main()
