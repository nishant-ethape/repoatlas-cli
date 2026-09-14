"""
RepoAtlas configuration management.

Handles reading/writing ~/.repoatlas/config.yaml and merging with
environment variables (env vars always override config file).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# PyYAML is an optional dep — we fall back to a simple key=value format
try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

_CONFIG_DIR  = Path.home() / ".repoatlas"
_CONFIG_FILE = _CONFIG_DIR / "config.yaml"

_DEFAULTS: dict[str, Any] = {
    "chroma_path":    str(_CONFIG_DIR / "chroma_db"),
    "provider":       "ollama",
    "chat_model":     "qwen2.5-coder:3b",
    "embed_model":    "nomic-embed-text",
    "ollama_base_url": "http://localhost:11434",
}

# Env-var → config key mapping
_ENV_MAP = {
    "REPOATLAS_CHROMA_PATH":  "chroma_path",
    "REPOATLAS_PROVIDER":     "provider",
    "REPOATLAS_MODEL":        "chat_model",
    "REPOATLAS_EMBED_MODEL":  "embed_model",
    "OLLAMA_BASE_URL":        "ollama_base_url",
}


def load() -> dict[str, Any]:
    """
    Return merged config: defaults → config file → env vars.

    Priority (highest wins): env vars > config.yaml > built-in defaults.
    """
    cfg = dict(_DEFAULTS)

    # Layer 1: config file
    if _CONFIG_FILE.exists():
        try:
            if _HAS_YAML:
                with _CONFIG_FILE.open() as fh:
                    from_file = yaml.safe_load(fh) or {}
            else:
                from_file = _parse_simple(_CONFIG_FILE.read_text())
            cfg.update({k: v for k, v in from_file.items() if v is not None})
        except Exception:
            pass  # silently ignore malformed config

    # Layer 2: environment variables (always override)
    for env_key, cfg_key in _ENV_MAP.items():
        val = os.environ.get(env_key)
        if val:
            cfg[cfg_key] = val

    return cfg


def save(cfg: dict[str, Any]) -> None:
    """Write *cfg* to ~/.repoatlas/config.yaml (creates dir if needed)."""
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if _HAS_YAML:
        import yaml
        with _CONFIG_FILE.open("w") as fh:
            yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)
    else:
        # Fallback: plain key=value
        lines = [f"{k}: {v}\n" for k, v in cfg.items()]
        _CONFIG_FILE.write_text("".join(lines))


def _parse_simple(text: str) -> dict[str, str]:
    """Minimal YAML-like key: value parser (used when PyYAML not installed)."""
    result: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            k, _, v = line.partition(":")
            result[k.strip()] = v.strip()
    return result
