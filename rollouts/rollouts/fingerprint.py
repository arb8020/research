"""Config fingerprinting for reproducibility.

Provides stable hashes of configuration to enable:
- Quick comparison of eval/session configs ("are these equivalent?")
- Code version tracking (git sha + dirty status)
- Dataset identity (path + content checksum)

Design: Shared core utilities, domain-specific wrappers.

Usage:
    from .fingerprint import fingerprint_eval, require_clean_git

    # For evals (errors on dirty git by default)
    fp = fingerprint_eval(config, tools=["bash", "read"], dataset_path=Path("data.jsonl"))
    print(fp)  # {"config_hash": "abc123...", "git_sha": "def456", ...}

    # For sessions (allows dirty git)
    fp = fingerprint_session(model="anthropic/claude-sonnet-4-5-20250929", env="coding", tools=["read", "write"])
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .dtypes import EvalConfig

# ── Core Utilities ──────────────────────────────────────────────────────────


def canonical_json(obj: Any) -> str:
    """Deterministic JSON encoding for hashing.

    - Sorted keys
    - No whitespace
    - ASCII only
    - Handles non-JSON types via str()
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def hash_str(s: str) -> str:
    """SHA-256 hash of string, first 16 chars."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def git_info() -> tuple[str, bool]:
    """Get current git hash (8 chars) and dirty status.

    Returns ("unknown", False) if not in a git repo or git unavailable.
    """
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            .decode()
            .strip()[:8]
        )

        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            .decode()
            .strip()
        )

        return sha, dirty
    except Exception:
        return "unknown", False


def require_clean_git(*, allow_dirty: bool = False) -> tuple[str, bool]:
    """Get git info, erroring if there are uncommitted changes.

    Args:
        allow_dirty: If True, warn instead of error on dirty state

    Returns:
        (git_sha, git_dirty) tuple

    Raises:
        RuntimeError: If git is dirty and allow_dirty=False
    """
    sha, dirty = git_info()

    if dirty and not allow_dirty:
        raise RuntimeError(
            "Uncommitted changes detected. Eval results may not be reproducible.\n"
            "Options:\n"
            "  1. Commit your changes: git commit -am 'wip'\n"
            "  2. Override with --allow-dirty flag\n"
            f"Git SHA: {sha} (dirty)"
        )

    if dirty and allow_dirty:
        print(
            "⚠ Running with uncommitted changes (git_dirty=True). Results may not be reproducible.",
            file=sys.stderr,
        )

    return sha, dirty


def callable_name(fn: Callable[..., Any], *, allow_lambda: bool = False) -> str:
    """Get qualified name for a callable.

    Returns module.qualname (e.g., 'evals.gsm8k.score_fn').

    Note: We don't hash source code. Git SHA is the authoritative code version.
    If you have uncommitted changes, the eval will error by default.

    Raises:
        ValueError: If fn is a lambda and allow_lambda=False
    """
    module = getattr(fn, "__module__", "")
    qualname = getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn)))
    name = f"{module}.{qualname}" if module else qualname

    # Reject lambdas for critical functions (score_fn, prepare_messages)
    if not allow_lambda and getattr(fn, "__name__", "") == "<lambda>":
        raise ValueError(
            f"Lambdas cannot be fingerprinted reliably. Define a named function instead of: {name}"
        )

    return name


def file_checksum(path: Path, max_bytes: int = 1024 * 1024) -> str:
    """Fast checksum of file contents.

    Hashes first max_bytes (default 1MB) for speed on large files.
    Returns "missing" if file doesn't exist.
    """
    try:
        with open(path, "rb") as f:
            content = f.read(max_bytes)
        return hashlib.sha256(content).hexdigest()[:16]
    except OSError:
        return "missing"


# ── Fingerprint Helpers ─────────────────────────────────────────────────────


def fingerprint_short(fp: dict[str, Any]) -> str:
    """Short format: 'abc123@def456*' (dirty marked with *)"""
    dirty = "*" if fp.get("git_dirty") else ""
    config_hash = str(fp.get("config_hash", "????????"))[:8]
    git_sha = fp.get("git_sha", "unknown")
    return f"{config_hash}@{git_sha}{dirty}"


# ── Eval Fingerprint ────────────────────────────────────────────────────────


def fingerprint_eval(
    config: EvalConfig,
    tools: list[str] | None = None,
    dataset_path: Path | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Compute fingerprint for an evaluation run.

    Args:
        config: EvalConfig with endpoint, score_fn, etc.
        tools: Actual tool names from environment (introspected at runtime)
        dataset_path: Path to dataset file for checksum
        allow_dirty: If False (default), error on uncommitted git changes

    Includes in hash:
        - endpoint (provider, model, params - minus api_key)
        - score_fn name
        - prepare_messages name
        - environment_factory name (if present)
        - handle_stop name (if present)
        - max_samples
        - dataset checksum (if path provided)

    Raises:
        RuntimeError: If git has uncommitted changes and allow_dirty=False
    """
    from dataclasses import asdict

    # Require clean git by default
    git_sha, git_dirty = require_clean_git(allow_dirty=allow_dirty)

    cfg: dict[str, Any] = {}

    # Endpoint (sanitize API key)
    endpoint_dict = asdict(config.endpoint)
    endpoint_dict.pop("api_key", None)
    cfg["endpoint"] = endpoint_dict

    # Function identities (names only - git SHA tracks code)
    cfg["score_fn"] = callable_name(config.score_fn)
    cfg["prepare_messages"] = callable_name(config.prepare_messages)

    if config.environment_factory:
        cfg["environment_factory"] = callable_name(config.environment_factory, allow_lambda=True)

    # Execution params that affect results
    cfg["max_samples"] = config.max_samples

    # Stop handler (if present in run_config)
    if config.run_config and config.run_config.handle_stop:
        cfg["handle_stop"] = callable_name(config.run_config.handle_stop, allow_lambda=True)

    # Dataset checksum
    if dataset_path:
        cfg["dataset_checksum"] = file_checksum(dataset_path)
        cfg["dataset_path"] = str(dataset_path)

    # Compute hash
    config_hash = hash_str(canonical_json(cfg))

    # Metadata for display
    return {
        "config_hash": config_hash,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "domain": "eval",
        "endpoint_summary": f"{config.endpoint.provider}/{config.endpoint.model}",
        "tools": tools or [],
        "dataset_path": str(dataset_path) if dataset_path else None,
    }


# ── Session Fingerprint ─────────────────────────────────────────────────────


def fingerprint_session(
    model: str,
    env: str,
    tools: list[str],
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Compute fingerprint for an interactive session.

    Args:
        model: Model identifier (e.g., "anthropic/claude-sonnet-4-5-20250929")
        env: Environment name (e.g., "coding", "git", "none")
        tools: List of tool names available
        system_prompt: Custom system prompt (hashed, not stored)

    Note: Sessions allow dirty git by default (interactive work often has
    uncommitted changes). The fingerprint still records git_dirty=True.
    """
    cfg: dict[str, Any] = {
        "model": model,
        "env": env,
        "tools": sorted(tools),
    }

    if system_prompt:
        cfg["system_prompt_hash"] = hash_str(system_prompt)

    config_hash = hash_str(canonical_json(cfg))
    git_sha, git_dirty = git_info()

    return {
        "config_hash": config_hash,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "domain": "session",
        "model": model,
        "env": env,
        "tools": sorted(tools),
    }


# ── Comparison Utilities ────────────────────────────────────────────────────


def compare_fingerprints(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Compare two fingerprints for reproducibility analysis."""
    return {
        "same_config": a.get("config_hash") == b.get("config_hash"),
        "same_code": a.get("git_sha") == b.get("git_sha"),
        "a": fingerprint_short(a),
        "b": fingerprint_short(b),
    }


def warn_if_changed(saved: dict[str, Any], current: dict[str, Any], context: str = "") -> None:
    """Print warning if fingerprint changed (for resume scenarios).

    Called when resuming a session or continuing an eval.
    """
    if not saved:
        return  # No saved fingerprint (legacy data)

    saved_hash = saved.get("config_hash", "")
    current_hash = current.get("config_hash", "")
    if saved_hash and current_hash and saved_hash != current_hash:
        print(f"⚠ Config changed since {context} was created", file=sys.stderr)
        print(f"  Saved:   {fingerprint_short(saved)}", file=sys.stderr)
        print(f"  Current: {fingerprint_short(current)}", file=sys.stderr)

    saved_git = saved.get("git_sha", "")
    current_git = current.get("git_sha", "")
    if saved_git and current_git and saved_git != current_git and saved_hash == current_hash:
        print(f"ℹ Code version changed: {saved_git} → {current_git}", file=sys.stderr)
