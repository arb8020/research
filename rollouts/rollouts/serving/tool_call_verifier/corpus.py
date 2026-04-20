"""K2VV corpus download + cache, modeled on HF-hub cache semantics.

First call downloads the 50%-sample tarball to the cache dir and extracts
it; later calls return the cached JSONL path. Idempotent — safe to call
from concurrent processes because the extraction step writes to a temp
dir and atomically renames.

Cache location defaults to $KIMI_VERIFIER_CACHE or
$XDG_CACHE_HOME/rollouts/kimi_verifier_v0/, falling back to
~/.cache/rollouts/kimi_verifier_v0/. On a remote node this resolves
relative to the node's $HOME.
"""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

K2VV_TARBALL_URL = "https://statics.moonshot.cn/k2vv/tool-calls.tar.gz"
# Inside the tarball, the JSONL currently lands at tool-calls/samples.jsonl.
# If upstream moves this, update or probe.
_TARBALL_JSONL_RELPATH = "tool-calls/samples.jsonl"


def _cache_root() -> Path:
    override = os.environ.get("KIMI_VERIFIER_CACHE")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "rollouts" / "kimi_verifier_v0"
    return Path.home() / ".cache" / "rollouts" / "kimi_verifier_v0"


def _download_tarball(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading K2VV corpus tarball to %s", dest)
    with urllib.request.urlopen(K2VV_TARBALL_URL) as response:  # noqa: S310
        with tempfile.NamedTemporaryFile(
            dir=str(dest.parent), delete=False, suffix=".partial"
        ) as tmp:
            shutil.copyfileobj(response, tmp)
            tmp_path = Path(tmp.name)
    tmp_path.rename(dest)


def _extract_tarball(tarball: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting K2VV corpus %s -> %s", tarball, target_dir)
    with tarfile.open(tarball, "r:gz") as tf:
        # data filter available on 3.12+; fall back to safe extract otherwise.
        extract_kwargs: dict[str, object] = {}
        if hasattr(tarfile, "data_filter"):
            extract_kwargs["filter"] = "data"
        tf.extractall(path=str(target_dir), **extract_kwargs)


def ensure_k2vv_corpus() -> Path:
    """Return a local path to the K2VV samples.jsonl, downloading if needed.

    Idempotent. If the tarball is already extracted, returns immediately.
    If only the tarball is cached, extracts it. If neither, downloads then
    extracts.
    """
    cache = _cache_root()
    tarball_path = cache / "tool-calls.tar.gz"
    extract_root = cache / "extracted"
    jsonl_path = extract_root / _TARBALL_JSONL_RELPATH

    if jsonl_path.exists():
        return jsonl_path

    if not tarball_path.exists():
        _download_tarball(tarball_path)

    _extract_tarball(tarball_path, extract_root)

    if not jsonl_path.exists():
        raise RuntimeError(
            f"Extracted K2VV corpus is missing expected entry {_TARBALL_JSONL_RELPATH!r} "
            f"under {extract_root}. Upstream tarball layout may have changed."
        )

    return jsonl_path
