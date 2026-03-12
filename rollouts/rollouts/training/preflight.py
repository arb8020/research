"""Workload-owned runtime compatibility preflights.

These checks belong in Rollouts because they define workload/runtime semantics:

- what package/API combinations are valid for a backend
- what imports/symbols must exist before a backend can initialize
- what stage name and failure meaning we assign to that validation

Argus may record the results, but it should not define this ontology.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RuntimePreflightResult:
    """Result of a workload-owned runtime compatibility check."""

    name: str
    stage: str
    ok: bool
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def require_ok(self) -> None:
        if self.ok:
            return
        detail_lines = [f"{key}={value}" for key, value in sorted(self.details.items())]
        suffix = f" Details: {', '.join(detail_lines)}" if detail_lines else ""
        raise RuntimeError(
            f"{self.stage} failed for {self.name}: {self.error or 'unknown error'}."
            f"{suffix}"
        )


def preflight_torchtitan_runtime() -> RuntimePreflightResult:
    """Check that the current runtime can import and initialize TorchTitan.

    This is intentionally cheap. It should fail before any real training work
    starts and before distributed init if the environment is incompatible.
    """

    details: dict[str, Any] = {}

    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised in target runtime
        return RuntimePreflightResult(
            name="torchtitan",
            stage="TRAIN_BACKEND_IMPORT_OK",
            ok=False,
            details=details,
            error=f"torch import failed: {type(exc).__name__}: {exc}",
        )

    details["torch_version"] = getattr(torch, "__version__", "unknown")
    details["torch_file"] = getattr(torch, "__file__", "unknown")
    details["has_torch_attention_varlen"] = (
        importlib.util.find_spec("torch.nn.attention.varlen") is not None
    )

    if not details["has_torch_attention_varlen"]:
        return RuntimePreflightResult(
            name="torchtitan",
            stage="TRAIN_BACKEND_IMPORT_OK",
            ok=False,
            details=details,
            error="required module 'torch.nn.attention.varlen' is missing",
        )

    try:
        torchtitan = importlib.import_module("torchtitan")
        details["torchtitan_file"] = getattr(torchtitan, "__file__", "unknown")
    except Exception as exc:  # pragma: no cover - exercised in target runtime
        return RuntimePreflightResult(
            name="torchtitan",
            stage="TRAIN_BACKEND_IMPORT_OK",
            ok=False,
            details=details,
            error=f"torchtitan import failed: {type(exc).__name__}: {exc}",
        )

    try:
        importlib.import_module("torchtitan.protocols.train_spec")
    except Exception as exc:  # pragma: no cover - exercised in target runtime
        return RuntimePreflightResult(
            name="torchtitan",
            stage="TRAIN_BACKEND_IMPORT_OK",
            ok=False,
            details=details,
            error=f"train_spec import failed: {type(exc).__name__}: {exc}",
        )

    return RuntimePreflightResult(
        name="torchtitan",
        stage="TRAIN_BACKEND_IMPORT_OK",
        ok=True,
        details=details,
    )


def require_torchtitan_runtime() -> RuntimePreflightResult:
    """Run the TorchTitan preflight and raise on failure."""

    result = preflight_torchtitan_runtime()
    result.require_ok()
    return result
