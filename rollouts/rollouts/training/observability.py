"""Small observability helpers for async RL pipeline state."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StatsProvider(Protocol):
    """Narrow protocol for components that can emit runtime stats."""

    def stats(self) -> dict[str, Any]: ...


def flatten_numeric_stats(
    data: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, float]:
    """Flatten nested numeric stats for metrics logging."""
    flat: dict[str, float] = {}
    for key, value in data.items():
        metric_key = f"{prefix}{key}"
        if isinstance(value, bool):
            flat[metric_key] = 1.0 if value else 0.0
            continue
        if isinstance(value, (int, float)):
            flat[metric_key] = float(value)
            continue
        if isinstance(value, dict):
            nested = flatten_numeric_stats(value, prefix=f"{metric_key}_")
            flat.update(nested)
    return flat
