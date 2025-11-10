"""Metrics logging for training (separate from error logging).

Tiger Style: Simple, bounded, explicit.
Casey: Protocol-based, no retention, dependency injection.
Sean: Boring file writes, no magic.

This is for TRAINING METRICS (loss, reward, etc).
For ERROR LOGS, use shared/logging_config.py instead.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

# ══════════════════════════════════════════════════════════════
# Protocol (Casey: minimal interface, decoupled)
# ══════════════════════════════════════════════════════════════

class MetricsLogger(Protocol):
    """Protocol for metrics logging.

    Tiger Style: Minimal interface, no complex class hierarchy.
    Casey: No retention - just log and finish.
    Ray-Ready: Protocol allows swapping implementations for distributed.

    Example:
        >>> logger = JSONLLogger(Path("logs/exp_001"))
        >>> logger.log({"loss": 0.5, "reward": 0.3}, step=100)
        >>> logger.finish()
    """

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics for a training step.

        Args:
            metrics: Dict of metric_name -> value (numeric)
            step: Training step number
        """
        ...

    def finish(self) -> None:
        """Cleanup and finalize logging (flush buffers, close files)."""
        ...


# ══════════════════════════════════════════════════════════════
# JSONL Logger (Default, no dependencies)
# ══════════════════════════════════════════════════════════════

@dataclass
class JSONLLogger:
    """JSONL metrics logger (default, no dependencies).

    Tiger Style: Bounded file size, explicit, simple file writes.
    Sean: Boring - just append JSON lines to a file.

    File format (metrics.jsonl):
        {"step": 0, "timestamp": 1704816000.0, "loss": 1.0, "reward": 0.3}
        {"step": 1, "timestamp": 1704816000.1, "loss": 0.9, "reward": 0.4}
        ...

    Example:
        >>> logger = JSONLLogger(Path("logs/exp_001"))
        >>> logger.log({"loss": 0.5}, step=100)
        >>> logger.finish()
    """

    log_dir: Path
    max_lines: int = 1_000_000  # Tiger: Bounded! Prevents unbounded growth

    def __post_init__(self):
        # Tiger: Assert preconditions
        assert self.log_dir is not None, "log_dir cannot be None"
        assert self.max_lines > 0, f"max_lines must be > 0, got {self.max_lines}"

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self._line_count = 0

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics (pure function, no retention).

        Tiger: Assert inputs, bounded writes.
        Sean: Boring file append.
        """
        # Tiger: Assert preconditions
        assert isinstance(metrics, dict), f"metrics must be dict, got {type(metrics)}"
        assert isinstance(step, int), f"step must be int, got {type(step)}"
        assert step >= 0, f"step must be >= 0, got {step}"

        # Tiger: Assert metric values are numeric
        for key, value in metrics.items():
            assert isinstance(value, (int, float)), \
                f"Metric '{key}' must be numeric, got {type(value)}"

        # Tiger: Bounded - prevent unbounded growth
        if self._line_count >= self.max_lines:
            # Could rotate file here (metrics.jsonl -> metrics.1.jsonl)
            # For now, just stop logging (fail-safe)
            return

        # Create entry (Sean: explicit, no magic)
        entry = {
            "step": step,
            "timestamp": time.time(),
            **metrics,
        }

        # Write JSONL (one JSON object per line)
        with open(self.metrics_file, "a") as f:
            json.dump(entry, f)
            f.write("\n")

        self._line_count += 1

    def finish(self) -> None:
        """Cleanup (nothing to do for file-based logging)."""
        pass  # Files are auto-closed


# ══════════════════════════════════════════════════════════════
# W&B Logger (Optional, requires wandb package)
# ══════════════════════════════════════════════════════════════

@dataclass
class WandbLogger:
    """Weights & Biases logger (optional, requires wandb package).

    Example:
        >>> logger = WandbLogger(project="my-project", name="exp-001")
        >>> logger.log({"loss": 0.5}, step=100)
        >>> logger.finish()
    """

    project: str
    name: str | None = None
    entity: str | None = None
    tags: list[str] | None = None

    def __post_init__(self):
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            raise ImportError(
                "wandb not installed. Install with: pip install wandb"
            )

        # Initialize W&B run
        self.run = self.wandb.init(
            project=self.project,
            name=self.name,
            entity=self.entity,
            tags=self.tags,
        )

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log to W&B."""
        self.wandb.log(metrics, step=step)

    def finish(self) -> None:
        """Finish W&B run."""
        self.run.finish()


# ══════════════════════════════════════════════════════════════
# Composite Logger (Casey: redundancy - multiple backends)
# ══════════════════════════════════════════════════════════════

@dataclass
class CompositeLogger:
    """Log to multiple backends simultaneously.

    Casey: Redundancy - log to JSONL (always) + W&B (dashboards).

    Example:
        >>> logger = CompositeLogger([
        ...     JSONLLogger(Path("logs/exp_001")),
        ...     WandbLogger(project="my-project"),
        ... ])
        >>> logger.log({"loss": 0.5}, step=100)
        >>> logger.finish()
    """

    loggers: list[MetricsLogger]

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log to all backends."""
        for logger in self.loggers:
            logger.log(metrics, step)

    def finish(self) -> None:
        """Finish all backends."""
        for logger in self.loggers:
            logger.finish()


# ══════════════════════════════════════════════════════════════
# Helper: Compute statistics from JSONL (Casey: no retention)
# ══════════════════════════════════════════════════════════════

def compute_stats_from_jsonl(
    metrics_file: Path,
    key: str,
) -> dict[str, float]:
    """Compute statistics for a metric from JSONL file.

    Casey: No retention - compute on demand from file instead of keeping
    accumulators in memory during training.

    Args:
        metrics_file: Path to metrics.jsonl
        key: Metric name (e.g., "loss", "reward")

    Returns:
        Dict with mean, min, max, std of the metric

    Example:
        >>> stats = compute_stats_from_jsonl(Path("logs/exp_001/metrics.jsonl"), "loss")
        >>> print(f"Mean loss: {stats['mean']:.4f}")
    """
    import json

    values = []
    with open(metrics_file) as f:
        for line in f:
            entry = json.loads(line)
            if key in entry:
                values.append(entry[key])

    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}

    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std = variance ** 0.5

    return {
        "mean": mean,
        "min": min(values),
        "max": max(values),
        "std": std,
    }
