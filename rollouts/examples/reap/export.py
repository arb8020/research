"""Export pruned model to HuggingFace format."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from typing import Any

import torch.nn as nn

logger = logging.getLogger(__name__)


def save_pruned_model(
    model: nn.Module,
    tokenizer: Any,
    output_dir: Path,
    config_overrides: dict | None = None,
) -> None:
    """Save pruned model in HuggingFace format.

    Args:
        model: The pruned model
        tokenizer: Associated tokenizer
        output_dir: Directory to save to
        config_overrides: Additional config values to save
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving pruned model to {output_dir}")

    # Save model weights
    model.save_pretrained(output_dir, safe_serialization=True)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    # Save additional metadata
    if config_overrides:
        metadata_path = output_dir / "reap_config.json"
        with open(metadata_path, "w") as f:
            json.dump(config_overrides, f, indent=2, default=str)
        logger.info(f"Saved REAP config to {metadata_path}")

    logger.info(f"Model saved to {output_dir}")


def get_output_path(
    base_dir: Path,
    model_name: str,
    dataset_name: str,
    prune_method: str,
    compression_ratio: float,
    seed: int,
) -> Path:
    """Generate output path for pruned model.

    Format: {base_dir}/{model_name}/{dataset_name}/{method}-seed_{seed}-{ratio}/
    """
    # Clean names for filesystem
    model_clean = model_name.replace("/", "_")
    dataset_clean = dataset_name.replace("/", "_")

    return (
        base_dir
        / model_clean
        / dataset_clean
        / f"{prune_method}-seed_{seed}-{compression_ratio:.2f}"
    )
