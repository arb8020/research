"""Dataset loading for calibration."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def load_calibration_data(
    dataset_name: str,
    tokenizer: Any,
    num_samples: int,
    max_seq_len: int,
    seed: int = 42,
) -> list[dict[str, Tensor]]:
    """Load and tokenize calibration dataset.

    Args:
        dataset_name: HuggingFace dataset identifier
        tokenizer: Tokenizer for the model
        num_samples: Number of samples to load
        max_seq_len: Maximum sequence length
        seed: Random seed for shuffling

    Returns:
        List of dicts with 'input_ids' and 'attention_mask' tensors
    """
    from datasets import load_dataset

    logger.info(f"Loading {num_samples} samples from {dataset_name}")

    # Load dataset
    if "/" in dataset_name:
        ds = load_dataset(dataset_name, split="train")
    else:
        ds = load_dataset(dataset_name, split="train")

    # Shuffle and select
    ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))

    # Determine text field
    text_field = _get_text_field(ds)

    # Tokenize
    samples = []
    for row in ds:
        text = row[text_field]
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)

        tokens = tokenizer(
            text,
            max_length=max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        # Skip very short sequences
        if tokens["input_ids"].shape[1] < 32:
            continue

        samples.append({
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
        })

        if len(samples) >= num_samples:
            break

    logger.info(f"Loaded {len(samples)} samples")
    return samples


def _get_text_field(dataset: Any) -> str:
    """Determine the text field name in the dataset."""
    # Common field names
    candidates = ["text", "content", "instruction", "prompt", "question", "input"]

    features = dataset.features
    for candidate in candidates:
        if candidate in features:
            return candidate

    # Fall back to first string field
    for name, feature in features.items():
        if hasattr(feature, "dtype") and feature.dtype == "string":
            return name

    raise ValueError(f"Could not find text field in dataset. Fields: {list(features.keys())}")


def batch_iterator(
    samples: list[dict[str, Tensor]],
    batch_size: int,
    device: torch.device,
) -> Iterator[dict[str, Tensor]]:
    """Yield batches of samples, padded to same length.

    Args:
        samples: List of tokenized samples
        batch_size: Number of samples per batch
        device: Device to put tensors on

    Yields:
        Batched and padded tensors
    """
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]

        # Find max length in batch
        max_len = max(s["input_ids"].shape[0] for s in batch)

        # Pad sequences
        input_ids = []
        attention_mask = []

        for s in batch:
            seq_len = s["input_ids"].shape[0]
            pad_len = max_len - seq_len

            input_ids.append(
                torch.cat([
                    s["input_ids"],
                    torch.zeros(pad_len, dtype=s["input_ids"].dtype),
                ])
            )
            attention_mask.append(
                torch.cat([
                    s["attention_mask"],
                    torch.zeros(pad_len, dtype=s["attention_mask"].dtype),
                ])
            )

        yield {
            "input_ids": torch.stack(input_ids).to(device),
            "attention_mask": torch.stack(attention_mask).to(device),
        }
