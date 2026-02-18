"""Data buffer for managing training datasets.

Pure functional API with frozen state dataclass.
Follows CLASSES_VS_FUNCTIONAL.md: state is just data, operations are pure functions.

Pattern:
    samples, new_state = get_samples(samples, state, n=32)

This replaces the previous class-based DataBuffer.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Import Sample directly to avoid pulling in torch via training/__init__.py
from ...training.types import Sample

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class BufferState:
    """Immutable state for data buffer iteration.

    Frozen dataclass - state is just data, not behavior.
    All operations return new state instead of mutating.

    Attributes:
        epoch_id: Current epoch number (increments on wraparound)
        sample_offset: Current position in dataset
        seed: Random seed for deterministic shuffling

    Example:
        >>> state = BufferState(seed=42)
        >>> samples, new_state = get_samples(data, state, n=32)
        >>> assert new_state.sample_offset == 32
    """

    epoch_id: int = 0
    sample_offset: int = 0
    seed: int = 42


def get_samples(
    samples: list[Sample],
    state: BufferState,
    n: int,
    n_samples_per_prompt: int = 1,
) -> tuple[list[list[Sample]], BufferState]:
    """Get next batch of samples with GRPO grouping.

    Pure function: returns (sample_groups, new_state).
    Does not mutate inputs.

    Args:
        samples: Source dataset (list of Sample objects)
        state: Current buffer state
        n: Number of unique prompts to return
        n_samples_per_prompt: Copies per prompt for GRPO (default 1)

    Returns:
        Tuple of:
        - List of sample groups (each group has n_samples_per_prompt copies)
        - New buffer state

    Example:
        >>> samples = [Sample(prompt="Q1"), Sample(prompt="Q2")]
        >>> state = BufferState(seed=42)
        >>> groups, new_state = get_samples(samples, state, n=1, n_samples_per_prompt=4)
        >>> assert len(groups) == 1
        >>> assert len(groups[0]) == 4  # 4 copies of same prompt
        >>> assert new_state.sample_offset == 1
    """
    assert n > 0, "Must request at least 1 sample"
    assert len(samples) > 0, "Dataset is empty"
    assert n_samples_per_prompt >= 1, "n_samples_per_prompt must be >= 1"

    # Work with a shuffled copy based on current epoch
    shuffled = _shuffle_for_epoch(samples, state.seed, state.epoch_id)

    result_groups: list[list[Sample]] = []
    epoch_id = state.epoch_id
    sample_offset = state.sample_offset
    global_index = epoch_id * len(samples) + sample_offset

    while len(result_groups) < n:
        # Check if we need to advance epoch
        if sample_offset >= len(shuffled):
            epoch_id += 1
            sample_offset = 0
            shuffled = _shuffle_for_epoch(samples, state.seed, epoch_id)

        # Get the base sample
        base_sample = shuffled[sample_offset]

        # Create group with n_samples_per_prompt copies
        group: list[Sample] = []
        for i in range(n_samples_per_prompt):
            # Create copy with group_index and index set
            # Note: response is a computed property derived from trajectory, not a field
            sample_copy = Sample(
                prompt=base_sample.prompt,
                trajectory=base_sample.trajectory,
                tokens=list(base_sample.tokens),
                loss_mask=list(base_sample.loss_mask),
                reward=base_sample.reward,
                group_index=len(result_groups),
                index=global_index * n_samples_per_prompt + i,
                metadata=dict(base_sample.metadata),
                rollout_log_probs=base_sample.rollout_log_probs,
                status=base_sample.status,
            )
            group.append(sample_copy)

        result_groups.append(group)
        sample_offset += 1
        global_index += 1

    new_state = BufferState(
        epoch_id=epoch_id,
        sample_offset=sample_offset,
        seed=state.seed,
    )

    return result_groups, new_state


def get_samples_flat(
    samples: list[Sample],
    state: BufferState,
    n: int,
) -> tuple[list[Sample], BufferState]:
    """Get next batch of samples without grouping.

    Convenience wrapper for get_samples with n_samples_per_prompt=1.
    Returns flat list instead of nested groups.

    Args:
        samples: Source dataset
        state: Current buffer state
        n: Number of samples to return

    Returns:
        Tuple of (samples, new_state)

    Example:
        >>> samples = [Sample(prompt="Q1"), Sample(prompt="Q2")]
        >>> state = BufferState(seed=42)
        >>> batch, new_state = get_samples_flat(samples, state, n=2)
        >>> assert len(batch) == 2
    """
    groups, new_state = get_samples(samples, state, n, n_samples_per_prompt=1)
    return [group[0] for group in groups], new_state


def _shuffle_for_epoch(
    samples: list[Sample],
    seed: int,
    epoch_id: int,
) -> list[Sample]:
    """Create shuffled copy of samples for given epoch.

    Pure function - does not mutate input.

    Args:
        samples: Source samples
        seed: Base random seed
        epoch_id: Epoch number (combined with seed for determinism)

    Returns:
        New list with samples in shuffled order
    """
    rng = random.Random(seed + epoch_id)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    return shuffled


def state_to_dict(state: BufferState) -> dict[str, Any]:
    """Serialize buffer state for checkpointing.

    Args:
        state: Buffer state to serialize

    Returns:
        Dict representation

    Example:
        >>> state = BufferState(epoch_id=5, sample_offset=10, seed=42)
        >>> d = state_to_dict(state)
        >>> assert d["epoch_id"] == 5
    """
    return {
        "epoch_id": state.epoch_id,
        "sample_offset": state.sample_offset,
        "seed": state.seed,
    }


def state_from_dict(data: dict[str, Any]) -> BufferState:
    """Deserialize buffer state from checkpoint.

    Args:
        data: Dict from state_to_dict()

    Returns:
        BufferState instance

    Example:
        >>> d = {"epoch_id": 5, "sample_offset": 10, "seed": 42}
        >>> state = state_from_dict(d)
        >>> assert state.epoch_id == 5
    """
    return BufferState(
        epoch_id=data["epoch_id"],
        sample_offset=data["sample_offset"],
        seed=data["seed"],
    )


# ────────────────────── Dataset Loading ──────────────────────


def load_samples_from_parquet(
    path: Path | str,
    prompt_key: str = "prompt",
    label_key: str | None = None,
    limit: int | None = None,
) -> list[Sample]:
    """Load samples from Parquet file.

    Args:
        path: Path to parquet file (supports @[start:end] slice syntax)
        prompt_key: Column name for prompts
        label_key: Optional column name for ground truth labels
        limit: Optional limit on number of samples

    Returns:
        List of Sample objects

    Example:
        >>> samples = load_samples_from_parquet("data.parquet", label_key="answer")
        >>> samples = load_samples_from_parquet("data.parquet@[0:100]")  # first 100
    """
    import pandas as pd

    path_str = str(path)

    # Parse optional slice syntax: path@[start:end]
    slice_start, slice_end = None, None
    if "@[" in path_str and path_str.endswith("]"):
        path_str, slice_part = path_str.rsplit("@[", 1)
        slice_part = slice_part[:-1]  # Remove ]
        if ":" in slice_part:
            parts = slice_part.split(":")
            slice_start = int(parts[0]) if parts[0] else None
            slice_end = int(parts[1]) if parts[1] else None

    df = pd.read_parquet(path_str)

    # Apply slice
    if slice_start is not None or slice_end is not None:
        df = df.iloc[slice_start:slice_end]

    # Apply limit
    if limit is not None:
        df = df.head(limit)

    samples = []
    for _i, row in df.iterrows():
        prompt = row.get(prompt_key, row.to_dict())

        metadata: dict[str, Any] = {}
        if label_key and label_key in row:
            metadata["label"] = row[label_key]
        metadata["_raw"] = row.to_dict()

        samples.append(
            Sample(
                prompt=prompt,
                metadata=metadata,
                index=len(samples),
            )
        )

    return samples


def load_samples_from_hf(
    dataset_name: str,
    split: str = "train",
    subset: str | None = None,
    prompt_key: str = "prompt",
    label_key: str | None = None,
    limit: int | None = None,
) -> list[Sample]:
    """Load samples from HuggingFace datasets.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "openai/gsm8k")
        split: Dataset split (e.g., "train", "test")
        subset: Dataset subset/config (e.g., "main" for gsm8k)
        prompt_key: Field name for prompts
        label_key: Optional field name for ground truth labels
        limit: Optional limit on number of samples

    Returns:
        List of Sample objects

    Example:
        >>> samples = load_samples_from_hf(
        ...     "openai/gsm8k",
        ...     subset="main",
        ...     split="test",
        ...     prompt_key="question",
        ...     label_key="answer",
        ...     limit=100,
        ... )
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, subset, split=split)

    samples = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break

        prompt = row.get(prompt_key, dict(row))

        metadata: dict[str, Any] = {}
        if label_key and label_key in row:
            metadata["label"] = row[label_key]
        metadata["_raw"] = dict(row)

        samples.append(
            Sample(
                prompt=prompt,
                metadata=metadata,
                index=i,
            )
        )

    return samples


def load_samples_from_jsonl(
    path: Path,
    prompt_key: str = "prompt",
    label_key: str | None = None,
    limit: int | None = None,
) -> list[Sample]:
    """Load samples from JSONL file.

    Pure function - no side effects, just loads and returns.

    Args:
        path: Path to JSONL file
        prompt_key: Key to extract prompt from each line
        label_key: Optional key for ground truth label (stored in metadata)
        limit: Optional limit on number of samples to load

    Returns:
        List of Sample objects

    Example:
        >>> # If file contains: {"prompt": "Q1", "answer": "A1"}
        >>> samples = load_samples_from_jsonl("data.jsonl", label_key="answer")
        >>> assert samples[0].metadata["label"] == "A1"
    """
    samples = []

    with open(path) as f:
        for line_num, line in enumerate(f, start=1):
            if limit and line_num > limit:
                break

            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            # Extract prompt
            if prompt_key == "messages":
                prompt = data[prompt_key]
            elif prompt_key in data:
                prompt = data[prompt_key]
            else:
                prompt = data

            # Build metadata
            metadata: dict[str, Any] = {}
            if label_key and label_key in data:
                metadata["label"] = data[label_key]

            # Store full original data in metadata
            metadata["_raw"] = data

            samples.append(
                Sample(
                    prompt=prompt,
                    metadata=metadata,
                    index=line_num - 1,
                )
            )

    return samples


def load_samples_from_list(
    prompts: list[str | dict[str, Any]],
    labels: list[Any] | None = None,
) -> list[Sample]:
    """Create samples from Python lists.

    Args:
        prompts: List of prompts (strings or chat message dicts)
        labels: Optional list of ground truth labels

    Returns:
        List of Sample objects

    Example:
        >>> samples = load_samples_from_list(
        ...     prompts=["Q1", "Q2"],
        ...     labels=["A1", "A2"],
        ... )
        >>> assert samples[0].metadata["label"] == "A1"
    """
    samples = []

    for i, prompt in enumerate(prompts):
        metadata: dict[str, Any] = {}
        if labels is not None:
            metadata["label"] = labels[i]

        samples.append(
            Sample(
                prompt=prompt,
                metadata=metadata,
                index=i,
            )
        )

    return samples


# ────────────────────── Pretraining Data ──────────────────────


def get_token_batch(
    tokens: torch.Tensor,
    state: BufferState,
    batch_size: int,
    seq_len: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], BufferState]:
    """Get (input_ids, labels) batch for pretraining.

    Pure function: returns ((input_ids, labels), new_state).
    Labels are input_ids shifted by 1 (next token prediction).

    Args:
        tokens: 1D tensor of token IDs [total_tokens] - can be memmap'd
        state: Current buffer state (uses sample_offset as token position)
        batch_size: Number of sequences per batch
        seq_len: Length of each sequence

    Returns:
        Tuple of:
        - (input_ids, labels) tensors, each [batch_size, seq_len]
        - New buffer state

    Example:
        >>> import torch
        >>> tokens = torch.arange(1000)
        >>> state = BufferState(seed=42)
        >>> (inputs, labels), new_state = get_token_batch(tokens, state, batch_size=2, seq_len=128)
        >>> assert inputs.shape == (2, 128)
        >>> assert labels.shape == (2, 128)
        >>> assert (labels == inputs.roll(-1, dims=1)[:, :-1]).all()  # shifted by 1
    """

    total_tokens = tokens.shape[0]
    tokens_per_batch = batch_size * (seq_len + 1)  # +1 for labels shift

    # Current position in token stream
    offset = state.sample_offset
    epoch_id = state.epoch_id

    # Check if we need to wrap around
    if offset + tokens_per_batch > total_tokens:
        epoch_id += 1
        offset = 0

    # Extract contiguous chunk
    chunk = tokens[offset : offset + tokens_per_batch]

    # Reshape into sequences: [batch_size, seq_len + 1]
    chunk = chunk.view(batch_size, seq_len + 1)

    # Split into input and labels (shifted by 1)
    input_ids = chunk[:, :-1].contiguous()  # [batch_size, seq_len]
    labels = chunk[:, 1:].contiguous()  # [batch_size, seq_len]

    new_state = BufferState(
        epoch_id=epoch_id,
        sample_offset=offset + tokens_per_batch,
        seed=state.seed,
    )

    return (input_ids, labels), new_state


def load_tokens_from_bin(path: Path | str) -> torch.Tensor:
    """Load pre-tokenized data from .bin file (modded-nanogpt format).

    File format:
    - 256 int32 header (magic=20240520, version=1, num_tokens, ...)
    - uint16 tokens

    Args:
        path: Path to .bin file

    Returns:
        1D int64 tensor of token IDs
    """
    import torch

    path = Path(path)
    assert path.exists(), f"File not found: {path}"

    # Read header
    header = torch.from_file(str(path), shared=False, size=256, dtype=torch.int32)
    assert header[0] == 20240520, f"Invalid magic number: {header[0]}"
    assert header[1] == 1, f"Unsupported version: {header[1]}"
    num_tokens = int(header[2])

    # Read tokens
    with open(path, "rb") as f:
        f.seek(256 * 4)  # Skip header
        tokens = torch.frombuffer(f.read(), dtype=torch.uint16).to(torch.int64)

    assert len(tokens) == num_tokens, f"Token count mismatch: {len(tokens)} != {num_tokens}"
    return tokens


def load_tokens_from_npy(path: Path | str) -> torch.Tensor:
    """Load pre-tokenized data from .npy file (nmoe format).

    Args:
        path: Path to .npy file (uint32 tokens)

    Returns:
        1D int64 tensor of token IDs
    """
    import numpy as np
    import torch

    path = Path(path)
    assert path.exists(), f"File not found: {path}"

    # Memory-map for efficiency
    arr = np.load(str(path), mmap_mode="r")
    return torch.from_numpy(arr.astype(np.int64))


def load_fineweb_tokens(
    split: str = "train",
    num_chunks: int = 1,
    cache_dir: Path | str | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> torch.Tensor:
    """Download and load fineweb-10B tokens (GPT-2 tokenized).

    Uses kjj0/fineweb10B-gpt2 from HuggingFace Hub.
    Each chunk is ~100M tokens.

    For distributed training, each rank loads different chunks:
        - rank 0: chunks 1, 1+world_size, 1+2*world_size, ...
        - rank 1: chunks 2, 2+world_size, 2+2*world_size, ...

    Args:
        split: "train" or "val"
        num_chunks: Total number of 100M token chunks to load (train has 103 chunks)
        cache_dir: Where to cache downloaded files (default: ~/.cache/fineweb10B)
        rank: Current process rank (for distributed sharding)
        world_size: Total number of processes (for distributed sharding)

    Returns:
        1D int64 tensor of token IDs
    """
    import torch
    from huggingface_hub import hf_hub_download

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "fineweb10B"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    chunks = []

    if split == "val":
        # Validation is a single file - all ranks load it
        fname = "fineweb_val_000000.bin"
        local_path = cache_dir / fname
        if not local_path.exists():
            hf_hub_download(
                repo_id="kjj0/fineweb10B-gpt2",
                filename=fname,
                repo_type="dataset",
                local_dir=str(cache_dir),
            )
        chunks.append(load_tokens_from_bin(local_path))
    else:
        # Training chunks - shard across ranks
        # Each rank loads chunks: rank+1, rank+1+world_size, rank+1+2*world_size, ...
        for i in range(rank + 1, num_chunks + 1, world_size):
            fname = f"fineweb_train_{i:06d}.bin"
            local_path = cache_dir / fname
            if not local_path.exists():
                hf_hub_download(
                    repo_id="kjj0/fineweb10B-gpt2",
                    filename=fname,
                    repo_type="dataset",
                    local_dir=str(cache_dir),
                )
            chunks.append(load_tokens_from_bin(local_path))

    return torch.cat(chunks) if len(chunks) > 1 else chunks[0]


# ────────────────────── Legacy Class (TODO: migrate callers) ──────────────────────


@dataclass
class DataBuffer:
    """Stateful wrapper for backwards compatibility.

    TODO: Migrate callers to use BufferState + get_samples() instead:
        # Old (stateful class):
        buffer = DataBuffer(prompts=["Q1", "Q2"])
        batch = buffer.get_prompts(2)

        # New (functional):
        samples = load_samples_from_list(["Q1", "Q2"])
        state = BufferState(seed=42)
        batch, state = get_samples_flat(samples, state, n=2)

    Callers to migrate:
        - rollouts/training/grpo.py
        - rollouts/training/rollout_gen/async_rollout_manager.py
        - rollouts/training/rollout_gen/rollout_generation.py
        - rollouts/training/loops/rl_loop.py
    """

    prompts: list[str | dict[str, Any]]
    epoch_id: int = 0
    sample_offset: int = 0
    seed: int = 42

    def __post_init__(self) -> None:
        self._samples = load_samples_from_list(self.prompts)
        self._state = BufferState(
            epoch_id=self.epoch_id,
            sample_offset=self.sample_offset,
            seed=self.seed,
        )

    def get_prompts(self, n: int) -> list[str | dict[str, Any]]:
        """Get next batch of prompts."""
        samples, self._state = get_samples_flat(self._samples, self._state, n)
        self.epoch_id = self._state.epoch_id
        self.sample_offset = self._state.sample_offset
        return [s.prompt for s in samples]

    def save_state(self) -> dict[str, Any]:
        return state_to_dict(self._state)

    def load_state(self, state: dict[str, Any]) -> None:
        self._state = state_from_dict(state)
        self.epoch_id = self._state.epoch_id
        self.sample_offset = self._state.sample_offset
