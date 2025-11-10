"""Data buffer for managing prompt datasets.

SLIME-inspired stateful data source with minimal state:
- Tracks position in dataset (epoch_id, sample_offset)
- Handles epoch boundaries with deterministic shuffling
- Provides next batch of prompts

This is the ONLY stateful component in the training system.
Everything else is pure functions.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataBuffer:
    """Stateful data source for training prompts.

    This is the only component with mutable state in the training system.
    Inspired by SLIME's RolloutDataSource.

    Attributes:
        prompts: List of prompt strings or dicts (chat messages)
        epoch_id: Current epoch number (increments on wraparound)
        sample_offset: Current position in dataset
        seed: Random seed for deterministic shuffling
        _rng: Internal random number generator (lazily initialized)

    Example:
        >>> buffer = DataBuffer(prompts=["What is 2+2?", "Calculate 5*7"])
        >>> batch = buffer.get_prompts(1)
        >>> assert batch == ["What is 2+2?"]
        >>> batch = buffer.get_prompts(2)  # Wraps to next epoch
        >>> assert len(batch) == 2
        >>> assert buffer.epoch_id == 1
    """

    prompts: list[str | dict[str, Any]]
    epoch_id: int = 0
    sample_offset: int = 0
    seed: int = 42
    _rng: random.Random = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize RNG with seed."""
        if self._rng is None:
            self._rng = random.Random(self.seed + self.epoch_id)

    def get_prompts(self, n: int) -> list[str | dict[str, Any]]:
        """Get next batch of prompts, handling epoch wraparound.

        SLIME-style API: Returns next n prompts from dataset.
        When reaching end of dataset:
        - Increments epoch_id
        - Shuffles with new seed (deterministic based on epoch_id)
        - Continues from beginning

        Args:
            n: Number of prompts to return

        Returns:
            List of prompts (may be strings or chat message dicts)

        Example:
            >>> buffer = DataBuffer(prompts=["a", "b", "c"])
            >>> buffer.get_prompts(2)
            ['a', 'b']
            >>> buffer.get_prompts(2)  # Wraps
            ['c', 'a']  # New epoch, shuffled
        """
        # Preconditions
        assert n > 0, "Must request at least 1 prompt"
        assert len(self.prompts) > 0, "Dataset is empty"

        result = []

        while len(result) < n:
            # How many we can get from current epoch
            remaining_in_epoch = len(self.prompts) - self.sample_offset
            needed = n - len(result)
            take = min(remaining_in_epoch, needed)

            # Take from current position
            result.extend(
                self.prompts[self.sample_offset : self.sample_offset + take]
            )
            self.sample_offset += take

            # Wraparound if needed
            if self.sample_offset >= len(self.prompts):
                self._advance_epoch()

        # Postcondition
        assert len(result) == n, "Must return exact number of prompts requested"

        return result

    def _advance_epoch(self):
        """Advance to next epoch with deterministic shuffle."""
        self.epoch_id += 1
        self.sample_offset = 0

        # Shuffle with deterministic seed based on epoch
        self._rng = random.Random(self.seed + self.epoch_id)
        self._rng.shuffle(self.prompts)

    def save_state(self) -> dict[str, Any]:
        """Save buffer state for checkpointing.

        Returns:
            Dict with epoch_id, sample_offset, seed

        Example:
            >>> buffer = DataBuffer(prompts=["a", "b"])
            >>> buffer.get_prompts(1)
            >>> state = buffer.save_state()
            >>> assert state["sample_offset"] == 1
        """
        return {
            "epoch_id": self.epoch_id,
            "sample_offset": self.sample_offset,
            "seed": self.seed,
        }

    def load_state(self, state: dict[str, Any]):
        """Restore buffer state from checkpoint.

        Args:
            state: Dict from save_state()

        Example:
            >>> buffer = DataBuffer(prompts=["a", "b"])
            >>> buffer.load_state({"epoch_id": 5, "sample_offset": 1, "seed": 42})
            >>> assert buffer.epoch_id == 5
        """
        self.epoch_id = state["epoch_id"]
        self.sample_offset = state["sample_offset"]
        self.seed = state["seed"]
        self._rng = random.Random(self.seed + self.epoch_id)

        # Re-shuffle to correct epoch state
        for _ in range(self.epoch_id):
            temp_rng = random.Random(self.seed + _)
            temp_rng.shuffle(self.prompts)


def load_prompts_from_jsonl(
    path: Path,
    prompt_key: str = "prompt",
    limit: int | None = None,
) -> list[str | dict[str, Any]]:
    """Load prompts from JSONL file.

    Pure function - no side effects, just loads and returns.

    Args:
        path: Path to JSONL file
        prompt_key: Key to extract prompt from each line
        limit: Optional limit on number of prompts to load

    Returns:
        List of prompts (strings or dicts)

    Example:
        >>> # If file contains: {"prompt": "Q1"}\\n{"prompt": "Q2"}
        >>> prompts = load_prompts_from_jsonl("data.jsonl")
        >>> assert len(prompts) == 2

    Note:
        If prompt_key is "messages", returns the raw dict (for chat messages).
        Otherwise extracts the specific key value.
    """
    prompts = []

    with open(path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            if limit and line_num > limit:
                break

            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            # Extract prompt based on key
            if prompt_key == "messages":
                # Keep full dict for chat messages
                prompts.append(data[prompt_key])
            elif prompt_key in data:
                prompts.append(data[prompt_key])
            else:
                # Fallback: use the whole dict
                prompts.append(data)

    return prompts


def load_prompts_from_list(
    prompts: list[str],
) -> list[str]:
    """Load prompts from Python list.

    Pure function - just returns the input for consistency with other loaders.

    Args:
        prompts: List of prompt strings

    Returns:
        Same list (for API consistency)

    Example:
        >>> prompts = load_prompts_from_list(["Q1", "Q2"])
        >>> assert len(prompts) == 2
    """
    return prompts


def create_buffer_from_jsonl(
    path: Path,
    prompt_key: str = "prompt",
    limit: int | None = None,
    seed: int = 42,
) -> DataBuffer:
    """Create DataBuffer from JSONL file.

    Convenience function combining load + buffer creation.

    Args:
        path: Path to JSONL file
        prompt_key: Key to extract prompt from each line
        limit: Optional limit on number of prompts
        seed: Random seed for shuffling

    Returns:
        DataBuffer initialized with prompts

    Example:
        >>> buffer = create_buffer_from_jsonl("prompts.jsonl", limit=100)
        >>> batch = buffer.get_prompts(32)
    """
    prompts = load_prompts_from_jsonl(path, prompt_key, limit)
    return DataBuffer(prompts=prompts, seed=seed)
