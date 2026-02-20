"""iGSM synthetic data generator for pretraining.

Generates grade-school math word problems with step-by-step solutions on-the-fly.

Based on:
- https://github.com/facebookresearch/iGSM
- Physics of Language Models: Part 2.1 and 2.2

Requirements:
    git clone https://github.com/facebookresearch/iGSM.git /tmp/iGSM
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Any, Literal
from unittest.mock import MagicMock

import numpy as np
import torch

# iGSM path
_IGSM_PATH = "/tmp/iGSM"
_igsm_setup_done = False


def _setup_igsm_imports() -> None:
    """Set up imports for iGSM, handling path conflicts with rollouts/tools/."""
    global _igsm_setup_done
    if _igsm_setup_done:
        return

    # Remove any cached tools module that might conflict
    for mod in list(sys.modules.keys()):
        if mod == "tools" or mod.startswith("tools."):
            del sys.modules[mod]

    # Insert iGSM path at the very beginning
    if _IGSM_PATH in sys.path:
        sys.path.remove(_IGSM_PATH)
    sys.path.insert(0, _IGSM_PATH)

    # Mock matplotlib (only used for visualization, not generation)
    for mod_name in [
        "matplotlib",
        "matplotlib.pyplot",
        "matplotlib.patches",
        "matplotlib.lines",
        "matplotlib.colors",
        "matplotlib.cm",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()

    _igsm_setup_done = True


def ensure_igsm_available() -> bool:
    """Check if iGSM is available."""
    _setup_igsm_imports()
    try:
        from data_gen.pretrain.id_gen import IdGen  # noqa: F401
        from tools.tools import fix_seed, tokenizer  # noqa: F401

        return True
    except ImportError as e:
        print(f"iGSM not available: {e}")
        print(f"Make sure iGSM is cloned to {_IGSM_PATH}")
        return False


def get_tokenizer():
    """Get the GPT2 tokenizer used by iGSM."""
    _setup_igsm_imports()
    from tools.tools import tokenizer

    return tokenizer


@dataclass
class iGSMConfig:
    """Configuration for iGSM data generation."""

    # Difficulty parameters
    max_op: int = 15  # Maximum number of operations
    max_edge: int = 20  # Maximum number of edges in structure graph

    # Problem formatting
    perm_level: int = 5  # Random shuffle level (5 = full randomization)
    detail_level: int = 0  # Solution detail level (0 = most detailed)
    be_shortest: bool = True  # Use shortest solution path

    # Data format
    p_format: str = "pq"  # Problem format ("pq" = problem-question)

    # Token IDs (GPT2 tokenizer)
    prob_start_token: int = 222
    sol_start_token: int = 223
    ans_start_token: int = 224
    eos_token: int = 50256

    # Vocabulary (GPT2 has 50257 tokens)
    vocab_size: int = 50257

    def __post_init__(self):
        assert self.max_op > 0, f"max_op must be positive, got {self.max_op}"
        assert self.max_edge > 0, f"max_edge must be positive, got {self.max_edge}"


class iGSMDataLoader:
    """Data loader that generates iGSM math problems on-the-fly.

    Compatible with rollouts/pretrain/dataloader.py interface.
    """

    def __init__(
        self,
        config: iGSMConfig,
        seq_len: int,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        device: str | torch.device = "cuda",
        seed: int = 42,
        mode: Literal["train", "val"] = "train",
    ):
        """Initialize iGSM data loader.

        Args:
            config: iGSM configuration
            seq_len: Maximum sequence length (will pad/truncate)
            batch_size: Batch size
            rank: Current process rank (for distributed training)
            world_size: Total number of processes
            device: Device for output tensors
            seed: Random seed
            mode: "train" or "val" (affects RNG seed)
        """
        if not ensure_igsm_available():
            raise RuntimeError("iGSM not available - clone to /tmp/iGSM")

        self.config = config
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.mode = mode

        # Each rank gets its own seed
        self.seed = seed + rank + (1000000 if mode == "val" else 0)

        # Import iGSM components
        from data_gen.pretrain.id_gen import IdGen
        from tools.tools import fix_seed

        self.IdGen = IdGen
        self.fix_seed = fix_seed

        # Initialize RNG
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.RandomState(self.seed)

        # Global sequence counter for deterministic resume
        self._global_seq_idx = 0

        # Compute per-step slicing for distributed training
        self._seqs_per_step = batch_size
        q, r = divmod(self._seqs_per_step, world_size)
        self._mine = q + (1 if rank < r else 0)
        self._start_off = rank * q + min(rank, r)

        # Available hash values for problem generation
        self._ava_hash_pool = list(range(23))

    def _generate_one(self) -> list[int]:
        """Generate a single problem-solution-answer sequence.

        Returns:
            Token ID sequence: [prob_start] + prob + [sol_start] + sol + [ans_start] + ans + [eos]
        """
        gen_seed = self.seed + self._global_seq_idx
        self.fix_seed(gen_seed)

        id_gen = self.IdGen(
            max_op=self.config.max_op,
            max_edge=self.config.max_edge,
            perm_level=self.config.perm_level,
            detail_level=self.config.detail_level,
            be_shortest=self.config.be_shortest,
        )

        id_gen.gen_prob(self._ava_hash_pool, p_format=self.config.p_format)

        token_ids = (
            [self.config.prob_start_token]
            + id_gen.prob_token
            + [self.config.sol_start_token]
            + id_gen.sol_token
            + [self.config.ans_start_token]
            + id_gen.ans_token
            + [self.config.eos_token]
        )

        self._global_seq_idx += 1
        return token_ids

    def _pad_or_truncate(self, seq: list[int]) -> list[int]:
        """Pad or truncate sequence to seq_len + 1 (for labels)."""
        if len(seq) >= self.seq_len + 1:
            return seq[: self.seq_len + 1]
        else:
            return seq + [self.config.eos_token] * (self.seq_len + 1 - len(seq))

    @torch.no_grad()
    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get next batch of (input_ids, labels).

        Returns:
            input_ids: [batch_size, seq_len]
            labels: [batch_size, seq_len] (shifted by 1)
        """
        seqs: list[list[int]] = []

        for i in range(self._seqs_per_step):
            if self._start_off <= i < self._start_off + self._mine:
                seq = self._generate_one()
                seq = self._pad_or_truncate(seq)
                seqs.append(seq)
            else:
                # Still advance counter for consistency across ranks
                self._global_seq_idx += 1

        if not seqs:
            raise StopIteration("No sequences for this rank")

        batch = torch.tensor(seqs, device=self.device)
        return batch[:, :-1], batch[:, 1:]

    def state_dict(self) -> dict[str, Any]:
        """Get loader state for checkpointing."""
        return {
            "version": 1,
            "global_seq_idx": self._global_seq_idx,
            "seed": self.seed,
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore loader state from checkpoint."""
        self._global_seq_idx = state.get("global_seq_idx", 0)
        rng_state = state.get("rng_state")
        if rng_state:
            self.rng.setstate(rng_state)


# Difficulty presets from paper
DIFFICULTY_PRESETS = {
    "easy": {"max_op": 10, "max_edge": 15},
    "med": {"max_op": 15, "max_edge": 20},
    "hard": {"max_op": 21, "max_edge": 28},
}


def build_igsm_loader(
    difficulty: Literal["easy", "med", "hard"] = "med",
    seq_len: int = 512,
    batch_size: int = 4,
    rank: int = 0,
    world_size: int = 1,
    device: str | torch.device = "cuda",
    seed: int = 42,
    mode: Literal["train", "val"] = "train",
    **kwargs: Any,
) -> iGSMDataLoader:
    """Build an iGSM data loader with preset difficulty.

    Args:
        difficulty: "easy", "med", or "hard"
        seq_len: Sequence length
        batch_size: Batch size
        rank: Current process rank
        world_size: Total number of processes
        device: Device for output tensors
        seed: Random seed
        mode: "train" or "val"
        **kwargs: Additional args for iGSMConfig (override presets)

    Returns:
        iGSMDataLoader instance
    """
    if difficulty not in DIFFICULTY_PRESETS:
        raise ValueError(
            f"Unknown difficulty: {difficulty}. Choose from {list(DIFFICULTY_PRESETS.keys())}"
        )

    preset = DIFFICULTY_PRESETS[difficulty].copy()
    preset.update(kwargs)

    config = iGSMConfig(**preset)

    return iGSMDataLoader(
        config=config,
        seq_len=seq_len,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        device=device,
        seed=seed,
        mode=mode,
    )
