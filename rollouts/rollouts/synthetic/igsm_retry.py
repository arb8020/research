"""iGSM retry/correction data generator for learning from mistakes.

Based on Physics of Language Models: Part 2.2
"How to Learn From Mistakes on Grade-School Math Problems"

Generates synthetic "retry" data where the model:
1. Makes a mistake (uses wrong parameter)
2. Says "[BACK]" (retry keyword)
3. Corrects itself with the right parameter

Usage:
    from rollouts.synthetic import build_igsm_retry_loader

    loader = build_igsm_retry_loader(
        difficulty="med",
        retry_rate=0.1,
        retry_type="strong",
        seq_len=768,
        batch_size=4,
    )
    input_ids, labels = loader.next()

Requirements:
    git clone https://github.com/facebookresearch/iGSM.git /tmp/iGSM
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Any, Literal
from unittest.mock import MagicMock

import torch

from .igsm import _IGSM_PATH, ensure_igsm_available

# Ensure iGSM path is set up
if _IGSM_PATH not in sys.path:
    sys.path.insert(0, _IGSM_PATH)

# Mock matplotlib
for _mod_name in [
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.patches",
    "matplotlib.lines",
    "matplotlib.colors",
    "matplotlib.cm",
]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()


# The retry token used in the paper
RETRY_TOKEN = "[BACK]"


def _get_retry_keyword_tokens() -> list[int]:
    """Get token IDs for the retry keyword."""
    from const.params import retry_key_word
    from tools.tools import tokenizer

    return tokenizer.encode(" " + retry_key_word + ".", return_tensors="pt")[0].tolist()


@dataclass
class iGSMRetryConfig:
    """Configuration for iGSM retry data generation."""

    # Base iGSM params
    max_op: int = 15
    max_edge: int = 20
    perm_level: int = 5
    detail_level: int = 0

    # Retry-specific params
    retry_rate: float = 0.1  # Probability of inserting a retry at each step
    retry_type: Literal["strong", "weak"] = "strong"
    # strong: wrong param must not have appeared yet in solution
    # weak: any future param works (simpler, nearly as good per paper)

    # Token IDs
    prob_start_token: int = 222
    sol_start_token: int = 223
    ans_start_token: int = 224
    eos_token: int = 50256
    vocab_size: int = 50257

    def __post_init__(self):
        assert 0 <= self.retry_rate <= 1, f"retry_rate must be in [0, 1], got {self.retry_rate}"


class iGSMRetryDataLoader:
    """Data loader that generates iGSM problems with retry/correction pairs."""

    def __init__(
        self,
        config: iGSMRetryConfig,
        seq_len: int,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        device: str | torch.device = "cuda",
        seed: int = 42,
        mode: Literal["train", "val"] = "train",
    ):
        if not ensure_igsm_available():
            raise RuntimeError("iGSM not available - clone to /tmp/iGSM")

        self.config = config
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.mode = mode

        self.seed = seed + rank + (1000000 if mode == "val" else 0)

        # Import retry generators
        from data_gen.pretrain.id_retry_gen import IdGen as IdRetryGen
        from data_gen.pretrain.id_retry_weak_gen import IdGen as IdRetryWeakGen
        from tools.tools import fix_seed

        if config.retry_type == "strong":
            self.IdGenClass = IdRetryGen
        else:
            self.IdGenClass = IdRetryWeakGen

        self.fix_seed = fix_seed
        self.rng = random.Random(self.seed)
        self._global_seq_idx = 0

        # Distributed slicing
        self._seqs_per_step = batch_size
        q, r = divmod(self._seqs_per_step, world_size)
        self._mine = q + (1 if rank < r else 0)
        self._start_off = rank * q + min(rank, r)

        self._ava_hash_pool = list(range(23))

    def _generate_one(self) -> list[int]:
        """Generate a single problem with retry/corrections."""
        gen_seed = self.seed + self._global_seq_idx
        self.fix_seed(gen_seed)

        id_gen = self.IdGenClass(
            max_op=self.config.max_op,
            max_edge=self.config.max_edge,
            perm_level=self.config.perm_level,
            detail_level=self.config.detail_level,
            retry_rate=self.config.retry_rate,
        )

        id_gen.gen_prob(self._ava_hash_pool, p_format="pq")
        id_gen.insert_retry()

        # token_id already has retries inserted
        token_ids = id_gen.token_id

        self._global_seq_idx += 1
        return token_ids

    def _pad_or_truncate(self, seq: list[int]) -> list[int]:
        """Pad or truncate sequence to seq_len + 1."""
        if len(seq) >= self.seq_len + 1:
            return seq[: self.seq_len + 1]
        else:
            return seq + [self.config.eos_token] * (self.seq_len + 1 - len(seq))

    @torch.no_grad()
    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get next batch of (input_ids, labels)."""
        seqs: list[list[int]] = []

        for i in range(self._seqs_per_step):
            if self._start_off <= i < self._start_off + self._mine:
                seq = self._generate_one()
                seq = self._pad_or_truncate(seq)
                seqs.append(seq)
            else:
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


# Difficulty presets
DIFFICULTY_PRESETS = {
    "easy": {"max_op": 10, "max_edge": 15},
    "med": {"max_op": 15, "max_edge": 20},
    "hard": {"max_op": 21, "max_edge": 28},
}


def build_igsm_retry_loader(
    difficulty: Literal["easy", "med", "hard"] = "med",
    retry_rate: float = 0.1,
    retry_type: Literal["strong", "weak"] = "strong",
    seq_len: int = 512,
    batch_size: int = 4,
    rank: int = 0,
    world_size: int = 1,
    device: str | torch.device = "cuda",
    seed: int = 42,
    mode: Literal["train", "val"] = "train",
    **kwargs: Any,
) -> iGSMRetryDataLoader:
    """Build an iGSM retry data loader.

    Args:
        difficulty: "easy", "med", or "hard"
        retry_rate: Probability of inserting a retry (0.0 to 1.0)
        retry_type: "strong" (stricter) or "weak" (simpler, nearly as good)
        seq_len: Sequence length
        batch_size: Batch size
        rank: Current process rank
        world_size: Total number of processes
        device: Device for output tensors
        seed: Random seed
        mode: "train" or "val"
        **kwargs: Additional args for iGSMRetryConfig

    Returns:
        iGSMRetryDataLoader instance
    """
    if difficulty not in DIFFICULTY_PRESETS:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    preset = DIFFICULTY_PRESETS[difficulty].copy()
    preset.update(kwargs)

    config = iGSMRetryConfig(
        retry_rate=retry_rate,
        retry_type=retry_type,
        **preset,
    )

    return iGSMRetryDataLoader(
        config=config,
        seq_len=seq_len,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        device=device,
        seed=seed,
        mode=mode,
    )


def count_retries(token_ids: list[int]) -> int:
    """Count number of retry tokens in a sequence."""
    try:
        retry_tokens = _get_retry_keyword_tokens()
        count = 0
        for i in range(len(token_ids) - len(retry_tokens) + 1):
            if token_ids[i : i + len(retry_tokens)] == retry_tokens:
                count += 1
        return count
    except Exception:
        # If we can't get retry tokens, return 0
        return 0
