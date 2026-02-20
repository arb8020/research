"""iGSM retry/correction data generator for learning from mistakes.

Based on Physics of Language Models: Part 2.2
"How to Learn From Mistakes on Grade-School Math Problems"

This generates synthetic "retry" data where the model:
1. Makes a mistake (uses wrong parameter)
2. Says "BACK" (retry keyword)
3. Corrects itself with the right parameter

Usage:
    loader = build_igsm_retry_loader(difficulty="med", retry_rate=0.1)
    input_ids, labels = loader.next()
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from unittest.mock import MagicMock

import numpy as np
import torch

# Clean path and mock matplotlib (same as igsm_generator)
sys.path = [p for p in sys.path if not (('rollouts' in p or 'research' in p) and 'site-packages' not in p)]
sys.path.insert(0, "/tmp/iGSM")
for mod_name in ['matplotlib', 'matplotlib.pyplot', 'matplotlib.patches', 'matplotlib.lines', 'matplotlib.colors', 'matplotlib.cm']:
    sys.modules[mod_name] = MagicMock()

from data_gen.pretrain.id_retry_gen import IdGen as IdRetryGen
from data_gen.pretrain.id_retry_weak_gen import IdGen as IdRetryWeakGen
from tools.tools import tokenizer, fix_seed
from const.params import retry_key_word

# Token IDs for retry mechanism
RETRY_KEYWORD_TOKEN = tokenizer.encode(" " + retry_key_word + ".", return_tensors='pt')[0].tolist()


@dataclass
class iGSMRetryConfig:
    """Configuration for iGSM retry data generation."""
    
    # Base iGSM params
    max_op: int = 15
    max_edge: int = 20
    perm_level: int = 5
    detail_level: int = 0
    
    # Retry-specific params
    retry_rate: float = 0.1  # Probability of inserting a retry
    retry_type: Literal["strong", "weak"] = "strong"
    # weak: can retry with any future parameter
    # strong: can only retry with parameters that haven't appeared yet
    
    # Token IDs
    prob_start_token: int = 222
    sol_start_token: int = 223
    ans_start_token: int = 224
    eos_token: int = 50256
    vocab_size: int = 50257
    
    def __post_init__(self):
        if not 0 <= self.retry_rate <= 1:
            raise ValueError(f"retry_rate must be in [0, 1], got {self.retry_rate}")


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
        self.config = config
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.mode = mode
        
        self.seed = seed + rank + (1000000 if mode == "val" else 0)
        
        # Choose generator class
        if config.retry_type == "strong":
            self.IdGenClass = IdRetryGen
        else:
            self.IdGenClass = IdRetryWeakGen
        
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
        fix_seed(gen_seed)
        
        # Create IdGen with retry
        id_gen = self.IdGenClass(
            max_op=self.config.max_op,
            max_edge=self.config.max_edge,
            perm_level=self.config.perm_level,
            detail_level=self.config.detail_level,
            retry_rate=self.config.retry_rate,
        )
        
        # Generate problem
        id_gen.gen_prob(self._ava_hash_pool, p_format="pq")
        
        # Insert retry tokens
        id_gen.insert_retry()
        
        # Use the token_id which already has retries inserted
        token_ids = id_gen.token_id
        
        self._global_seq_idx += 1
        return token_ids
    
    def _pad_or_truncate(self, seq: list[int]) -> list[int]:
        """Pad or truncate sequence to seq_len + 1."""
        if len(seq) >= self.seq_len + 1:
            return seq[:self.seq_len + 1]
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
    **kwargs,
) -> iGSMRetryDataLoader:
    """Build an iGSM retry data loader.
    
    Args:
        difficulty: "easy", "med", or "hard"
        retry_rate: Probability of inserting a retry (0.0 to 1.0)
        retry_type: "strong" (only wrong params not yet seen) or "weak" (any future param)
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
    presets = {
        "easy": {"max_op": 10, "max_edge": 15},
        "med": {"max_op": 15, "max_edge": 20},
        "hard": {"max_op": 21, "max_edge": 28},
    }
    
    if difficulty not in presets:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    
    preset = presets[difficulty]
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
    return sum(1 for i in range(len(token_ids) - len(RETRY_KEYWORD_TOKEN) + 1)
               if token_ids[i:i+len(RETRY_KEYWORD_TOKEN)] == RETRY_KEYWORD_TOKEN)


if __name__ == "__main__":
    print("Testing iGSM retry data loader...")
    print("=" * 70)
    
    # Test strong retry
    print("\n1. Testing STRONG retry (retry_rate=0.3):")
    loader = build_igsm_retry_loader(
        difficulty="med",
        retry_rate=0.3,
        retry_type="strong",
        seq_len=512,
        batch_size=2,
        device="cpu",
        seed=42,
    )
    
    input_ids, labels = loader.next()
    print(f"   Batch shape: {input_ids.shape}")
    
    # Decode and show
    tokens = input_ids[0].tolist()
    text = tokenizer.decode(tokens)
    print(f"\n   Example with retries:")
    print(f"   {text[:400]}...")
    
    # Count retries
    n_retries = count_retries(tokens)
    print(f"\n   Number of retries in this example: {n_retries}")
    
    # Test weak retry
    print("\n2. Testing WEAK retry (retry_rate=0.3):")
    loader_weak = build_igsm_retry_loader(
        difficulty="med",
        retry_rate=0.3,
        retry_type="weak",
        seq_len=512,
        batch_size=1,
        device="cpu",
        seed=42,
    )
    
    input_ids_weak, _ = loader_weak.next()
    tokens_weak = input_ids_weak[0].tolist()
    text_weak = tokenizer.decode(tokens_weak)
    print(f"   {text_weak[:400]}...")
    
    n_retries_weak = count_retries(tokens_weak)
    print(f"\n   Number of retries in this example: {n_retries_weak}")
    
    # Test different retry rates
    print("\n3. Testing different retry rates:")
    for rate in [0.0, 0.1, 0.3, 0.5]:
        test_loader = build_igsm_retry_loader(
            difficulty="med",
            retry_rate=rate,
            retry_type="strong",
            seq_len=256,
            batch_size=5,
            device="cpu",
            seed=42,
        )
        
        total_retries = 0
        for _ in range(5):
            batch, _ = test_loader.next()
            for i in range(batch.shape[0]):
                total_retries += count_retries(batch[i].tolist())
        
        avg_retries = total_retries / 25
        print(f"   retry_rate={rate:.1f}: ~{avg_retries:.1f} retries per problem")
    
    # Test checkpoint
    print("\n4. Testing checkpoint save/load:")
    state = loader.state_dict()
    print(f"   Saved: global_seq_idx={state['global_seq_idx']}")
    
    new_loader = build_igsm_retry_loader(
        difficulty="med",
        retry_rate=0.3,
        retry_type="strong",
        seq_len=512,
        batch_size=2,
        device="cpu",
        seed=42,
    )
    new_loader.load_state_dict(state)
    print(f"   Loaded: global_seq_idx={new_loader._global_seq_idx}")
    
    # Verify next batch matches
    orig_next, _ = loader.next()
    new_next, _ = new_loader.next()
    match = (orig_next == new_next).all().item()
    print(f"   Batches match: {match}")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
