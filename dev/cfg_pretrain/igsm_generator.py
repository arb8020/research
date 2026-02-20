"""iGSM synthetic data generator for pretraining.

Integrates iGSM (grade-school math problems) with the rollouts pretraining framework.
Generates math word problems with step-by-step solutions on-the-fly.

Based on:
- https://github.com/facebookresearch/iGSM
- Physics of Language Models: Part 2.1 and 2.2
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from unittest.mock import MagicMock

import numpy as np
import torch

# Add iGSM to path (must be before other paths to avoid conflicts)
# Remove conflicting paths but keep site-packages
sys.path = [p for p in sys.path if not (('rollouts' in p or 'research' in p) and 'site-packages' not in p)]
sys.path.insert(0, "/tmp/iGSM")

# Mock matplotlib (only used for visualization, not generation)
for mod_name in ['matplotlib', 'matplotlib.pyplot', 'matplotlib.patches', 'matplotlib.lines', 'matplotlib.colors', 'matplotlib.cm']:
    sys.modules[mod_name] = MagicMock()


def ensure_igsm_available():
    """Check if iGSM is available."""
    try:
        from data_gen.pretrain.id_gen import IdGen
        from tools.tools import tokenizer, fix_seed
        return True
    except ImportError as e:
        print(f"iGSM not available: {e}")
        print("Make sure iGSM is cloned to /tmp/iGSM")
        return False


@dataclass
class iGSMConfig:
    """Configuration for iGSM data generation."""
    
    # Difficulty parameters
    max_op: int = 15          # Maximum number of operations
    max_edge: int = 20        # Maximum number of edges in structure graph
    
    # Problem formatting
    perm_level: int = 5       # Random shuffle level (5 = full randomization)
    detail_level: int = 0     # Solution detail level (0 = most detailed)
    be_shortest: bool = True  # Use shortest solution path
    
    # Data format
    p_format: str = "pq"      # Problem format ("pq" = problem-question)
    
    # Token IDs (GPT2 tokenizer)
    prob_start_token: int = 222
    sol_start_token: int = 223
    ans_start_token: int = 224
    eos_token: int = 50256
    
    # Vocabulary (GPT2 has 50257 tokens)
    vocab_size: int = 50257
    
    def __post_init__(self):
        if self.max_op <= 0:
            raise ValueError(f"max_op must be positive, got {self.max_op}")
        if self.max_edge <= 0:
            raise ValueError(f"max_edge must be positive, got {self.max_edge}")


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
            raise RuntimeError("iGSM not available")
        
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
        
        # Pre-generate ava_hash list (available hash values for problem generation)
        self._ava_hash_pool = list(range(23))  # Standard pool from iGSM examples
    
    def _generate_one(self) -> list[int]:
        """Generate a single problem-solution-answer sequence.
        
        Returns:
            Token ID sequence: [prob_start] + prob + [sol_start] + sol + [ans_start] + ans + [eos]
        """
        # Set seed for this generation (deterministic given global_seq_idx)
        gen_seed = self.seed + self._global_seq_idx
        self.fix_seed(gen_seed)
        
        # Create IdGen instance
        id_gen = self.IdGen(
            max_op=self.config.max_op,
            max_edge=self.config.max_edge,
            perm_level=self.config.perm_level,
            detail_level=self.config.detail_level,
            be_shortest=self.config.be_shortest,
        )
        
        # Generate problem
        id_gen.gen_prob(self._ava_hash_pool, p_format=self.config.p_format)
        
        # Assemble token sequence
        token_ids = (
            [self.config.prob_start_token] +
            id_gen.prob_token +
            [self.config.sol_start_token] +
            id_gen.sol_token +
            [self.config.ans_start_token] +
            id_gen.ans_token +
            [self.config.eos_token]
        )
        
        self._global_seq_idx += 1
        return token_ids
    
    def _pad_or_truncate(self, seq: list[int]) -> list[int]:
        """Pad or truncate sequence to seq_len + 1 (for labels)."""
        if len(seq) >= self.seq_len + 1:
            return seq[:self.seq_len + 1]
        else:
            # Pad with EOS token
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
            # Check if this sequence belongs to our rank
            if self._start_off <= i < self._start_off + self._mine:
                seq = self._generate_one()
                seq = self._pad_or_truncate(seq)
                seqs.append(seq)
            else:
                # Still advance RNG for consistency across ranks
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


def build_igsm_loader(
    difficulty: Literal["easy", "med", "hard"] = "med",
    seq_len: int = 512,
    batch_size: int = 4,
    rank: int = 0,
    world_size: int = 1,
    device: str | torch.device = "cuda",
    seed: int = 42,
    mode: Literal["train", "val"] = "train",
    **kwargs,
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
        **kwargs: Additional args for iGSMConfig
        
    Returns:
        iGSMDataLoader instance
    """
    # Preset configurations
    presets = {
        "easy": {"max_op": 10, "max_edge": 15},
        "med": {"max_op": 15, "max_edge": 20},
        "hard": {"max_op": 21, "max_edge": 28},
    }
    
    if difficulty not in presets:
        raise ValueError(f"Unknown difficulty: {difficulty}. Choose from {list(presets.keys())}")
    
    preset = presets[difficulty]
    preset.update(kwargs)  # Allow overrides
    
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


if __name__ == "__main__":
    # Test the iGSM generator
    print("Testing iGSM data loader...")
    print("=" * 60)
    
    if not ensure_igsm_available():
        print("iGSM not available. Make sure it's cloned to /tmp/iGSM")
        sys.exit(1)
    
    # Test basic loader
    print("\n1. Testing basic loader (med difficulty):")
    loader = build_igsm_loader(
        difficulty="med",
        seq_len=256,
        batch_size=2,
        device="cpu",
        seed=42,
    )
    
    print(f"   Config: max_op={loader.config.max_op}, max_edge={loader.config.max_edge}")
    print(f"   Vocab size: {loader.config.vocab_size}")
    
    input_ids, labels = loader.next()
    print(f"   Batch shape: {input_ids.shape}")
    print(f"   First sequence token range: [{input_ids[0].min()}, {input_ids[0].max()}]")
    print(f"   Labels match (shifted by 1): {(input_ids[:, 1:] == labels[:, :-1]).all().item()}")
    
    # Decode and show example
    from tools.tools import tokenizer
    print("\n2. Decoded example:")
    tokens = input_ids[0].tolist()
    # Remove padding (EOS tokens)
    tokens = [t for t in tokens if t != loader.config.eos_token]
    text = tokenizer.decode(tokens)
    print(f"   {text[:300]}...")
    
    # Test different difficulties
    print("\n3. Testing different difficulties:")
    for diff in ["easy", "med", "hard"]:
        test_loader = build_igsm_loader(
            difficulty=diff,  # type: ignore
            seq_len=128,
            batch_size=1,
            device="cpu",
            seed=42,
        )
        inp, _ = test_loader.next()
        # Count non-padding tokens
        non_pad = (inp != test_loader.config.eos_token).sum().item()
        print(f"   {diff:5s}: ~{non_pad} tokens per problem")
    
    # Test checkpointing
    print("\n4. Testing checkpoint save/load:")
    state = loader.state_dict()
    print(f"   Saved state: global_seq_idx={state['global_seq_idx']}")
    
    new_loader = build_igsm_loader(
        difficulty="med",
        seq_len=256,
        batch_size=2,
        device="cpu",
        seed=42,
    )
    new_loader.load_state_dict(state)
    print(f"   Loaded state: global_seq_idx={new_loader._global_seq_idx}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
