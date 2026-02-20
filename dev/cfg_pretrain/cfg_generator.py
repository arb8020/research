"""CFG-based synthetic data generator for pretraining.

Integrates PhysicsLM4's Lano-cfg with the rollouts pretraining framework.
Generates sequences from Context-Free Grammars for language model pretraining.

Based on:
- PhysicsLM4/data-synthetic-pretrain/Lano-cfg/data_cfg.py
- rollouts/pretrain/dataloader.py
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class Cursor:
    """Position in a multi-shard dataset."""
    file_idx: int = 0
    pos_in_file: int = 0
    wrap_count: int = 0


class CFGNode:
    """A node in the CFG tree."""
    rng: random.Random | None = None
    nt_counter: list[int] = [0] * 15
    
    def __init__(self, config: CFGConfig, depth: int, id: int, name: str):
        self.config = config
        self.depth = depth
        self.id = id
        self.name = name
        self.children: list[list[int]] | None = None
    
    def generate_leaf(self) -> list[int]:
        """Generate a terminal symbol."""
        if hasattr(self.config, 'multi_vocab'):
            # real T symbol ids start from vocab_size+4
            choice = self.config.content_map[self.id - 3 - self.config.vocab_size - 1]
            if 'NU1' in self.config.multi_vocab:
                return [random.choices(choice, range(1, len(choice) + 1), k=1)[0], self.id]
            else:
                return [choice[random.randint(0, len(choice) - 1)], self.id]
        return [self.id]
    
    def generate(self, parents: list[int] = []) -> list[list[int]]:
        """Generate a sequence from this node."""
        rng = CFGNode.rng
        if self.depth == 0:
            CFGNode.nt_counter = [1] * 15
        CFGNode.nt_counter[self.depth] += 1
        
        if self.children is None:
            return [self.generate_leaf() + [-CFGNode.nt_counter[self.depth]] + parents]
        
        cc = rng.randint(0, len(self.children) - 1)
        res = []
        for c in self.children[cc]:
            res += self.config.all[self.depth + 1][c].generate(
                [self.id, -CFGNode.nt_counter[self.depth]] + parents
            )
        return res


@dataclass
class CFGConfig:
    """Configuration for a Context-Free Grammar."""
    depth: int = 7
    num_sym: int = 30  # number of T symbols
    vocab_size: int = 30
    deg_min: int = 2
    deg_max: int = 3
    len_min: int = 1
    len_max: int = 2
    disallow_duplicate_sym: bool = False
    disallow_duplicate_seq: bool = False
    num_sym_mode: int = 1
    eos_token: int = 0
    mask_token: int = 0
    sep_token: int = 0
    
    # Internal structures
    sizes: list[int] = field(default_factory=list)
    idx: list[list[int]] = field(default_factory=list)
    all: list[list[CFGNode]] = field(default_factory=list)
    count: int = 0
    
    def __post_init__(self):
        if not self.sizes:
            self._build_graph()
    
    def _build_graph(self):
        """Build the CFG graph structure."""
        self.sizes = [0] * (self.depth + 1)
        self.sizes[0] = 1
        
        if isinstance(self.num_sym_mode, list):
            assert len(self.num_sym_mode) == self.depth + 1
            self.sizes = self.num_sym_mode
        elif self.num_sym_mode in [1, 3]:
            for i in range(1, self.depth + 1):
                self.sizes[i] = max(1, self.num_sym * i // self.depth)
        elif self.num_sym_mode == 2:
            for i in range(1, self.depth + 1):
                self.sizes[i] = self.num_sym
        
        assert self.sizes[-1] == self.num_sym, f"Expected {self.num_sym} symbols at last layer, got {self.sizes[-1]}"
        
        self.idx = [[None for _ in range(p)] for p in self.sizes]
        names = [[None for _ in range(p)] for p in self.sizes]
        self.all = [[None for _ in range(p)] for p in self.sizes]
        
        count = self.vocab_size + 3  # skip three numbers for <SEP> <MASK> and future use
        
        for i in range(self.depth, -1, -1):
            for j in range(self.sizes[i]):
                count += 1
                self.idx[i][j] = count
                names[i][j] = self._col_to_name(count + 25)
        
        self.count = count
        
        for i in range(self.depth, -1, -1):
            depth_hash = []
            for j in range(self.sizes[i]):
                self.all[i][j] = CFGNode(self, depth=i, id=self.idx[i][j], name=names[i][j])
                cur = self.all[i][j]
                if i != self.depth:
                    degree = random.randint(self.deg_min, self.deg_max)
                    cur.children = []
                    for k in range(degree):
                        llen = random.randint(self.len_min, self.len_max)
                        child = []
                        child_n = ""
                        for l in range(llen):
                            nextid = random.randint(0, self.sizes[i + 1] - 1)
                            while self.disallow_duplicate_sym and nextid in child:
                                nextid = random.randint(0, self.sizes[i + 1] - 1)
                            child += [nextid]
                            child_n += "|" + str(nextid)
                        while self.disallow_duplicate_seq and child_n in depth_hash:
                            llen = random.randint(self.len_min, self.len_max)
                            child = []
                            child_n = ""
                            for l in range(llen):
                                nextid = random.randint(0, self.sizes[i + 1] - 1)
                                while self.disallow_duplicate_sym and nextid in child:
                                    nextid = random.randint(0, self.sizes[i + 1] - 1)
                                child += [nextid]
                                child_n += "|" + str(nextid)
                        cur.children += [child]
                        depth_hash += [child_n]
    
    def _col_to_name(self, col: int) -> str:
        """Convert column number to Excel-style name."""
        result = ""
        while col > 0:
            col, remainder = divmod(col - 1, 26)
            result = chr(65 + remainder) + result
        return result
    
    def generate_sequence(self, rng: random.Random) -> list[int]:
        """Generate a sequence of terminal symbols."""
        CFGNode.rng = rng
        output = self.all[0][0].generate()
        return [a[0] for a in output]
    
    @staticmethod
    def from_graph(file: str | Path) -> CFGConfig:
        """Load CFG from JSON file."""
        with open(file, 'r') as f:
            data = json.load(f)
        
        config = CFGConfig(
            depth=data['depth'],
            num_sym=data['num_sym'],
            vocab_size=data.get('vocab_size', data['num_sym']),
            deg_min=data.get('deg_min', 2),
            deg_max=data.get('deg_max', 3),
            len_min=data.get('len_min', 1),
            len_max=data.get('len_max', 2),
            disallow_duplicate_sym=data.get('disallow_duplicate_sym', False),
            disallow_duplicate_seq=data.get('disallow_duplicate_seq', False),
            num_sym_mode=data.get('num_sym_mode', 1),
            eos_token=data.get('eos_token', 0),
            mask_token=data.get('mask_token', 0),
            sep_token=data.get('sep_token', 0),
            sizes=data['sizes'],
            idx=data['idx'],
            count=data['count'],
        )
        
        # Reconstruct nodes
        config.all = [[None for _ in range(config.sizes[p])] for p in range(config.depth + 1)]
        for i in range(config.depth, -1, -1):
            for j in range(config.sizes[i]):
                config.all[i][j] = CFGNode(config, depth=i, id=config.idx[i][j], name=data['all'][i][j]['name'])
                if i != config.depth:
                    config.all[i][j].children = data['all'][i][j]['children']
        
        return config
    
    def save_graph(self, file: str | Path):
        """Save CFG to JSON file."""
        data = {
            'depth': self.depth,
            'num_sym': self.num_sym,
            'vocab_size': self.vocab_size,
            'deg_min': self.deg_min,
            'deg_max': self.deg_max,
            'len_min': self.len_min,
            'len_max': self.len_max,
            'disallow_duplicate_sym': self.disallow_duplicate_sym,
            'disallow_duplicate_seq': self.disallow_duplicate_seq,
            'num_sym_mode': self.num_sym_mode,
            'eos_token': self.eos_token,
            'mask_token': self.mask_token,
            'sep_token': self.sep_token,
            'sizes': self.sizes,
            'idx': self.idx,
            'count': self.count,
            'all': [[{'depth': node.depth, 'id': node.id, 'name': node.name, 
                     'children': node.children} for node in layer] for layer in self.all]
        }
        with open(file, 'w') as f:
            json.dump(data, f, indent=2)


class CFGDataLoader:
    """Data loader that generates sequences from a CFG on-the-fly.
    
    Compatible with rollouts/pretrain/dataloader.py interface.
    """
    
    def __init__(
        self,
        config: CFGConfig,
        seq_len: int,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        device: str | torch.device = "cuda",
        seed: int = 42,
    ):
        self.config = config
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.seed = seed
        
        # Each rank gets its own RNG stream
        self.rng = random.Random(seed + rank)
        
        # Global sequence counter for deterministic resume
        self._global_seq_idx = 0
        
        # Compute per-step slicing for distributed training
        self._seqs_per_step = batch_size
        q, r = divmod(self._seqs_per_step, world_size)
        self._mine = q + (1 if rank < r else 0)
        self._start_off = rank * q + min(rank, r)
    
    def _generate_sequence(self) -> list[int]:
        """Generate a single sequence from the CFG."""
        return self.config.generate_sequence(self.rng)
    
    def _pad_or_truncate(self, seq: list[int]) -> list[int]:
        """Pad or truncate sequence to seq_len."""
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
                seq = self._generate_sequence()
                seq = self._pad_or_truncate(seq)
                seqs.append(seq)
            else:
                # Still advance RNG for consistency
                self._generate_sequence()
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


def build_cfg_loader(
    cfg_path: str | Path,
    seq_len: int,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    device: str | torch.device = "cuda",
    seed: int = 42,
) -> CFGDataLoader:
    """Build a CFG data loader from a config file.
    
    Args:
        cfg_path: Path to CFG JSON file
        seq_len: Sequence length
        batch_size: Batch size
        rank: Current process rank
        world_size: Total number of processes
        device: Device for output tensors
        seed: Random seed
        
    Returns:
        CFGDataLoader instance
    """
    config = CFGConfig.from_graph(cfg_path)
    return CFGDataLoader(
        config=config,
        seq_len=seq_len,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        device=device,
        seed=seed,
    )


if __name__ == "__main__":
    # Test the CFG generator
    import sys
    
    # Try to load from PhysicsLM4 configs
    cfg_path = Path("/Users/chiraagbalu/research/PhysicsLM4/data-synthetic-pretrain/Lano-cfg/configs/cfg3f.json")
    
    if not cfg_path.exists():
        print(f"Config not found at {cfg_path}")
        print("Creating a simple test CFG instead...")
        
        # Create a simple test CFG
        config = CFGConfig(
            depth=3,
            num_sym=3,
            vocab_size=3,
            deg_min=2,
            deg_max=2,
            len_min=1,
            len_max=2,
            num_sym_mode=2,
        )
        config.save_graph("test_cfg.json")
        cfg_path = "test_cfg.json"
    
    print(f"Loading CFG from {cfg_path}")
    loader = build_cfg_loader(
        cfg_path=cfg_path,
        seq_len=128,
        batch_size=4,
        device="cpu",
        seed=42,
    )
    
    print(f"CFG vocab size: {loader.config.vocab_size}")
    print(f"CFG depth: {loader.config.depth}")
    print(f"CFG num_sym: {loader.config.num_sym}")
    
    # Generate a few batches
    for i in range(3):
        input_ids, labels = loader.next()
        print(f"\nBatch {i+1}:")
        print(f"  input_ids shape: {input_ids.shape}")
        print(f"  First sequence: {input_ids[0].tolist()[:20]}...")
        print(f"  Labels match: {(input_ids[:, 1:] == labels[:, :-1]).all().item()}")
