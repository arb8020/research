# CFG-Based Pretraining Implementation Summary

## Overview

This implementation integrates PhysicsLM4's Lano-cfg synthetic data generation with the rollouts pretraining framework to enable training language models on Context-Free Grammar (CFG) generated sequences.

## Files Created

### Core Implementation

1. **`cfg_generator.py`** (350 lines)
   - `CFGConfig`: Python dataclass for CFG configuration
   - `CFGNode`: Tree node structure for CFG generation
   - `CFGDataLoader`: PyTorch-compatible data loader
   - `build_cfg_loader()`: Factory function for creating loaders
   - Compatible with `rollouts/pretrain/dataloader.py` interface

2. **`train_cfg.py`** (470 lines)
   - Modified from `rollouts/pretrain/train.py`
   - Integrates CFG data loader
   - Adjusts model vocab_size dynamically based on CFG
   - Supports single-GPU and multi-GPU (torchrun) training
   - Muon + AdamW optimizer hybrid
   - Checkpointing with fingerprint verification

3. **`eval_cfg.py`** (200 lines)
   - Sequence verification using PhysicsLM4's DP solver
   - Ground-truth probability computation
   - Model evaluation metrics (loss, perplexity, accuracy, KL-divergence)

4. **`demo.py`** (170 lines)
   - 4 demos showing: basic usage, dataloader, ground-truth probs, custom CFGs

### Configuration Files

5. **`configs/cfg_tiny.py`**
   - 4.2M parameter model for fast iteration
   - Uses cfg3f.json (depth=6, vocab=3)

6. **`configs/cfg_small.py`**
   - ~30M parameter model
   - Uses cfg3k.json (more complex CFG)

### Documentation

7. **`README.md`**
   - Usage instructions
   - Architecture overview
   - Configuration examples

8. **`IMPLEMENTATION_SUMMARY.md`** (this file)
   - Technical details
   - Design decisions
   - Future work

## Key Design Decisions

### 1. Compatibility with rollouts/pretrain

The implementation maintains compatibility with the existing training framework:
- Same `TrainConfig` and `ModelConfig` dataclasses
- Same optimizer setup (Muon + AdamW)
- Same checkpointing format
- Same logging and metrics

### 2. DataLoader Interface

`CFGDataLoader` follows the same interface as `DeterministicLoader`:
```python
loader.next() -> (input_ids, labels)  # Both [batch, seq_len]
loader.state_dict() -> dict  # For checkpointing
loader.load_state_dict(dict)  # For resuming
```

### 3. Vocabulary Handling

The CFG generates terminal symbols (1, 2, 3, ...). Special tokens are:
- EOS: vocab_size + 1
- MASK: vocab_size + 2  
- SEP: vocab_size + 3

The model's vocab_size is set to `cfg.vocab_size + 4` to accommodate these.

### 4. Deterministic Generation

Each rank gets its own RNG stream seeded by `seed + rank`:
- Ensures reproducibility across runs
- Prevents duplicate sequences in distributed training
- Enables exact resume via RNG state checkpointing

### 5. Sequence Padding

CFG-generated sequences have variable length. The dataloader:
- Truncates sequences longer than `seq_len + 1`
- Pads shorter sequences with EOS tokens
- Returns `(input_ids, labels)` where labels are shifted by 1

## Usage Examples

### Training

```bash
# Tiny model on CPU (for testing)
python train_cfg.py configs/cfg_tiny.py

# Small model on GPU
python train_cfg.py configs/cfg_small.py

# Multi-GPU training
torchrun --standalone --nproc_per_node=8 train_cfg.py configs/cfg_small.py

# Resume from checkpoint
python train_cfg.py configs/cfg_small.py --resume
```

### Evaluation

```bash
# Verify a sequence
python eval_cfg.py --cfg /path/to/cfg3f.json --verify "3,2,2,1,1,2"

# Evaluate model checkpoint
python eval_cfg.py --cfg /path/to/cfg3f.json --checkpoint output/step_01000.pt
```

### Demo

```bash
# Run all demos
python demo.py
```

## Integration with PhysicsLM4

The implementation uses:
- `data_cfg.py`: Original CFG_Config class with DP solvers
- `configs/cfg3f.json`, `cfg3k.json`: Pre-built CFG configurations

Minor modifications made to `data_cfg.py`:
- Made `xlsxwriter` optional (replaced with custom `_col_to_name()` function)

## Available CFG Configs

From PhysicsLM4:
- **cfg3f.json**: Depth=6, vocab=3, ~160 tokens/sequence
- **cfg3k.json**: Depth=6, vocab=3, more complex rules
- **cfg3e1.json**, **cfg3e2.json**: Extended variants
- **cfg3b.json**, **cfg3g-i.json**: Other variants

## Metrics

Training logs:
- Loss
- Learning rate scale
- Gradient norm
- Tokens/second
- Elapsed time

Evaluation metrics:
- Perplexity
- Next-token accuracy
- KL-divergence from ground-truth CFG distribution

## Future Work

1. **Multi-CFG Mixing**: Support training on mixtures of different CFGs
2. **BOS-Aligned Packing**: Pack documents to start with BOS token
3. **FP8 Training**: Add support for H100+ FP8 training
4. **W&B Integration**: Add weights & biases logging
5. **More Evaluation**: Per-layer probing, attention analysis
6. **Custom CFG Builder**: GUI or DSL for creating CFGs
7. **Sequence Length Curriculum**: Start with short sequences, increase over time

## References

- PhysicsLM4: https://github.com/facebookresearch/PhysicsLM4
- Lano-cfg paper: (see PhysicsLM4 repo)
- Muon optimizer: Moonlight (arXiv:2502.16982), Polar Express (arXiv:2505.16932)
- rollouts/pretrain: /Users/chiraagbalu/research/rollouts/rollouts/pretrain/
