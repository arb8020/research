# Synthetic Data Pretraining

This directory implements synthetic data generation for language model pretraining using:
- **Context-Free Grammars (CFGs)** from PhysicsLM4
- **Grade-School Math (iGSM)** from Physics of Language Models Part 2

Both integrate with the `rollouts/pretrain` framework.

## Overview

| Generator | Domain | Vocab Size | Sequence Length | Use Case |
|-----------|--------|------------|-----------------|----------|
| **CFG (Lano)** | Hierarchical structures | Small (3-9) | ~100-500 tokens | Structure learning |
| **iGSM** | Grade-school math | GPT2 (50257) | ~100-500 tokens | Math reasoning |
| **iGSM-Retry** | Math with corrections | GPT2 (50257) | ~100-800 tokens | Learning from mistakes |

## Structure

```
cfg_pretrain/
├── cfg_generator.py          # CFG data loader
├── igsm_generator.py         # iGSM data loader
├── igsm_retry_generator.py   # iGSM with retry/correction
├── train_cfg.py              # Training script for CFG
├── train_igsm.py             # Training script for iGSM
├── eval_cfg.py               # Evaluation utilities
├── demo.py                   # Demo script
├── demo_retry.py             # Retry demo
├── configs/
│   ├── cfg_tiny.py           # Tiny CFG model
│   ├── cfg_small.py          # Small CFG model
│   └── igsm_tiny.py          # Tiny iGSM model
└── README.md
```

## Quick Start

### CFG Training

```bash
# Tiny model (CPU-friendly)
python train_cfg.py configs/cfg_tiny.py

# Small model (GPU recommended)
python train_cfg.py configs/cfg_small.py

# Multi-GPU
torchrun --standalone --nproc_per_node=8 train_cfg.py configs/cfg_small.py
```

### iGSM Training

```bash
# Make sure iGSM is cloned
# git clone https://github.com/facebookresearch/iGSM.git /tmp/iGSM

# Tiny model
python train_igsm.py configs/igsm_tiny.py
```

## CFG (Context-Free Grammar)

### Available CFGs

From PhysicsLM4 (`/Users/chiraagbalu/research/PhysicsLM4/data-synthetic-pretrain/Lano-cfg/configs/`):

| Config | Depth | Vocab | Avg Seq Len | Description |
|--------|-------|-------|-------------|-------------|
| cfg3b.json | 6 | 3 | 244 | Balanced |
| cfg3f.json | 6 | 3 | 163 | Small, fast |
| cfg3g.json | 6 | 3 | 265 | Medium |
| cfg3i.json | 6 | 3 | 301 | Large |
| cfg3j.json | 7 | 3 | 533 | Very large |
| cfg3k.json | 7 | 3 | 471 | Complex |
| cfg3e1.json | 6 | 9 | 171 | Large vocab |
| cfg3e2.json | 6 | 4 | 139 | Small vocab |

### Usage

```python
from cfg_generator import build_cfg_loader

loader = build_cfg_loader(
    cfg_path="path/to/cfg3f.json",
    seq_len=512,
    batch_size=8,
    device="cuda",
)

input_ids, labels = loader.next()  # [batch, seq_len]
```

### Verify CFG Sequence

```python
from cfg_generator import CFGConfig

config = CFGConfig.from_graph("cfg3f.json")
seq = [3, 2, 2, 1, 1, 2, ...]

# Check if sequence satisfies CFG
is_valid = config.solve_dp_noneq_fast(seq, no_debug=True)
```

## iGSM (Grade-School Math)

### Difficulty Levels

| Level | max_op | max_edge | Description |
|-------|--------|----------|-------------|
| easy | 10 | 15 | Simple problems |
| med | 15 | 20 | Medium difficulty |
| hard | 21 | 28 | Complex problems |

### Usage

```python
from igsm_generator import build_igsm_loader

loader = build_igsm_loader(
    difficulty="med",  # "easy", "med", or "hard"
    seq_len=512,
    batch_size=4,
    device="cuda",
)

input_ids, labels = loader.next()
```

### Example Output

**Problem:**
```
The number of each Penguin Beach's Giraffe equals 6. The number of each 
Octopus Den's Leopard equals each Octopus Den's Giraffe. How many Animal 
does Penguin Beach have?
```

**Solution:**
```
Define Penguin Beach's Giraffe as e; so e = 6. Define Penguin Beach's 
Animal as J; so J = e = 6.
```

**Answer:** `6`

### Token Format

```
[222] + problem_tokens + [223] + solution_tokens + [224] + answer_tokens + [50256]
```

Where:
- `222` = problem start
- `223` = solution start  
- `224` = answer start
- `50256` = EOS (GPT2)

## Training Features

Both trainers support:

- **Muon optimizer** for 2D weight matrices
- **AdamW** for embeddings/norms/biases
- **torch.compile** for CUDA performance
- **Gradient accumulation**
- **Distributed training** (torchrun)
- **Checkpointing** with fingerprint verification
- **Deterministic resume** (RNG state saved)

## Creating Custom Data

### Custom CFG

```python
from cfg_generator import CFGConfig

config = CFGConfig(
    depth=6,
    num_sym=3,
    vocab_size=3,
    deg_min=2,
    deg_max=3,
)
config.save_graph("my_cfg.json")
```

### Custom iGSM

```python
from igsm_generator import iGSMConfig, iGSMDataLoader

config = iGSMConfig(
    max_op=20,
    max_edge=25,
    perm_level=5,
)

loader = iGSMDataLoader(
    config=config,
    seq_len=512,
    batch_size=4,
    device="cuda",
)
```

## Evaluation

### CFG Evaluation

```bash
# Verify a sequence
python eval_cfg.py --cfg /path/to/cfg3f.json --verify "3,2,2,1,1,2"

# Evaluate model
python eval_cfg.py --cfg /path/to/cfg3f.json --checkpoint output/step_01000.pt
```

### Demo

```bash
# Run all demos
python demo.py
```

## References

- **PhysicsLM4**: https://github.com/facebookresearch/PhysicsLM4
- **iGSM**: https://github.com/facebookresearch/iGSM
- **Physics of Language Models**:
  - Part 1: Learning Hierarchical Language Structures
  - Part 2.1: Grade-School Math and the Hidden Reasoning Process
  - Part 2.2: How to Learn From Mistakes on Grade-School Math Problems
  - Part 4.1: Architecture Design and the Magic of Canon Layers
- **rollouts/pretrain**: `/Users/chiraagbalu/research/rollouts/rollouts/pretrain/`

## iGSM-Retry (Learning from Mistakes)

From **Physics of Language Models: Part 2.2**, iGSM-Retry generates synthetic "correction pairs" where the model learns to recover from mistakes.

### How It Works

1. Model makes a mistake (uses wrong parameter)
2. Model says "BACK" (retry keyword)
3. Model corrects itself with the right parameter

**Example:**
```
❌ Define Octopus Den's Leopard as r; so r = t = 6  (WRONG!)
🔄 BACK
✅ Define Octopus Den's Leopard as r; so r = e = 21 (CORRECT!)
```

### Usage

```python
from igsm_retry_generator import build_igsm_retry_loader

loader = build_igsm_retry_loader(
    difficulty="med",
    retry_rate=0.1,        # Probability of retry (0.0 to 1.0)
    retry_type="strong",   # "strong" or "weak"
    seq_len=512,
    batch_size=4,
    device="cuda",
)

input_ids, labels = loader.next()
```

### Retry Types

- **strong**: Can only retry with parameters that haven't appeared yet (harder)
- **weak**: Can retry with any future parameter (easier)

### Demo

```bash
python demo_retry.py
```

## References

- **PhysicsLM4**: https://github.com/facebookresearch/PhysicsLM4
- **iGSM**: https://github.com/facebookresearch/iGSM
- **Physics of Language Models**:
  - Part 1: Learning Hierarchical Language Structures
  - Part 2.1: Grade-School Math and the Hidden Reasoning Process
  - Part 2.2: How to Learn From Mistakes on Grade-School Math Problems
  - Part 4.1: Architecture Design and the Magic of Canon Layers
- **rollouts/pretrain**: `/Users/chiraagbalu/research/rollouts/rollouts/pretrain/`

## TODO

- [x] iGSM retry/correction data (Part 2.2)
- [ ] Mixed data training (CFG + iGSM)
- [ ] Evaluation metrics (perplexity, accuracy, KL-divergence)
- [ ] BOS-aligned packing for document boundaries
- [ ] FP8 training support (H100+)
- [ ] W&B logging integration
