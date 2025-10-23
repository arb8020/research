# Nano-Inference Learning Path

A pedagogical study of how modded-nanogpt achieved 19x speedup (45min → 2.3min) through systematic optimization.

## Quick Start for LLM Agents

Extract code snapshots for study:
```bash
SOURCE_REPO="/Users/chiraagbalu/research/examples/modded-nanogpt/kellerjordan_original/modded-nanogpt"
TARGET_DIR="/Users/chiraagbalu/research/examples/nano-inference"

# Example: Extract milestone 1 (Muon)
cd $SOURCE_REPO && cp -r records/101024_Muon/ $TARGET_DIR/history/01_muon/
```

## Learning Milestones

| # | Speedup | Record Directory | Target | Key Optimization |
|---|---------|------------------|--------|------------------|
| 0 | 45min | `4a452a2` (commit) | `history/4a452a2/` ✓ | Baseline (llm.c + improvements) |
| 1 | 24.9min | `records/101024_Muon/` | `history/01_muon/` | Newton-Schulz optimizer |
| 2 | 15.2min | `records/101424_ModernArch/` | `history/02_modern_arch/` | RoPE, QK-norm, ReLU² |
| 3 | 13.1min | `records/101724_DistributedMuon/` | `history/03_distributed_muon/` | Parallel Newton-Schulz |
| 4 | 8.2min | `records/110624_ShortcutsTweaks/` | `history/04_skip_connections/` | U-net, value embeddings |
| 5 | 5.0min | `records/111924_FlexAttention/` | `history/05_flex_attention/` | 64K sliding window |
| 6 | 2.7min | `records/090325_FA3/` | `history/06_flash_attn3/` | Flash Attention 3 |
| 7 | 2.6min | `records/052425_FasterReduce/` | `history/07_gradient_comm/` | Gradient overlap |
| 8 | 2.5min | `records/091525_AsyncDataLoadAttnFinalWindow/` | `history/08_async_data/` | Async data loading |
| 9 | 2.3min | Current `train_gpt.py` | `history/09_current/` | All optimizations |

## Study Order

**Phase 1: Algorithmic Foundations (45min → 13min)**
1. Muon optimizer - Biggest single win
2. Modern architecture - RoPE, QK-norm, ReLU²
3. Distributed Muon - Multi-GPU optimization

**Phase 2: Attention Innovations (13min → 5min)**
4. Skip connections - U-net patterns
5. FlexAttention - Long context efficiency

**Phase 3: Systems Engineering (5min → 2.3min)**
6. Flash Attention 3 - Kernel optimization
7. Gradient communication - Overlap compute/comm
8. Async data loading - I/O efficiency
9. Final state - All tricks combined

## Repository Structure

```
nano-inference/
├── README.md                    # This file
├── data/                        # Data download/preprocessing scripts
│   ├── README.md               # Data setup instructions
│   ├── cached_fineweb10B.py    # Fast: download pre-tokenized data
│   ├── fineweb.py              # Slow: tokenize from scratch
│   └── fineweb10B/             # Created after running download script
├── train/                       # Your training script implementations
│   ├── README.md               # Training directory guide
│   └── train_gpt2.py           # Your evolving implementation
└── history/                     # Reference snapshots at each milestone
    ├── 4a452a2/                # ✓ Baseline (already extracted)
    ├── 01_muon/                # Milestone 1
    ├── 02_modern_arch/         # Milestone 2
    └── ...
```

## Key Concepts by Milestone

### 0. Baseline
- Standard transformer (GPT-2 architecture)
- AdamW optimizer
- Learned positional embeddings
- Already optimized: RMSNorm, unit norm gradients

### 1. Muon Optimizer (45min → 24.9min)
- Newton-Schulz5 iteration replaces SVD
- 2D params use Muon, 1D params use AdamW
- Quintic polynomial coefficients: (3.4445, -4.7750, 2.0315)
- Orthogonalization for better conditioning

### 2. Modern Architecture (24.9min → 15.2min)
- **RoPE**: Rotary position embeddings
- **QK-norm**: Normalize queries/keys in attention
- **ReLU²**: Squared ReLU activation
- **Zero-init**: Initialize projections to zero

### 3. Distributed Muon (15.2min → 13.1min)
- Parallelize Newton-Schulz across GPUs
- Reduce optimizer communication overhead

### 4. Skip Connections (13.1min → 8.2min)
- Untie embedding from head
- U-net skip patterns between blocks
- Value embeddings in attention
- Better gradient flow

### 5. FlexAttention (8.2min → 5.0min)
- 64K context (up from 1024)
- Sliding window attention
- Window warmup schedule
- YaRN for context scaling

### 6. Flash Attention 3 (→ 2.7min)
- Memory-efficient attention kernel
- IO-aware tiling algorithm
- 2048 max document length

### 7. Gradient Communication (→ 2.6min)
- Overlap computation with communication
- reduce_scatter vs all_reduce
- Better gradient bucketing

### 8. Async Data Loading (→ 2.5min)
- Non-blocking batch fetching
- EoS token alignment
- Reduce data loading overhead

### 9. Current State (→ 2.3min)
- Backout mechanism (disable early layers)
- All previous optimizations
- Many smaller tricks

## Data Setup

The codebase uses FineWeb-10B dataset:
- **Format**: Pre-tokenized binary files (.bin)
- **Download**: `python data/cached_fineweb10B.py 9` (1.8 GB for 9 chunks)
- **Full dataset**: 103 chunks (~20 GB)
- **Speedrun uses**: ~900M tokens (first 9 chunks)

## References

- Original repo: https://github.com/KellerJordan/modded-nanogpt
- Muon optimizer: https://kellerjordan.github.io/posts/muon/
- llm.c baseline: https://github.com/karpathy/llm.c

## Progress

- [x] Extract baseline commit (4a452a2)
- [ ] Extract Muon optimizer
- [ ] Extract modern architecture
- [ ] Extract distributed Muon
- [ ] Extract skip connections
- [ ] Extract FlexAttention
- [ ] Extract Flash Attention 3
- [ ] Extract gradient communication
- [ ] Extract async data loading
- [ ] Extract current state

---

Last updated: 2025-10-22
