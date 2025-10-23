# Training Scripts Directory

This directory is for your evolving training scripts as you progress through the milestones.

## Suggested Structure

As you learn each milestone, implement and test here:

```
train/
├── train_gpt2_baseline.py      # Your baseline implementation
├── train_gpt2_muon.py          # After studying milestone 1 (Muon)
├── train_gpt2_modern_arch.py   # After studying milestone 2 (RoPE, QK-norm, etc.)
├── train_gpt2_distributed.py   # After studying milestone 3 (Distributed Muon)
└── ...                         # Continue as you learn
```

## Workflow

1. **Study** a milestone from `history/XX_name/`
2. **Implement** the changes in a new file here (e.g., `train_gpt2_muon.py`)
3. **Test** your implementation (compare to reference in `history/`)
4. **Move to next** milestone

## Alternative: Single File Evolution

Or keep one file that evolves:

```
train/
└── train_gpt2.py  # Your implementation, updated as you learn
```

## Data Location

The training scripts should point to data in:
- `../data/fineweb10B/*.bin` (after running `python ../data/cached_fineweb10B.py 9`)
