# Quick Start: Multi-GPU SFT with FSDP

## TL;DR - You're Ready to Run!

```bash
cd dev/integration_training
python train.py configs/02_debug_sft_fsdp.py
```

That's it. train.py auto-detects FSDP and relaunches with torchrun.

---

## What You Have

✅ **FSDPTrainingBackend** - Full implementation in `rollouts/rollouts/training/backends/fsdp.py`
✅ **train.py** - Auto-detects FSDP and handles torchrun
✅ **Config** - `configs/02_debug_sft_fsdp.py` ready to use
✅ **deploy.py** - Automatically uses torchrun for remote FSDP

---

## How It Works

1. You run: `python train.py configs/02_debug_sft_fsdp.py`
2. train.py detects `train_backend="fsdp"` + `len(gpu_ranks) > 1`
3. Auto-relaunches itself with `torchrun --nproc_per_node=4`
4. **torchrun** spawns 4 processes (one per GPU)
5. Each process runs `train.py` from the beginning (now with RANK env var set)
6. Calls `create_fsdp_backend()` which:
   - Initializes `torch.distributed` (if not already done)
   - Creates FSDPTrainingBackend with your model
   - FSDP wraps model and shards it across GPUs
7. Training proceeds - FSDP handles all communication via NCCL

---

## Local Testing

```bash
cd dev/integration_training

# Standard pattern - just run train.py!
python train.py configs/02_debug_sft_fsdp.py

# FSDP is auto-detected, torchrun is auto-launched
# Uses gpu_ranks from config (e.g., [0,1,2,3])
```

**Single GPU (no FSDP):**
```bash
# Edit config to have gpu_ranks=[0] or train_backend="pytorch"
python train.py configs/01_debug_sft.py
```

---

## Remote Deployment

```bash
python deploy.py configs/02_debug_sft_fsdp.py --ssh root@host:port
```

deploy.py automatically:
- Detects `train_backend="fsdp"`
- Uses `torchrun --nproc_per_node=4` instead of `python`
- Sets `CUDA_VISIBLE_DEVICES` to your `gpu_ranks`

---

## What We DON'T Use (Yet)

❌ **miniray** - Only for multi-node (future)
❌ **Worker pattern** - torchrun handles process spawning
❌ **Service pattern** - Deleted (incomplete)

**Why?** torchrun is battle-tested and handles single-node multi-GPU perfectly.

---

## Config Structure

```python
config = Config(
    target=TargetConfig(
        gpu_ranks=[0, 1, 2, 3],  # Which GPUs to use
        train_backend="fsdp",     # Use FSDP backend
    ),
    # ... rest of config
)
```

---

## Next Steps

1. **Test it**: `python train.py configs/02_debug_sft_fsdp.py`
2. **Add data**: Point config at your actual SFT dataset
3. **Scale up**: Increase batch size, steps, etc.
4. **Multi-node (later)**: When you need 2+ machines, THEN add miniray

---

## Troubleshooting

**"torch.distributed not initialized"**
- Don't worry - train.py auto-launches with torchrun
- If still fails, check torchrun is installed

**"NCCL error"**
- Check CUDA_VISIBLE_DEVICES matches your gpu_ranks
- Make sure all GPUs are free (no other processes)

**"Model doesn't fit on GPU"**
- FSDP shards the model, but each GPU still needs to hold its shard
- Try smaller model or more GPUs

---

## Architecture Docs

See `rollouts/docs/MULTI_GPU_DESIGN.md` for full design rationale.
