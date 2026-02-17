# Implementation TODO

> Each step is end-to-end runnable. Abstractions are guesses - we'll adjust as we learn.

## Reference Implementations

External repos to validate against:

| Repo | What to validate | Link |
|------|------------------|------|
| **llm.c** | Training loop correctness, loss curves | https://github.com/karpathy/llm.c |
| **modded-nanogpt** | Speedrun patterns, multi-GPU | https://github.com/KellerJordan/modded-nanogpt |
| **LLMs-from-scratch** | Step-by-step correctness | https://github.com/rasbt/LLMs-from-scratch |
| **torchtitan** | Production pretraining patterns | https://github.com/pytorch/torchtitan |
| **nanotron** | Distributed training, checkpoints | https://github.com/huggingface/nanotron |
| **lm-evaluation-harness** | Model quality benchmarks | https://github.com/EleutherAI/lm-evaluation-harness |

Internal references:
- `/Users/chiraagbalu/research/rollouts/rollouts/tools/functional_extractor/llama_functional.py` - Functional model parity test
- `/Users/chiraagbalu/research/nmoe/configs/gates/gates.toml` - Gate-based validation pattern
- `/Users/chiraagbalu/research/rollouts/rollouts/inference/tests/run_gpu_tests.py` - Modal sandbox testing

---

## Step 1: Train on Random Data

**Goal**: `python -m pretrain.train` runs, loss goes down.

**Files:**
- `pyproject.toml` - deps: torch, numpy
- `pretrain/__init__.py`
- `pretrain/config.py` - ModelConfig, TrainConfig (frozen dataclasses)
- `pretrain/models/llama.py` - functional forward (hardcode tiny config first)
- `pretrain/train.py` - training loop with random data

**Abstractions:**
- `forward(input_ids, weights, config) -> logits` - functional model signature
- `TrainConfig` - separates model config from training config
- Weights are `dict[str, Tensor]` not nn.Module

**Validation:**
```bash
# 1. Runs without error
python -m pretrain.train

# 2. Loss decreases (overfitting random data is fine)
# Expected: step=0 loss=~10.8 (ln(vocab_size)), step=100 loss=<5.0

# 3. Compare loss curve shape to llm.c
# Clone: git clone https://github.com/karpathy/llm.c /tmp/llm.c
# Their train_gpt2.py has similar tiny config - loss should decrease similarly
```

**Reference:** llm.c `train_gpt2.py` lines 1-200 (simple training loop)

---

## Step 2: Real Data (Single GPU)

**Goal**: Train on tokenized shards, not random data.

**Files:**
- `pretrain/data.py` - load_shard(), distributed_data_generator()
- `scripts/prep_data.py` - tokenize a small dataset to .npy

**Abstractions:**
- Generator yields `{input_ids, labels}` batches
- Shards are just .npy files (no custom format)
- Generator supports `.send()` for dynamic batch size (even if unused yet)

**Validation:**
```bash
# 1. Prep data matches tiktoken exactly
python scripts/prep_data.py --input tiny.txt --output data/
python -c "
import tiktoken, numpy as np
enc = tiktoken.get_encoding('cl100k_base')
expected = enc.encode(open('tiny.txt').read())
actual = np.load('data/shard_0.npy')
assert list(actual) == expected, 'Tokenization mismatch'
"

# 2. Training loss is better than random init
python -m pretrain.train --data "data/*.npy" --steps 100
# Loss should start ~10.8, end ~3-4 on small text (not 5+ like random)

# 3. Compare to LLMs-from-scratch Chapter 5 loss curves
# https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/
```

**Reference:** LLMs-from-scratch Chapter 2 (data prep), Chapter 5 (training)

---

## Step 3: Metrics & Checkpoints

**Goal**: Can resume training, can query metrics.

**Files:**
- `pretrain/metrics.py` - MetricsWriter (parquet)
- Add checkpointing to train.py

**Abstractions:**
- Metrics are `(step, tag, value)` tuples → parquet
- Checkpoint is `{weights, optimizer_state, step, config_fingerprint}`
- Config fingerprinting for resume validation

**Validation:**
```bash
# 1. Checkpoint saves/loads correctly
python -m pretrain.train --steps 50
python -m pretrain.train --resume output/step_00000050.pt --steps 100
# Should continue from step 50, loss should be continuous (no spike)

# 2. Metrics are queryable
pip install duckdb
duckdb -c "SELECT step, value FROM 'output/metrics/*.parquet' WHERE tag='loss' ORDER BY step"

# 3. Deterministic resume (same loss after resume)
python -m pretrain.train --seed 42 --steps 100 > run1.log
python -m pretrain.train --seed 42 --steps 50
python -m pretrain.train --resume output/step_00000050.pt --steps 100 > run2.log
# Compare: step 60-100 losses should match between run1 and run2
```

**Reference:** nmoe `configs/gates/gates.toml` has `zero2:resume_determinism` gate

---

## Step 4: Multi-GPU (Local)

**Goal**: `torchrun --nproc_per_node=8` works.

**Files:**
- `pretrain/distributed.py` - setup_distributed(), all_reduce_grads()
- Update train.py to use distributed

**Abstractions:**
- `rank, world_size, local_rank = setup_distributed()`
- `all_reduce_grads(weights)` after backward
- `print0()` for rank-0 only logging
- Data generator shards by rank

**Validation:**
```bash
# 1. 2-GPU loss matches 1-GPU (gradient averaging)
python -m pretrain.train --steps 50 --seed 42 > single.log
torchrun --standalone --nproc_per_node=2 -m pretrain.train --steps 50 --seed 42 > multi.log
# Loss values should be very close (within 1e-5)

# 2. Throughput scales ~linearly
# 2 GPU should be ~1.8-2x tokens/sec of 1 GPU

# 3. No NCCL errors or hangs
NCCL_DEBUG=INFO torchrun --standalone --nproc_per_node=2 -m pretrain.train --steps 10
```

**Reference:** modded-nanogpt distributed setup (lines 44-54)

---

## Step 5: Modal (8xH100)

**Goal**: Run on cloud GPUs.

**Files:**
- `scripts/run_modal.py` - Modal app definition

**Abstractions:**
- Same train.py works locally and on Modal
- Config passed as dict (serializable)

**Validation:**
```bash
# 1. Runs on Modal without error
modal run scripts/run_modal.py --steps 100

# 2. Throughput sanity check
# 8xH100 should get ~100k+ tokens/sec for small model
# Compare to modded-nanogpt (~170k tok/s on 8xH100 for GPT-2 124M)

# 3. Logs are retrievable
modal app logs pretrain
```

**Reference:** rollouts `rollouts/inference/tests/run_gpu_tests.py` (Modal sandbox pattern)

---

## Step 6: Verify Model Correctness

**Goal**: Our functional Llama matches HuggingFace exactly.

**Files:**
- `tests/test_model_parity.py` - compare against HF SmolLM2

**Abstractions:**
- Weight loading from HF checkpoint to our dict format
- Forward pass should match to 1e-5

**Validation:**
```bash
# 1. Exact numerical parity with HuggingFace
pytest tests/test_model_parity.py -v
# Should pass with max_diff < 1e-5

# 2. Test multiple sequence lengths
pytest tests/test_model_parity.py --seq-lens 1,16,128,512

# 3. Test with and without attention mask
pytest tests/test_model_parity.py --with-padding
```

**Reference:** rollouts `llama_functional.py` (already tested against SmolLM2-135M)

---

## Step 7: Attention Variants

**Goal**: Swap attention without touching rest of code.

**Files:**
- `pretrain/models/attention/mha.py` - multi-head attention
- `pretrain/models/attention/gqa.py` - grouped query
- `pretrain/models/attention/mla.py` - multi-head latent (DeepSeek)
- Update config to select attention type

**Abstractions:**
- `attention_fn(hidden, weights, config) -> hidden`
- Config has `attention_type: str`
- All attention fns have same signature

**Validation:**
```bash
# 1. All variants train (loss decreases)
for attn in mha gqa mla; do
  python -m pretrain.train --attention $attn --steps 100
done

# 2. GQA matches MHA when num_kv_heads == num_heads
python -m pretrain.train --attention mha --num-heads 8 --num-kv-heads 8 --seed 42 > mha.log
python -m pretrain.train --attention gqa --num-heads 8 --num-kv-heads 8 --seed 42 > gqa.log
# Losses should be identical

# 3. Compare final loss across variants (ablation)
# MHA ~= GQA (slightly worse) > MLA (better for same params)
```

**Reference:** DeepSeek-V2 paper for MLA, Llama 2 paper for GQA

---

## Step 8: MoE

**Goal**: Train a mixture-of-experts model.

**Files:**
- `pretrain/models/moe.py` - MoE layer (router + experts)
- `pretrain/models/router.py` - top-k routing, aux loss

**Abstractions:**
- MoE replaces MLP in some/all layers
- Router returns `(expert_weights, expert_indices, aux_loss)`
- Aux loss added to CE loss

**Validation:**
```bash
# 1. MoE trains (loss decreases)
python -m pretrain.train --config configs/moe_tiny.toml --steps 500

# 2. Router load balancing (no dead experts)
# Check metrics: expert_load_cv < 0.5, dead_experts = 0
duckdb -c "SELECT * FROM 'output/metrics/*.parquet' WHERE tag LIKE 'router%'"

# 3. Aux loss is small relative to CE loss
# aux_loss should be ~0.01-0.1, not dominating

# 4. Compare MoE vs dense at same active params
# MoE should have better loss for same compute
```

**Reference:** Mixtral paper, nmoe router implementation

---

## Step 9: Experiment Harness

**Goal**: Run ablations with multiple seeds/variants.

**Files:**
- `research/lab.py` - experiment runner
- `pretrain/experiments.py` - SQLite tracking (optional)

**Abstractions:**
- `lab.run(config, variants=[...], seeds=[0,1,2])`
- Results aggregated with mean ± std
- Completed runs skipped on resume

**Validation:**
```python
# 1. Multi-seed produces consistent variance
from research import lab
exp = lab.run("seed_test", variants=["default"], seeds=[0,1,2,3,4])
exp.summary()
# Std dev should be reasonable (~0.01-0.1 for loss)

# 2. Resume skips completed runs
lab.run("seed_test", variants=["default"], seeds=[0,1,2,3,4])
# Should print "skipping completed" for all 5

# 3. Comparison is statistically valid
exp = lab.run("attn_ablation", variants=["mha", "gqa"], seeds=[0,1,2])
exp.compare("mha", "gqa")
# Should show p-value or confidence interval
```

**Reference:** nmoe `research/lab.py` (PhysicsExperiment class)

---

## Step 10: QAT

**Goal**: Quantization-aware training.

**Files:**
- `pretrain/quantization.py` - fake quantize ops
- Update model to use quantized matmuls

**Abstractions:**
- `fake_quantize(x, bits=8)` for weights/activations
- Training sees quantization noise, inference is quantized
- Config has `quantize: bool`

**Validation:**
```bash
# 1. QAT trains (loss decreases, slightly worse than FP)
python -m pretrain.train --quantize --steps 500 > qat.log
python -m pretrain.train --steps 500 > fp.log
# QAT loss should be ~5-10% higher

# 2. Quantized weights are actually quantized
python -c "
import torch
ckpt = torch.load('output/step_00000500.pt')
w = ckpt['weights']['layers.0.mlp.gate_proj.weight']
unique = len(torch.unique(w))
print(f'Unique values: {unique}')  # Should be 256 for INT8
"

# 3. Inference matches training (no additional quantization error)
# Forward pass with fake_quantize should match post-training quantized
```

**Reference:** PyTorch quantization docs, bitsandbytes patterns

---

## Step 11: Scale (if needed)

**Goal**: 2x8x multi-node training.

**Files:**
- Integration with miniray for multi-node
- Tensor parallelism for large models

This is speculative - only implement if we need 100B+ scale.

**Validation:**
```bash
# 1. 16 GPU loss matches 8 GPU (gradient averaging)
# 2. No cross-node NCCL errors
# 3. Throughput scales ~1.8x from 8 to 16 GPU
```

**Reference:** miniray `nccl.py`, torchtitan distributed patterns

---

## CI/CD Setup

When ready, add GitHub Actions:

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v

  # GPU tests run manually or on specific branches
  gpu-test:
    runs-on: self-hosted  # or Modal
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - run: python scripts/run_modal.py --steps 100 --validate
```

**Reference:** nmoe `.github/workflows/tier-b.yml` for gate-based GPU validation

---

## Current Status

Mark with [x] as completed:

- [ ] Step 1: Train on Random Data
- [ ] Step 2: Real Data
- [ ] Step 3: Metrics & Checkpoints
- [ ] Step 4: Multi-GPU (Local)
- [ ] Step 5: Modal (8xH100)
- [ ] Step 6: Verify Model Correctness
- [ ] Step 7: Attention Variants
- [ ] Step 8: MoE
- [ ] Step 9: Experiment Harness
- [ ] Step 10: QAT
- [ ] Step 11: Scale

---

## Session Notes

### Starting Point

Key insight: each step should be runnable end-to-end. Don't build abstractions we haven't tested.

Abstractions we're betting on:
1. Functional models (`forward(ids, weights, config) -> logits`)
2. Weights as `dict[str, Tensor]`
3. Generator-based data (supports dynamic batch size)
4. Parquet metrics (queryable, no external deps)
5. Frozen dataclass configs with fingerprinting

These might be wrong - we'll adjust as we learn.

### External Validation Sources

- **llm.c**: Simple reference for loss curves, training correctness
- **modded-nanogpt**: Throughput benchmarks, distributed patterns
- **HuggingFace**: Numerical parity for model forward pass
- **lm-evaluation-harness**: Model quality benchmarks (later)
