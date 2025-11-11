# SLIME Hyperparameters Quick Reference

## Source Files
- Main args: `references/slime/slime/utils/arguments.py` (1448 lines)
- FSDP args: `references/slime/slime/backends/fsdp_utils/arguments.py` (78 lines)
- SGLang args: `references/slime/slime/backends/sglang_utils/arguments.py` (126 lines)
- Megatron args: `references/slime/slime/backends/megatron_utils/arguments.py` (27 lines)

## Most Important Parameters for Tuning

### Training Stability (Critical)
```
--clip-grad 1.0              # Gradient clipping (prevents explosions)
--lr 1e-6 (megatron) / 2e-5 (fsdp)  # Learning rate
--warmup-ratio 0.03          # Linear warmup as fraction of total steps
--weight-decay 0.0           # L2 regularization
```

### Batch Configuration (Critical)
```
--rollout-batch-size N       # REQUIRED! Number of prompts per rollout
--global-batch-size M        # Total batch across all GPUs
--micro-batch-size 1         # Per-GPU batch size
--n-samples-per-prompt 1     # Responses per prompt
```

### PPO/RL Algorithms
```
--eps-clip 0.2               # PPO clip range (trust region)
--value-clip 0.2             # Value function clip
--gamma 1.0                  # Discount factor
--lambd 1.0                  # GAE lambda (0=TD, 1=MC)
--kl-coef 0.0                # KL penalty on reward
--kl-loss-coef 0.0           # KL penalty on loss
--advantage-estimator grpo   # Options: grpo, ppo, reinforce_plus_plus
```

### Generation Quality
```
--rollout-temperature 1.0    # Sampling temperature (0=greedy)
--rollout-top-p 1.0          # Nucleus sampling
--rollout-top-k -1           # Top-k sampling (-1=disabled)
--rollout-max-response-len 1024  # Max tokens to generate
```

### Adam Optimizer Details
```
--adam-beta1 0.9             # Momentum (first moment)
--adam-beta2 0.95            # Variance (second moment)
--adam-eps 1e-8              # Numerical stability
```

### Hardware/Distributed
```
--actor-num-nodes 1          # Training nodes
--actor-num-gpus-per-node 8  # Training GPUs per node
--rollout-num-gpus-per-engine 1  # Inference GPUs (tensor parallel)
--train-backend megatron     # Options: megatron, fsdp
--colocate false             # Share training and inference GPUs?
```

### Data & Evaluation
```
--prompt-data path.jsonl     # Input dataset path
--rollout-batch-size N       # Prompts per rollout
--use-dynamic-batch-size false  # Adaptive batching for variable lengths
--max-tokens-per-gpu None    # For dynamic batching
--eval-interval None         # Eval frequency (if None: no eval)
```

## Parameter Relationships (Important!)

### Batch Size Computation
```
Total Training Batch = rollout_batch_size * n_samples_per_prompt / num_steps_per_rollout
Per-GPU Batch = global_batch_size / num_gpus
```

### Warmup Schedule
```
Warmup Steps = total_training_steps * warmup_ratio
LR(step) = lr * min(1.0, step / warmup_steps)
```

### PPO Loss (with cliprange epsilon)
```
ratio = exp(log_prob_new - log_prob_old)
clipped_ratio = clip(ratio, 1-eps_clip, 1+eps_clip)
loss = -min(ratio, clipped_ratio) * advantage
```

### KL Divergence Penalty
- `kl_coef`: Applied to **reward** before advantage computation
- `kl_loss_coef`: Applied to **loss** after policy gradient

**Cannot use both simultaneously** (validation error)

## Meta Switches

### Loss Functions
- `--loss-type policy_loss` - PPO-style (default)
- `--loss-type sft_loss` - Supervised learning
- `--loss-type custom_loss` - User-defined (requires --custom-loss-function-path)

### Advantage Estimators
- `grpo` - REINFORCE + baseline (default)
- `ppo` - Full PPO with critic
- `reinforce_plus_plus` - REINFORCE variants (requires --normalize-advantages)
- `gspo` - GSPO algorithm

### Backends
- `megatron` - Default, uses Megatron LLM engine
- `fsdp` - Fully Sharded Data Parallel (experimental)

## Default Values Summary

| Parameter | Default | Type | Range |
|-----------|---------|------|-------|
| lr (megatron) | 1e-6 | float | 1e-7 to 1e-3 |
| lr (fsdp) | 2e-5 | float | 1e-7 to 1e-3 |
| warmup_ratio | 0.03 | float | 0-0.5 |
| adam_beta1 | 0.9 | float | 0.8-0.99 |
| adam_beta2 | 0.95 | float | 0.9-0.999 |
| weight_decay | 0.0 | float | 0-0.1 |
| clip_grad | 1.0 | float | 0.1-10.0 |
| eps_clip | 0.2 | float | 0.1-0.5 |
| gamma | 1.0 | float | 0.9-1.0 |
| lambd | 1.0 | float | 0-1.0 |
| kl_coef | 0.0 | float | 0-1.0 |
| temperature | 1.0 | float | 0-2.0 |
| top_p | 1.0 | float | 0-1.0 |
| max_response_len | 1024 | int | 128-4096 |

## Common Configuration Examples

### Debug/Small Scale (Single GPU)
```bash
--rollout-batch-size 8
--global-batch-size 8
--micro-batch-size 1
--actor-num-nodes 1
--actor-num-gpus-per-node 1
--lr 2e-5
--warmup-ratio 0.03
```

### Medium Scale (4 GPUs)
```bash
--rollout-batch-size 32
--global-batch-size 32
--micro-batch-size 4
--actor-num-nodes 1
--actor-num-gpus-per-node 4
--lr 1e-5
--warmup-ratio 0.03
```

### Large Scale (64 GPUs, 8 nodes)
```bash
--rollout-batch-size 256
--global-batch-size 256
--micro-batch-size 8
--actor-num-nodes 8
--actor-num-gpus-per-node 8
--lr 5e-6
--warmup-ratio 0.05
--train-backend megatron
```

## Key Insights from Code

1. **SLIME uses much smaller LR for Megatron** (1e-6 default vs 2e-5 for FSDP)
2. **Adam beta2 is aggressive** (0.95 vs standard 0.999) - faster adaptation
3. **Gradient clipping is always on** (default 1.0) - prevents instability
4. **No L2 regularization by default** (weight_decay=0.0)
5. **Dynamic batch sizing available** but not enabled by default
6. **Colocate mode** combines training and inference on same GPUs (useful for memory)
7. **Reference model required** for KL penalties (via --ref-load)
8. **FSDP backend is experimental** - use Megatron for production

## File Organization

All hyperparameters documented in 3 formats:
1. **SLIME_HYPERPARAMETERS.md** - Detailed explanations (this document's source)
2. **SLIME_HYPERPARAMETERS.json** - Structured reference (for tooling)
3. **SLIME_HYPERPARAMETERS_QUICK_REFERENCE.md** - Quick lookup (this file)

Total parameters tracked: 100+ across all categories
