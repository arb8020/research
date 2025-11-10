# Slime: Technical Reference Guide

## Directory Structure

```
/slime/
├── slime/                          # Main package
│   ├── backends/                   # Training backends
│   │   ├── megatron_utils/        # Megatron-LM integration (primary)
│   │   │   ├── actor.py           # MegatronTrainRayActor - training wrapper
│   │   │   ├── model.py           # Model initialization and training loop
│   │   │   ├── loss.py            # Loss computation (PPO, GRPO, value, SFT)
│   │   │   ├── data.py            # Data iterator and batch processing
│   │   │   ├── checkpoint.py      # Checkpoint save/load
│   │   │   ├── initialize.py      # Megatron initialization
│   │   │   ├── cp_utils.py        # Context parallelism utilities
│   │   │   ├── update_weight_utils.py  # Weight synchronization
│   │   │   └── megatron_to_hf/    # Model format converters
│   │   │
│   │   ├── fsdp_utils/            # Fully Sharded Data Parallel (alternative)
│   │   │   ├── actor.py           # FSDPTrainRayActor
│   │   │   ├── data_packing.py    # Sequence packing for efficiency
│   │   │   ├── checkpoint.py      # FSDP checkpoint management
│   │   │   └── fsdp_cpu_adam_wrapper.py  # CPU offload support
│   │   │
│   │   └── sglang_utils/          # SGLang inference integration
│   │       ├── sglang_engine.py   # SGLang engine wrapper
│   │       └── arguments.py       # SGLang-specific arguments
│   │
│   ├── ray/                        # Ray-based orchestration
│   │   ├── train_actor.py         # Base training actor class
│   │   ├── rollout.py             # RolloutManager - coordinates generation
│   │   ├── rollout_data_source.py # Dataset and sample management
│   │   ├── actor_group.py         # Actor group management
│   │   ├── placement_group.py     # GPU placement and allocation
│   │   ├── ray_actor.py           # Base Ray actor class
│   │   └── utils.py               # Ray utilities
│   │
│   ├── rollout/                    # Rollout and reward computation
│   │   ├── base_types.py          # Rollout output types
│   │   ├── sft_rollout.py         # SFT generation
│   │   ├── sglang_rollout.py      # SGLang-native rollout
│   │   ├── sleep_rollout.py       # Debug rollout (synthetic)
│   │   ├── filter_hub/            # Dynamic sample filtering
│   │   │   └── dynamic_sampling_filters.py
│   │   └── rm_hub/                # Reward models
│   │       ├── math_utils.py      # Math answer grading
│   │       ├── math_dapo_utils.py # Process rewards
│   │       ├── gpqa.py            # Graduate QA
│   │       ├── ifbench.py         # Instruction following
│   │       ├── deepscaler.py      # Composite rewards
│   │       └── f1.py              # F1-based scoring
│   │
│   ├── router/                     # Custom request routing
│   │   ├── router.py              # Main router with middleware
│   │   └── middleware_hub/        # Extensible middleware
│   │       └── radix_tree_middleware.py
│   │
│   └── utils/                      # Utilities
│       ├── arguments.py           # Comprehensive argument parsing
│       ├── ppo_utils.py           # PPO/KL/advantage computation
│       ├── types.py               # Sample, RolloutBatch types
│       ├── data.py                # Dataset class
│       ├── metric_utils.py        # Pass@k, statistics
│       ├── train_metric_utils.py  # Performance logging
│       ├── eval_config.py         # Evaluation dataset config
│       ├── distributed_utils.py   # Distributed communication
│       ├── memory_utils.py        # Memory profiling
│       ├── timer.py               # Performance timing
│       ├── wandb_utils.py         # W&B integration
│       ├── tensorboard_utils.py   # TensorBoard integration
│       ├── health_monitor.py      # Fault tolerance
│       ├── router_replay.py       # Replay functionality
│       ├── seqlen_balancing.py    # Balanced partitioning
│       └── debug_utils/           # Debugging utilities
│
├── train.py                        # Main training entry point
├── train_async.py                  # Async training variant
├── examples/                       # Example implementations
│   ├── search-r1/                 # Multi-turn search + tool calling
│   ├── multi_agent/               # Multi-agent coordination
│   ├── fully_async/               # Fully asynchronous RL
│   ├── true_on_policy/            # True on-policy training
│   ├── retool/                    # Tool-use and planning
│   ├── eval_multi_task/           # Multi-task evaluation
│   └── reproducibility/           # Reproducibility features
│
├── docs/                           # Documentation
│   └── en/
│       ├── get_started/
│       │   ├── quick_start.md
│       │   ├── usage.md
│       │   └── qa.md
│       ├── advanced/
│       │   ├── arch-support-beyond-megatron.md
│       │   ├── fault-tolerance.md
│       │   └── speculative-decoding.md
│       ├── developer_guide/
│       └── blogs/
│
└── tools/                          # Model conversion tools
    ├── convert_hf_to_torch_dist.py
    ├── convert_torch_dist_to_hf.py
    ├── convert_to_hf.py
    └── fp8 quantization tools
```

---

## Key Classes & Their Responsibilities

### Core Training Components

#### MegatronTrainRayActor (`/slime/backends/megatron_utils/actor.py`)

Main training actor for Megatron backend.

**Key Methods**:
- `init(args, role, wandb_run_id, with_ref)`: Initialize model, tokenizer, optimizer
- `train(rollout_id, rollout_data_ref)`: Async training on rollout batch
- `save_model(rollout_id)`: Save checkpoint
- `update_weights()`: Broadcast weights to rollout engines
- `offload()`: Move weights to CPU

**Internal Flow**:
1. Load model from checkpoint or init from scratch
2. Wrap with DDP for distributed training
3. For each rollout:
   - Get batch from rollout data
   - Forward pass to compute logits
   - Extract log probs and values
   - Compute loss (PPO/GRPO/Value)
   - Backward pass with gradient accumulation
   - Optimizer step
   - Broadcast weights to rollout

#### FSDPTrainRayActor (`/slime/backends/fsdp_utils/actor.py`)

Alternative training actor for FSDP backend.

**Differences from Megatron**:
- Uses PyTorch FSDP instead of Megatron's distributed
- Simpler model sharding
- Better for HuggingFace models
- Built-in CPU offload support

#### RolloutManager (`/slime/ray/rollout.py`)

Orchestrates rollout generation and reward computation.

**Key Methods**:
- `generate(rollout_id)`: Generate rollouts using SGLang engines
- `eval(rollout_id)`: Run evaluation on eval datasets
- `save(rollout_id)`: Persist rollout data
- `dispose()`: Cleanup resources
- `onload()`: Load weights/KV cache to GPU
- `offload()`: Move to CPU to free VRAM

**Generation Pipeline**:
1. Get prompts from RolloutDataSource
2. Create generation requests
3. Send to SGLang engines via router
4. Collect generations
5. Compute rewards
6. Apply dynamic filters (if configured)
7. Convert to training format
8. Return as Ray object reference

#### RolloutDataSource (`/slime/ray/rollout_data_source.py`)

Manages dataset loading and sample creation.

**Key Methods**:
- `get_samples(num_samples)`: Get next batch of prompts
- `add_samples(samples)`: Add generated samples (for buffering)

**Flow**:
1. Load dataset (JSONL/Parquet) on init
2. Tokenize prompts
3. On each call, return next batch with shuffling/cycling
4. Track epoch and sample indices for reconstruction

---

## Loss Functions & RL Algorithms

### Loss Computation (`/slime/backends/megatron_utils/loss.py`)

#### policy_loss_function

Computes PPO/GSPO policy gradient loss.

```python
def policy_loss_function(args, batch, logits, sum_of_sample_mean):
    # 1. Extract advantages from batch
    advantages = torch.cat(batch["advantages"])
    old_log_probs = batch["rollout_log_probs"] or batch["log_probs"]
    
    # 2. Compute current log probs and entropy
    log_probs = get_log_probs_and_entropy(logits, batch, args)
    
    # 3. For GSPO: all-gather to get full sequence KL
    if args.advantage_estimator == "gspo":
        # Compute per-sample KL across distributed ranks
        ppo_kl = compute_per_sample_kl_with_cp(...)
    else:
        # Token-level KL ratio
        ppo_kl = old_log_probs - log_probs
    
    # 4. Compute clipped policy gradient loss
    pg_loss, clipfrac = compute_policy_loss(
        ppo_kl, advantages, 
        args.eps_clip, args.eps_clip_high
    )
    
    # 5. Apply off-policy correction (TIS) if configured
    if args.use_tis:
        pg_loss = pg_loss * tis_weights
    
    # 6. Add entropy regularization and KL loss
    entropy_loss = entropy.mean()
    loss = pg_loss - args.entropy_coef * entropy_loss
    if args.use_kl_loss:
        loss += args.kl_loss_coef * kl_loss
    
    return loss, metrics
```

#### value_loss_function

Computes clipped value head loss (PPO-style).

```python
def value_loss_function(args, batch, logits, sum_of_sample_mean):
    old_values = batch["values"]
    returns = batch["returns"]
    
    # Current value predictions
    new_values = get_values(logits, batch, args)
    
    # Clipped loss: max of clipped and unclipped
    unclipped = (new_values - returns) ** 2
    clipped = (old_values + (new_values - old_values).clamp(
        -args.value_clip, args.value_clip
    ) - returns) ** 2
    
    loss = torch.max(unclipped, clipped).mean()
    return loss, metrics
```

#### sft_loss_function

Standard cross-entropy loss for SFT training.

```python
def sft_loss_function(args, batch, logits, sum_of_sample_mean):
    # Compute log probs of ground truth tokens
    log_probs = get_log_probs_and_entropy(logits, batch, args)
    
    # NLL loss
    loss = -log_probs.mean()
    return loss, metrics
```

### Advantage Computation (`/slime/utils/ppo_utils.py`)

#### compute_approx_kl

Computes KL divergence approximation.

```python
def compute_approx_kl(log_probs, log_probs_base, kl_loss_type):
    log_ratio = log_probs - log_probs_base
    
    if kl_loss_type == "k1":
        return log_ratio  # Direct ratio
    elif kl_loss_type == "k2":
        return (log_ratio ** 2) / 2.0  # Squared
    elif kl_loss_type == "k3":
        # Reverse KL: E[exp(-r) - 1 + r]
        return (-log_ratio).exp() - 1 - log_ratio
    elif kl_loss_type == "low_var_kl":
        # Clamped for lower variance
        kl = (-log_ratio).exp() - 1 - log_ratio
        return torch.clamp(kl, min=-10, max=10)
```

#### get_advantages_and_returns

Computes GAE-style advantages for PPO.

```python
def get_advantages_and_returns(total_len, response_len, values, rewards, gamma, lambd):
    # Extract response values
    response_values = values[-(response_len+1):]
    
    # Compute TD residuals
    td_residuals = rewards[:-1] + gamma * response_values[1:] - response_values[:-1]
    
    # Compute advantages via GAE
    advantages = []
    gae = 0
    for t in reversed(range(len(td_residuals))):
        gae = td_residuals[t] + (gamma * lambd) * gae
        advantages.insert(0, gae)
    
    returns = advantages + response_values[-1]
    return advantages, returns
```

#### get_grpo_returns

Simple group-relative returns (no value function).

```python
def get_grpo_returns(rewards, kl):
    # For each sample, returns = reward (broadcasted)
    # This treats all tokens equally with the same reward signal
    returns = [torch.ones_like(k) * r for k, r in zip(kl, rewards)]
    return returns
```

#### get_reinforce_plus_plus_returns

Token-level returns with discount factor.

```python
def get_reinforce_plus_plus_returns(rewards, kl, loss_masks, 
                                     response_lengths, total_lengths,
                                     kl_coef, gamma):
    returns = []
    for i in range(len(rewards)):
        # Token-level rewards: -kl_coef * kl + final_reward at last token
        token_rewards = -kl_coef * kl[i]
        token_rewards[-1] += rewards[i]
        
        # Compute discounted returns backwards
        returns_seq = torch.zeros_like(token_rewards)
        running_return = 0.0
        for t in reversed(range(len(token_rewards))):
            running_return = token_rewards[t] + gamma * running_return
            returns_seq[t] = running_return
        
        returns.append(returns_seq)
    return returns
```

---

## Data Flow & Types

### Sample Lifecycle

```python
@dataclass
class Sample:
    # Generation inputs
    prompt: Union[str, list[dict]]     # User input
    tokens: list[int]                  # Tokenized
    
    # Generation outputs
    response: str                      # Model output
    response_length: int
    
    # Reward & training signals
    reward: Optional[float|dict]       # Scalar or multi-component
    loss_mask: list[int]              # Which tokens count in loss
    
    # Metadata for off-policy training
    weight_versions: list[str]         # Which model weights generated this
    rollout_log_probs: list[float]    # Log probs at generation time
    
    # Status tracking
    status: Status                     # COMPLETED, TRUNCATED, etc.
    spec_info: SpecInfo               # Speculative decoding stats
```

### RolloutBatch Type

Dict-based batch that flows through training pipeline:

```python
RolloutBatch = dict[str, list[torch.Tensor] | list[int] | list[float] | list[str]]

# Example batch structure:
{
    # Input sequences
    "unconcat_tokens": [torch.Tensor(seq_len), ...],  # Full sequences
    "total_lengths": [256, 512, ...],                  # Seq lengths
    "response_lengths": [100, 150, ...],               # Response-only lengths
    
    # Model outputs (from inference)
    "log_probs": [torch.Tensor(resp_len), ...],       # Policy log probs
    "ref_log_probs": [torch.Tensor(resp_len), ...],   # Reference log probs
    "rollout_log_probs": [torch.Tensor(...), ...],    # Generation-time log probs
    "values": [torch.Tensor(resp_len), ...],          # Value function outputs
    
    # Rewards & advantages
    "rewards": [0.5, 1.0, 0.0, ...],                  # Scalar rewards
    "loss_masks": [torch.Tensor(resp_len), ...],      # Response-only masks
    
    # Computed quantities
    "advantages": [torch.Tensor(resp_len), ...],      # Advantage estimates
    "returns": [torch.Tensor(resp_len), ...],         # Target returns
}
```

---

## Training Loop Details

### Megatron Forward/Backward Pass

**Forward**:
```python
# 1. Megatron's get_forward_backward_func handles pipelining
forward_backward_func(
    forward_step_func=lambda batch: forward_step(model, batch),
    backward_step_func=lambda logits, batch: loss_func(...),
    data_iterator=batch_iterator,
    model=model,
    num_microbatches=args.gradient_accumulation_steps,
    seq_length=None,  # Variable length
    forward_only=False,
)

# 2. Inside forward_step, for each micro-batch:
logits = model(tokens)  # Shape: [batch, seq_len, vocab_size]

# 3. Inside loss_function:
loss, normalizer, logging = loss_function(
    args, batch, num_microbatches, logits
)
```

**Backward**:
```python
# 1. Megatron handles backward automatically
# 2. Gradient accumulation across micro-batches
# 3. All-reduce across data-parallel group
# 4. Optimizer step

# After step:
optimizer.step()
opt_param_scheduler.step()
```

### Memory Optimization: Context Parallelism

When using CP (Context Parallelism), sequences are split across ranks:

```python
# Example with CP size=2:
# Rank 0: processes tokens [0:128]
# Rank 1: processes tokens [128:256]

# For PPO loss computation with CP:
def get_logits_and_tokens_offset_with_cp(total_len, response_len):
    # Returns which tokens each rank should compute loss for
    # Ensures response tokens are correctly attributed
    return chunk_size, chunks_offset, logits_offset, tokens_offset

# Loss is computed on CP-partitioned chunks, then reduced
```

---

## Reward Model Implementation Example

### Custom Reward Function

```python
# In custom_rewards.py:
from slime.utils.types import Sample

def compute_reward(samples: list[Sample], **kwargs) -> list[float]:
    """
    Args:
        samples: List of samples with response generated
        **kwargs: Additional context (args, model, etc.)
    
    Returns:
        List of scalar rewards, one per sample
    """
    rewards = []
    for sample in samples:
        # Option 1: Exact match grading
        if sample.label:
            reward = 1.0 if sample.response.strip() == sample.label.strip() else 0.0
        
        # Option 2: LLM-based reward
        else:
            # Call reward model
            score = reward_model.score(sample.prompt, sample.response)
            reward = score / max_score
        
        rewards.append(reward)
    
    return rewards

# In training script:
python train.py \
    --custom-rm-path custom_rewards.compute_reward \
    ...
```

---

## Configuration Patterns

### Typical PPO Setup

```bash
python train.py \
    --hf-checkpoint meta-llama/Llama-3-8b \
    --actor-num-nodes 2 \
    --actor-num-gpus-per-node 8 \
    --rollout-num-gpus 8 \
    \
    # RL algorithm
    --loss-type policy_loss \
    --advantage-estimator ppo \
    --kl-coef 0.05 \
    --kl-loss-type k3 \
    --eps-clip 0.2 \
    --entropy-coef 0.01 \
    --gamma 0.99 \
    --lambd 0.95 \
    \
    # Training
    --num-rollout 1000 \
    --rollout-batch-size 256 \
    --global-batch-size 256 \
    --micro-batch-size 8 \
    --lr 1e-6 \
    \
    # Data
    --rollout-global-dataset \
    --prompt-data prompts.jsonl \
    --rollout-max-prompt-len 1024 \
    --rollout-max-response-len 2048 \
    \
    # Output
    --output-dir ./checkpoints \
    --save-interval 50 \
    --eval-interval 100
```

### Typical GRPO Setup (Simpler)

```bash
python train.py \
    --hf-checkpoint meta-llama/Llama-3-8b \
    \
    # GRPO: simpler, no value function
    --loss-type policy_loss \
    --advantage-estimator grpo \
    --kl-coef 0.05 \
    --entropy-coef 0.0 \
    \
    # Don't need critic
    --use-critic false \
    \
    # Rest similar to PPO...
```

### Memory-Constrained Setup

```bash
python train.py \
    --hf-checkpoint meta-llama/Llama-3-70b \
    \
    # Tensor parallelism
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 2 \
    \
    # Memory saving
    --colocate \  # Share GPUs
    --offload-train \  # CPU offload during training
    --gradient-checkpointing \
    \
    # Smaller batches
    --global-batch-size 64 \
    --micro-batch-size 2 \
    
    # Smaller rollouts
    --rollout-batch-size 32 \
    --n-samples-per-prompt 1
```

---

## Performance Optimization Tips

### 1. Batch Size Tuning

**Global Batch Size** = (rollout_batch_size * n_samples_per_prompt * num_rollout_agents)

- Larger: Better gradient estimates, more stable training
- Smaller: More frequent weight updates, potential for mode-seeking

Typical: 256-512 for 70B models

### 2. Sequence Length Balancing

```python
# Enable in data.py:
from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions

# Reduces padding waste
# Improves compute efficiency
```

### 3. Data Packing (FSDP Only)

```bash
--fsdp-max-tokens-per-gpu 8192  # Pack sequences to this limit
```

Reduces padding from potentially 90% to 10-20%

### 4. Offloading Strategy

- **--colocate**: Best when memory is severely constrained
- **--offload-train --offload-rollout**: Good balance
- **No offload**: Best performance if memory allows

### 5. Learning Rate Scheduling

```bash
--lr-decay-style cosine
--lr-warmup-iters 500
--lr-decay-iters 10000
--min-lr 1e-7
```

### 6. KL Coefficient Tuning

- Start with `--kl-coef 0.05`
- If diverging: increase to 0.1 or 0.2
- If not exploring enough: decrease to 0.01

### 7. Advantage Normalization

```bash
--normalize-advantages  # Whiten advantages across DP group
```

Critical for stable training. Use distributed masked whitening.

---

## Debugging & Profiling

### Enable Detailed Logging

```bash
python train.py \
    ... \
    --wandb-project my-rl-training \
    --use-wandb \
    --log-model-config \
    --dump-details ./debug_data
```

### Check Generation Quality

```python
# In debug mode, slime saves:
# - First few prompts and generations
# - Reward distribution
# - Token-level metrics
```

### Profile Training Bottleneck

Look for in W&B:

- **perf/wait_time_ratio > 0.3**: Synchronization bottleneck
  - Solution: Increase batch size or use async training
  
- **perf/actor_train_tflops < 50**: Compute efficiency issue
  - Solution: Check if memory-bound, increase tensor parallelism
  
- **perf/log_probs_time > perf/actor_train_time**: Reference model is slow
  - Solution: Use quantized reference model or remove reference model

### Validation

```python
# Check advantage statistics in W&B:
# - advantages should have mean ≈ 0, std ≈ 1 (if normalized)
# - pg_clipfrac should be 0.2-0.3 (not 0, not >0.5)
# - entropy should be >0 (model still exploring)
# - ppo_kl should increase slightly then plateau
```

---

## Extension Points

### Custom Rollout Function

```python
# my_rollout.py
def generate(prompts, model, temperature=0.7, **kwargs):
    """
    Must return list[list[Sample]]
    """
    results = []
    for prompt_group in prompts:  # Groups of n_samples_per_prompt
        samples = []
        for _ in range(len(prompt_group)):
            output = model.generate(prompt_group[0].prompt, temperature=temperature)
            sample = Sample(
                prompt=prompt_group[0].prompt,
                response=output,
                tokens=[...],  # Your tokenization
            )
            samples.append(sample)
        results.append(samples)
    return results

# Usage:
python train.py --rollout-function-path my_rollout.generate ...
```

### Custom Reward Model

```python
# my_rewards.py
def compute_reward(samples, **kwargs):
    """Compute scalar reward for each sample"""
    scores = llm_judge([(s.prompt, s.response) for s in samples])
    return [s / 10.0 for s in scores]  # Normalize to [0,1]
```

### Custom Loss Function

```python
# my_loss.py
def custom_loss(args, batch, logits, sum_of_sample_mean):
    """
    Args:
        batch: RolloutBatch dict
        logits: Model outputs
    
    Returns:
        loss: scalar tensor
        metrics: dict of scalar tensors
    """
    # Your custom loss computation
    loss = ...
    metrics = {"my_metric": ...}
    return loss, metrics

# Usage:
python train.py --loss-type custom_loss \
                 --custom-loss-function-path my_loss.custom_loss ...
```

---

## Troubleshooting

| Problem | Diagnosis | Solution |
|---------|-----------|----------|
| Training diverges | KL too high, entropy → 0 | Increase --kl-coef, reduce --eps-clip |
| No learning signal | Rewards all same, pg_clipfrac ≈ 0 | Check reward function, increase batch size |
| VRAM OOM | Memory profiler shows peak > 80GB | Enable --offload, reduce --micro-batch-size |
| Slow training | perf/wait_time_ratio > 0.3 | Use async training, increase rollout batch size |
| Model output collapse | All samples identical | Add entropy regularization, reduce temperature |
| Eval metrics not improving | Training loss decreasing but eval flat | Check eval reward function, increase num_rollout |

---

