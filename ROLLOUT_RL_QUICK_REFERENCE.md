# RL Training & Rollout Infrastructure - Quick Reference

## Core Components

### 1. SFT Training (Supervised Fine-Tuning)
**File:** `rollouts/training/loops/sft_loop.py`
```python
from rollouts.training import run_sft_training, SFTTrainingConfig, PyTorchTrainingBackend

backend = PyTorchTrainingBackend(model, optimizer, loss_fn, ...)
config = SFTTrainingConfig(num_steps=1000, batch_size=4, ...)
metrics = await run_sft_training(backend, samples, config, metrics_logger)
```

### 2. RL Training (Reinforcement Learning with GRPO)
**File:** `rollouts/training/loops/rl_loop.py`
```python
from rollouts.training import (
    run_rl_training, RLTrainingConfig, AsyncRolloutManager,
    RolloutConfig, DataBuffer, PyTorchTrainingBackend
)
from rollouts.training.weight_sync import SGLangEngine, sync_weights_to_engines

# Setup
buffer = DataBuffer(prompts=[...])
rollout_config = RolloutConfig(batch_size=32, generate_fn=my_generate_fn)
rollout_manager = AsyncRolloutManager(buffer, rollout_config)
engines = [SGLangEngine("http://localhost:30000")]
backend = PyTorchTrainingBackend(model, optimizer, grpo_loss, ...)
config = RLTrainingConfig(num_steps=1000, sync_every=10)

# Run
metrics = await run_rl_training(backend, buffer, rollout_manager, engines, config)
```

## Key Data Types

### Sample (Universal Training Unit)
```python
from rollouts.training.types import Sample

sample = Sample(
    prompt="What is 2+2?",
    response="4",
    tokens=[...],
    loss_mask=[0.0, 0.0, 1.0, 1.0],  # 0=prompt, 1=response
    reward=1.0,  # RL only
    metadata={"correct": True},
)
```

### RolloutBatch (Ready for Training)
```python
from rollouts.training.types import RolloutBatch

batch = RolloutBatch(
    tokens=[[...], [...]],
    loss_masks=[[...], [...]],
    rewards=[1.0, 0.0],
    response_lengths=[5, 3],
    metadata={...},
)
```

### Configs
```python
# SFT Config
from rollouts.training.types import SFTTrainingConfig
config = SFTTrainingConfig(
    num_steps=1000,
    batch_size=4,
    log_every=100,
    checkpoint_every=500,
)

# RL Config
from rollouts.training.types import RLTrainingConfig
config = RLTrainingConfig(
    num_steps=1000,
    sync_every=10,        # Sync weights every N steps
    baseline=0.0,         # Advantage baseline
    log_every=10,
    checkpoint_every=100,
)

# Rollout Config
from rollouts.training.types import RolloutConfig
config = RolloutConfig(
    batch_size=32,
    n_samples_per_prompt=1,
    over_sampling_factor=1.5,    # SLIME D4: over-sample by 50%
    generate_fn=my_fn,            # User-provided generation
    reward_fn=None,               # Optional reward function
    filter_fn=None,               # Optional quality filter
)
```

## Rollout Generation

### Async Rollout Manager (Recommended)
```python
from rollouts.training import AsyncRolloutManager, DataBuffer, RolloutConfig

buffer = DataBuffer(prompts=[...])
config = RolloutConfig(batch_size=32, generate_fn=my_generate_fn)
manager = AsyncRolloutManager(buffer, config)

async with manager:
    batch = await manager.generate_batch(reward_fn=my_reward_fn)
    # batch.tokens, batch.loss_masks, batch.rewards ready for training
```

Features:
- **Parallel generation** using trio (structured concurrency)
- **Over-sampling** (SLIME D4): generates N*1.5, filters down to N
- **Partial caching**: leftover samples from one batch → next batch
- **Optional filtering**: quality/diversity checks

### Pure Function Rollout Generation
```python
from rollouts.training.rollout_gen import generate_rollout_batches

config = RolloutConfig(batch_size=32, generate_fn=my_generate_fn)
batches = generate_rollout_batches(buffer, config)  # Generator

for batch in batches:
    # batch is RolloutBatch ready for training
    pass
```

## Training Backends

### Protocol (Implement These Methods)
```python
class TrainingBackend(Protocol):
    async def forward_backward(self, batch: Dict) -> TrainFuture[Dict[str, float]]:
        """Returns {"loss": float, "grad_norm": float, ...}"""
        ...
    
    async def optim_step(self) -> TrainFuture[Dict[str, float]]:
        """Returns {"lr": float, "step": int, ...}"""
        ...
    
    async def save_checkpoint(self, step: int, metrics: Dict) -> Path:
        """Save checkpoint, return path"""
        ...
```

### PyTorch Backend (Single-GPU)
```python
from rollouts.training import PyTorchTrainingBackend

backend = PyTorchTrainingBackend(
    model=model,
    optimizer=optimizer,
    loss_fn=my_loss_fn,
    checkpoint_dir=Path("checkpoints"),
    device=torch.device("cuda:0"),
)
```

### FSDP Backend (Multi-GPU)
```python
import torch.distributed as dist
from rollouts.training.backends.fsdp import FSDPTrainingBackend, FSDPConfig

# Must initialize distributed training first
dist.init_process_group(backend="nccl")

backend = FSDPTrainingBackend(
    model=model,
    optimizer_fn=lambda m: torch.optim.AdamW(m.parameters()),
    loss_fn=my_loss_fn,
    checkpoint_dir=Path("checkpoints"),
    config=FSDPConfig(
        sharding_strategy="FULL_SHARD",
        mixed_precision=True,
        gradient_checkpointing=False,
    ),
)
```

### Factory Functions (Convenience)
```python
from rollouts.training.backends import create_pytorch_backend

backend = create_pytorch_backend(
    model_name="Qwen/Qwen2.5-7B",
    checkpoint_dir=Path("checkpoints"),
    device_type="cuda",
    learning_rate=1e-4,
)
```

## Loss Functions

### GRPO Loss (Group Relative Policy Optimization)
```python
from rollouts.training.rl_losses import grpo_loss

# Compute inside forward_backward()
loss = grpo_loss(
    logits=model_output.logits,      # [batch, seq_len, vocab_size]
    labels=batch["labels"],           # [batch, seq_len]
    loss_mask=batch["loss_mask"],     # [batch, seq_len]
    advantages=batch["advantages"],   # [batch]
)

# Formula: Loss = -E[log_prob(sequence) * advantage]
# Only trains on response tokens (loss_mask > 0)
```

### PPO Loss (Placeholder - Not Implemented Yet)
```python
from rollouts.training.rl_losses import ppo_loss

loss = ppo_loss(
    logits=...,
    labels=...,
    loss_mask=...,
    advantages=...,
    old_log_probs=...,
    clip_range=0.2,
)
```

## Weight Synchronization (D5)

### Sync to Inference Engines
```python
from rollouts.training.weight_sync import (
    SGLangEngine, VLLMEngine,
    sync_weights_to_engines
)

engines = [
    SGLangEngine("http://localhost:30000"),
    VLLMEngine("http://localhost:30001"),
]

# After save_checkpoint()
ckpt_path = await backend.save_checkpoint(step, metrics)

# Sync in parallel
responses = await sync_weights_to_engines(engines, str(ckpt_path))
```

### Implement Custom Engine
```python
class MyEngine:
    async def update_weights_from_checkpoint(self, checkpoint_path: str) -> dict:
        # Custom implementation
        return {"success": True}

# Just add to engines list - Protocol-based, no inheritance needed
```

## Data Management

### DataBuffer (Prompt Cycling with Epochs)
```python
from rollouts.training.datasets import DataBuffer

buffer = DataBuffer(
    prompts=["What is 2+2?", "Calculate 5*7", ...],
    seed=42,
)

# Get prompts (handles epoch wraparound)
batch = buffer.get_prompts(32)

# Deterministic shuffling on epoch boundary
# save_state() / load_state() for checkpointing
state = buffer.save_state()
buffer.load_state(state)
```

### SFT Dataset Loading
```python
from rollouts.training.datasets import load_sft_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

samples = load_sft_dataset(
    "gsm8k",
    tokenizer=tokenizer,
    split="train",
    subset="main",
    max_samples=1000,
    max_length=2048,
)
```

## SFT → RL Integration

### Unified Training Pipeline
```python
# From /dev/integration_training/train.py

# Config specifies mode
if mode == "sft+rl":
    # Step 1: SFT Training
    await run_sft(config, output_dir)
    
    # Step 2: Find latest checkpoint
    latest_ckpt = sorted(output_dir / "checkpoints" / "step_*")[-1]
    
    # Step 3: RL Training from that checkpoint
    await run_rl(config, output_dir, latest_ckpt)
```

### Checkpoint Flow
```
SFT Output:  output_dir/checkpoints/step_0/
                                    step_1/
                                    step_N/
                                           ├─ pytorch_model.bin
                                           ├─ optimizer.bin
                                           └─ metadata.json
                                                    ↓↓↓
RL Input:    Load latest_checkpoint into backend
             Same model, same tokenizer, different training loop
```

## Metrics & Logging

### Simple File-Based Logging
```python
from rollouts.training.metrics import JSONLLogger

logger = JSONLLogger(Path("logs/experiment_001"))

# Called from training loops
logger.log({"loss": 0.5, "reward": 1.0, "step": 42}, step=42)

# Finishes JSONL file
logger.finish()
```

## Practical Workflows

### Single-GPU SFT
```python
from rollouts.training.backends import create_pytorch_backend
from rollouts.training import run_sft_training, SFTTrainingConfig

backend = create_pytorch_backend(
    model_name="Qwen/Qwen2.5-7B",
    checkpoint_dir=Path("ckpts"),
    learning_rate=1e-4,
)

metrics = await run_sft_training(
    backend=backend,
    samples=sft_samples,
    config=SFTTrainingConfig(num_steps=1000, batch_size=4),
)
```

### Multi-GPU SFT with FSDP
```bash
# Auto-detected by train.py if gpu_ranks has multiple GPUs
torchrun --nproc_per_node=4 train.py config.py
```

### RL with SGLang Inference
```python
from rollouts.agents import ... # Load your agent

async def generate_fn(prompts):
    """User-provided: prompts → samples"""
    samples = []
    for prompt in prompts:
        response = await agent.rollout(prompt)
        sample = Sample(
            prompt=prompt,
            response=response.text,
            tokens=tokenizer.encode(response.text),
            loss_mask=[0]*len(prompt_tokens) + [1]*len(response_tokens),
            metadata={"correct": response.correct},
        )
        samples.append(sample)
    return samples

# Setup
rollout_config = RolloutConfig(
    batch_size=32,
    generate_fn=generate_fn,
    filter_fn=check_quality_and_diversity,
)

manager = AsyncRolloutManager(buffer, rollout_config)
engines = [SGLangEngine("http://localhost:30000")]

await run_rl_training(backend, buffer, manager, engines, rl_config)
```

## Common Patterns

### Advantage Computation
```python
from rollouts.training.loops.rl_loop import compute_advantages

# Simple: reward - baseline
advantages = compute_advantages(rewards, baseline=0.5)

# Used in prepare_grpo_batch()
rl_batch["advantages"] = torch.tensor(advantages)
```

### Sample Filtering (SLIME-Style)
```python
from rollouts.training.filters import (
    check_reward_nonzero_std,
    check_min_reward,
    check_response_diversity,
    check_quality_and_diversity,
)

# Custom filter function
def my_filter(samples: list[Sample]) -> bool:
    """Return True if group passes quality checks"""
    return all([
        check_min_reward(samples, threshold=0.5),
        check_response_diversity(samples),
    ])

config = RolloutConfig(
    batch_size=32,
    generate_fn=my_fn,
    filter_fn=my_filter,
)
```

### Pipelining with Futures
```python
# Trio-based futures allow pipelining
fwd_future = backend.forward_backward(batch)
# ... do other work ...
fwd_metrics = await fwd_future.result()

# Or wait immediately
opt_metrics = await backend.optim_step().result()
```

## Troubleshooting

### FSDP Issues
- Ensure `torch.distributed.init_process_group()` called before backend creation
- Use `torchrun` for multi-GPU launch
- Check `CUDA_VISIBLE_DEVICES` matches `gpu_ranks` config

### Checkpoint Loading
```python
# Currently TODO - not yet implemented!
# For now: backend = create_pytorch_backend(...)  # Fresh model
# Load weights manually if needed
```

### Over-Sampling Not Working
- Set `over_sampling_factor > 1.0` in RolloutConfig
- Must use `AsyncRolloutManager`, not `generate_rollout_batches()`

### Weight Sync Failing
- Check SGLang/vLLM servers are running
- Verify network connectivity to server URLs
- Monitor server logs for errors

## Files to Know

```
rollouts/training/
├── loops/
│   ├── sft_loop.py              # run_sft_training()
│   └── rl_loop.py               # run_rl_training()
├── rollout_gen/
│   ├── async_rollout_manager.py # AsyncRolloutManager
│   └── rollout_generation.py    # Pure generation functions
├── backends/
│   ├── pytorch.py               # PyTorchTrainingBackend
│   ├── fsdp.py                  # FSDPTrainingBackend
│   └── pytorch_factory.py       # create_pytorch_backend()
├── datasets/
│   ├── data_buffer.py           # DataBuffer
│   └── sft.py                   # SFT utilities
├── types.py                     # Sample, RolloutBatch, Configs
├── rl_losses.py                 # GRPO, PPO losses
├── weight_sync.py               # Engine sync
├── metrics.py                   # Logging
├── filters.py                   # Quality filters
└── agent_integration.py         # Agent → Sample conversion
```

## Design Principles

1. **Pure Functions** - Training loops have no hidden state
2. **Protocol-Based** - Backends use Protocol, no inheritance
3. **Explicit Parameters** - Dependencies passed explicitly, not globals
4. **Tiger Style** - Assertions for preconditions/postconditions
5. **SLIME Patterns** - Over-sampling, partial caching, weight versioning
6. **Tinker Futures** - Async pipelining with trio-based futures
