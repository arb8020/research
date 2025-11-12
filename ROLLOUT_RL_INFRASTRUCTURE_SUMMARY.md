# RL Training & Rollout Infrastructure Analysis

## Overview
This codebase has a well-structured rollout and RL training infrastructure built on principles from SLIME, Casey Muratori's immediate-mode design, and Tiger Style assertions. The system supports both SFT (Supervised Fine-Tuning) and RL (Reinforcement Learning) training stages.

## Core Architecture

### 1. Training Loop Structure (Pure Functional Design)

**Location:** `/Users/chiraagbalu/research/rollouts/rollouts/training/loops/`

#### SFT Training Loop
- **File:** `sft_loop.py` - `run_sft_training()`
- Pure function (no hidden state)
- Orchestrates: batch collation → forward/backward → optimizer step
- Supports metrics logging and checkpointing
- Takes stateful dependencies (backend, metrics_logger) as explicit parameters

```python
async def run_sft_training(
    backend: PyTorchTrainingBackend,
    samples: List[Sample],
    config: SFTTrainingConfig,
    metrics_logger: Optional[MetricsLogger] = None,
) -> List[Dict[str, float]]
```

#### RL Training Loop
- **File:** `rl_loop.py` - `run_rl_training()`
- Implements SLIME-inspired: Generate → Train → Weight Sync loop
- Key steps:
  1. Generate rollouts via AsyncRolloutManager
  2. Compute rewards (from environment grading)
  3. Prepare GRPO batch (convert rewards to advantages)
  4. Forward/backward pass via backend
  5. Sync weights to inference engines every N steps

```python
async def run_rl_training(
    backend: PyTorchTrainingBackend,
    data_buffer: DataBuffer,
    rollout_manager: AsyncRolloutManager,
    inference_engines: List[InferenceEngine],
    config: RLTrainingConfig,
    metrics_logger: Optional[MetricsLogger] = None,
) -> List[Dict[str, float]]
```

Key RL-specific helper functions:
- `compute_reward(sample)` - Extracts reward from sample metadata (1.0 if correct, 0.0 otherwise)
- `prepare_grpo_batch()` - Converts samples + rewards to training batch with advantages
- `compute_advantages()` - Calculates advantage estimates (reward - baseline)

### 2. Rollout Generation Infrastructure

**Location:** `/Users/chiraagbalu/research/rollouts/rollouts/training/rollout_gen/`

#### Synchronous Rollout Generation
- **File:** `rollout_generation.py`
- Pure functions:
  - `generate_rollout_batches()` - Generator that yields RolloutBatch objects indefinitely
  - `apply_sample_transforms()` - Apply filter_fn and reward_fn to samples
  - `convert_to_batch()` - Convert Sample list to RolloutBatch
  - `extract_sample_fields()` - Extract tokens, loss_masks, rewards
  - `compute_response_lengths()` - Calculate response length from loss_mask
  - `build_batch_metadata()` - Aggregate sample metadata into batch-level metadata

```python
def generate_rollout_batches(
    data_buffer: DataBuffer,
    config: RolloutConfig,
    **rollout_kwargs: Any,
) -> Iterator[RolloutBatch]
```

#### Asynchronous Rollout Generation (SLIME D4)
- **File:** `async_rollout_manager.py` - `AsyncRolloutManager`
- Features:
  - Parallel generation using trio (structured concurrency)
  - Dynamic over-sampling: generates N * over_sampling_factor samples, keeps best N
  - Partial sample caching for carry-over to next batch (SLIME feature!)
  - Supports both async and sync user-provided generate functions
  - Optional filtering by quality/diversity
  - Graceful abort handling with partial sample preservation

```python
@dataclass
class AsyncRolloutManager:
    data_buffer: DataBuffer
    config: RolloutConfig
    partial_samples: list[Sample] = field(default_factory=list)
    
    async def generate_batch(
        self,
        reward_fn: Optional[Callable[[Sample], float]] = None,
    ) -> RolloutBatch
```

**RolloutConfig** (frozen dataclass):
```python
@dataclass(frozen=True)
class RolloutConfig:
    batch_size: int
    n_samples_per_prompt: int = 1
    over_sampling_factor: float = 1.0
    generate_fn: Optional[Callable] = None  # User provides this
    reward_fn: Optional[Callable] = None    # Optional reward function
    filter_fn: Optional[Callable] = None    # Optional quality filter
```

#### Deprecated Rollout Manager
- **File:** `rollout_manager.py` - `RolloutManager` (DEPRECATED)
- Iterator-based API (deprecated in favor of pure functions + AsyncRolloutManager)
- Still supports state_dict() for checkpointing

### 3. Core Data Types

**Location:** `/Users/chiraagbalu/research/rollouts/rollouts/training/types.py`

#### Sample (Universal Currency)
```python
@dataclass
class Sample:
    prompt: str | list[dict[str, str]]
    response: str = ""
    tokens: list[int] = field(default_factory=list)
    loss_mask: list[float] = field(default_factory=list)  # 0=no loss, 1=compute loss
    reward: float = 0.0  # For RL training
    group_index: Optional[int] = None  # For GRPO grouping
    rollout_log_probs: Optional[list[float]] = None  # For off-policy correction
    metadata: dict[str, Any] = field(default_factory=dict)
    status: Status = Status.PENDING
```

#### RolloutBatch (Training-Ready)
```python
@dataclass
class RolloutBatch:
    tokens: list[list[int]]
    loss_masks: list[list[float]]
    rewards: list[float]
    response_lengths: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)
```

#### Training Configurations
- `SFTTrainingConfig(frozen=True)` - num_steps, batch_size, log_every, checkpoint_every
- `RLTrainingConfig(frozen=True)` - num_steps, sync_every, baseline, log_every, checkpoint_every

#### Futures (Tinker-Style Pipelining)
- `TrainFuture[T]` - Async-awaitable future using trio.Event
- `ImmediateTrainFuture[T]` - Pre-resolved future for synchronous operations
- Enables: submit work → continue → await result later

### 4. Training Backends (Protocol-Based)

**Location:** `/Users/chiraagbalu/research/rollouts/rollouts/training/backends/`

#### Backend Protocol (Minimal Surface)
```python
class TrainingBackend(Protocol):
    def forward_backward(self, batch: Dict[str, Any]) -> TrainFuture[Dict[str, float]]
    def optim_step(self) -> TrainFuture[Dict[str, float]]
    def get_weights(self) -> TrainFuture[Dict[str, Any]]
    def load_weights(self, weights: Dict[str, Any]) -> TrainFuture[None]
```

#### PyTorch Backend
- **File:** `pytorch.py` - `PyTorchTrainingBackend`
- Single-GPU training with standard PyTorch
- Features:
  - Async futures for pipelining
  - Weight version tracking (SLIME-inspired)
  - Simple checkpoint format
  - FSDP support detection

#### FSDP Backend
- **File:** `fsdp.py` - `FSDPTrainingBackend`
- Multi-GPU distributed training
- Features:
  - Fully Sharded Data Parallel
  - Mixed precision training
  - Activation checkpointing
  - Gradient norm clipping
  - Distributed checkpointing with get_model_state_dict()

#### Factory Functions (Tier 2 Convenience)
- **File:** `pytorch_factory.py`
- `create_pytorch_backend()` - One-liner backend setup
- `create_warmup_cosine_scheduler()` - LR scheduler with warmup
- Tier 1 granular functions also available:
  - `load_hf_model()`
  - `create_adamw_optimizer()`
  - `create_cross_entropy_loss()`

### 5. Loss Functions

**Location:** `/Users/chiraagbalu/research/rollouts/rollouts/training/rl_losses.py`

#### GRPO Loss (Group Relative Policy Optimization)
```python
def grpo_loss(
    logits: torch.Tensor,           # [batch, seq_len, vocab_size]
    labels: torch.Tensor,           # [batch, seq_len]
    loss_mask: torch.Tensor,        # [batch, seq_len]
    advantages: torch.Tensor,       # [batch]
) -> torch.Tensor
```
- Simplified policy gradient: -E[log_prob * advantage]
- Per-token loss masking support
- Currently used for RL training

#### PPO Loss (Placeholder)
```python
def ppo_loss(
    logits, labels, loss_mask, advantages,
    old_log_probs, clip_range=0.2
) -> torch.Tensor
```
- PPO clipped objective (not yet implemented)
- Intended for future use

### 6. Weight Synchronization (SLIME D5)

**Location:** `/Users/chiraagbalu/research/rollouts/rollouts/training/weight_sync.py`

- Stateless functions for syncing checkpoints to inference engines
- Supports both SGLang and vLLM inference servers
- Uses trio for structured concurrency (parallel sync)

**Interface:**
```python
class InferenceEngine(Protocol):
    async def update_weights_from_checkpoint(
        self,
        checkpoint_path: str,
    ) -> dict[str, Any]
```

**Implementations:**
- `SGLangEngine` - Calls /update_weights_from_disk endpoint
- `VLLMEngine` - Calls /collective_rpc with reload_weights method

**Orchestration:**
```python
async def sync_weights_to_engines(
    engines: list[InferenceEngine],
    checkpoint_path: str,
) -> list[dict[str, Any]]
```

### 7. Data Management

**Location:** `/Users/chiraagbalu/research/rollouts/rollouts/training/datasets/`

#### DataBuffer (Only Stateful Component)
- **File:** `data_buffer.py`
- Manages prompt iteration with automatic epoch advancement
- Deterministic shuffling with seed-based RNG
- Tracks epoch_id and sample_offset for checkpointing
- SLIME-style API: `get_prompts(n)` returns next n prompts

#### SFT Dataset Loading
- **File:** `sft.py`
- Pure functions for SFT sample preparation
- `compute_loss_mask()` - Creates loss mask for multi-turn conversations
- `tokenize_conversation()` - Handles chat message tokenization with span tracking
- Supports max_length truncation with validation

#### RL Dataset (Partial)
- **File:** Not fully implemented yet - needs reward model and rollout generation

### 8. Integration Points: SFT → RL

**Location:** `/Users/chiraagbalu/research/dev/integration_training/train.py`

Main orchestrator that shows how SFT and RL are connected:

```python
async def run_sft(config: Config, output_dir: Path):
    # 1. Load tokenizer + data mixture
    # 2. Create training backend (PyTorch or FSDP)
    # 3. Run SFT loop: await run_sft_training(backend, samples, config, logger)
    # 4. Save checkpoints to output_dir / "checkpoints"

async def run_rl(config: Config, output_dir: Path, source_checkpoint: str | None):
    # 1. Load from SFT checkpoint (if provided)
    # 2. Create backend (same as SFT)
    # 3. Setup rollout generation (TODO: needs implementation)
    # 4. Run RL loop: await run_rl_training(backend, buffer, rollout_mgr, engines, config)

def main():
    # Mode auto-detection: "sft", "rl", or "sft+rl"
    if mode == "sft+rl":
        # Step 1: Run SFT training
        trio.run(run_sft, config, output_dir)
        
        # Step 2: Find latest SFT checkpoint
        latest_checkpoint = sorted(checkpoint_dir.glob("step_*"))[-1]
        
        # Step 3: Run RL from that checkpoint
        trio.run(run_rl, config, output_dir, latest_checkpoint)
```

**Key Integration Points:**
1. **Model Checkpoint Flow:** SFT saves to `output_dir/checkpoints/step_NNNN/` → RL loads same checkpoint
2. **Config Inheritance:** Both SFT and RL use unified Config object with sft.* and rl.* subsections
3. **Backend Reuse:** Same PyTorch/FSDP backend code used for both training modes
4. **Tokenizer Sharing:** Same tokenizer instance used for both SFT data prep and RL rollout generation

### 9. Metrics & Logging

**Location:** `/Users/chiraagbalu/research/rollouts/rollouts/training/metrics.py`

- `MetricsLogger` (Protocol) - Minimal interface
- `JSONLLogger` - File-based metrics with JSONL format
- Used by both SFT and RL training loops
- Per-step logging with periodic checkpointing

### 10. Filtering & Quality Control (SLIME-Inspired)

**Location:** `/Users/chiraagbalu/research/rollouts/rollouts/training/filters.py`

Pre-built filters for RL sample quality:
- `check_reward_nonzero_std()` - Ensure reward variance
- `check_min_reward()` - Threshold-based filtering
- `check_response_diversity()` - Avoid duplicate responses
- `check_reasonable_length()` - Length constraints
- `check_quality_and_diversity()` - Combined check

## Design Patterns

### Casey Muratori's Immediate Mode
- Fine-grained functions (tier 1): load_hf_model(), create_adamw_optimizer()
- Convenience functions (tier 2): create_pytorch_backend()
- Both approaches available; users choose based on need

### Tiger Style
- Explicit assertions for preconditions/postconditions
- Clear error messages with context
- Explicit device handling with validation
- Explicit control flow (no callbacks)

### SLIME Patterns
- Dynamic over-sampling with partial sample caching
- Weight version tracking for distributed training
- Loss masking for multi-turn conversations
- RolloutDataSource-style data management

### Protocol-Based Design
- No inheritance hierarchy
- Minimal coupling (just type hints)
- Multiple implementations possible (SGLang, vLLM, etc.)

## Key Files Summary

```
rollouts/training/
├── loops/
│   ├── sft_loop.py          # SFT training orchestration
│   ├── rl_loop.py           # RL training orchestration (Generate → Train → Sync)
│   └── __init__.py
├── rollout_gen/
│   ├── rollout_generation.py     # Pure functions for rollout batching
│   ├── async_rollout_manager.py  # Async parallel generation with over-sampling
│   └── rollout_manager.py        # Deprecated class-based API
├── backends/
│   ├── protocol.py               # TrainingBackend protocol
│   ├── pytorch.py                # PyTorchTrainingBackend
│   ├── fsdp.py                   # FSDPTrainingBackend (multi-GPU)
│   ├── pytorch_factory.py        # Factory functions (Tier 1 & 2)
│   └── jax_backend.py, torchax_backend.py, torch_func.py
├── datasets/
│   ├── data_buffer.py            # DataBuffer (only stateful component)
│   ├── sft.py                    # SFT sample preparation
│   └── dataset_loaders.py
├── types.py                      # Sample, RolloutBatch, Configs, Futures
├── rl_losses.py                  # GRPO and PPO loss functions
├── weight_sync.py                # D5: Sync weights to inference engines
├── metrics.py                    # MetricsLogger, JSONLLogger
├── filters.py                    # Quality filtering for RL samples
├── agent_integration.py          # Convert agent trajectories to samples
└── distributed_utils.py          # Distributed training utilities
```

## Current Limitations & TODOs

1. **RL Dataset Loading:** Not fully implemented - needs reward function setup
2. **Checkpoint Loading:** RL mode shows NotImplementedError for loading SFT checkpoints
3. **Environment-Based Rewards:** Currently hardcoded to environment grading (1.0 if correct, 0.0 otherwise)
4. **Reward Models:** No separate reward model support yet (only environment-based rewards)
5. **Agent Integration:** Basic skeleton in agent_integration.py - needs completion for full end-to-end flow

## Execution Model

1. **SFT Training:**
   - Load samples from dataset mixture
   - Create backend (PyTorch or FSDP)
   - For each step: collate_batch → forward_backward → optim_step → log → checkpoint

2. **RL Training:**
   - Load SFT checkpoint (optional)
   - Create same backend
   - For each step:
     - rollout_manager.generate_batch() → parallel generation with over-sampling
     - compute_reward() → extract from metadata
     - prepare_grpo_batch() → compute advantages
     - forward_backward(rl_batch) → GRPO loss
     - optim_step()
     - Every N steps: save_checkpoint() → sync_weights_to_engines()

3. **Distributed Training (FSDP):**
   - Auto-detection in train.py
   - Re-launch with torchrun if needed
   - Each rank runs independently
   - Distributed checkpointing with barrier synchronization
   - Collective operations for weight aggregation

## Summary

The codebase implements a well-structured, principled RL training infrastructure with:
- Pure functional training loops (SFT and RL)
- Flexible rollout generation with async over-sampling
- Protocol-based backends (no inheritance)
- Integrated weight sync for distributed training
- Clear integration path from SFT → RL
- Both single-GPU and multi-GPU (FSDP) support
