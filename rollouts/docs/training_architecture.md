# Training Architecture

## Done When

We're done when we can run this:

```python
# GRPO training with nmoe backend on GLM-5 (700B-A40B)
config = GRPOConfig(
    model=ModelConfig(name="glm-5-700b"),
    trainer=TrainerConfig(backend="nmoe"),  # <-- pluggable backend
    inference=InferenceConfig(backend="sglang", tensor_parallel_size=8),
)

result = grpo_train(config, prompts, score_fn, environment_cls)
```

**Concrete tests:**

1. **FSDP backend + dense model (Qwen-0.6B):** Verify refactor didn't break anything
2. **NMOE backend + MoE model (zai-org/GLM-4.7-Flash):** Verify pluggable backends work with MoE
3. **Weight sync to SGLang:** Verify weight sync works with both backends

Run 10 steps each. No memory leak. Checkpoints save. Metrics log correctly.

Configs live in:

- `examples/training_architecture/test1_fsdp_dense_qwen_0_6b.py`
- `examples/training_architecture/test2_nmoe_moe_glm_4_7_flash.py`
- `examples/training_architecture/test3_weight_sync_fsdp_nccl_qwen_0_6b.py`
- `examples/training_architecture/test3_weight_sync_nmoe_nccl_qwen_0_6b.py`
- (Multi-node plumbing) `examples/training_architecture/multinode_fsdp_qwen_0_6b.py`

Run (from `rollouts/`):

```bash
python -m argus run --config examples/training_architecture/test1_fsdp_dense_qwen_0_6b.py
python -m argus run --config examples/training_architecture/test2_nmoe_moe_glm_4_7_flash.py
python -m argus run --config examples/training_architecture/test3_weight_sync_fsdp_nccl_qwen_0_6b.py
python -m argus run --config examples/training_architecture/test3_weight_sync_nmoe_nccl_qwen_0_6b.py
```

## Making a Backend Work

A backend is "working" when it can:
1. Load a model
2. Run forward/backward pass
3. Return weights for syncing to inference
4. Save/load checkpoints

### Testing a Backend

```bash
# 1. Unit test: does it load and run forward?
python -c "
from rollouts.training.backends.nmoe_backend import NmoeTrainingBackend, NmoeConfig
backend = NmoeTrainingBackend(model_name='zai-org/GLM-4.7-Flash', config=NmoeConfig())
# ... test forward_backward with dummy batch
"

# 2. Integration test: does it work with train()?
python -m argus run --config examples/training_architecture/test2_nmoe_moe_glm_4_7_flash.py

# 3. Weight sync test: can inference load the weights?
# Check that SGLang can load weights from backend.get_weights()
```

### NMOE Backend Integration

The `nmoe_backend.py` exists but needs:

1. **Model loading**: Use nmoe's `Transformer(cfg).cuda()` instead of HuggingFace
2. **Optimizer**: Use nmoe's `build_optimizer()` (Muon + AdamW)
3. **Forward/backward**: Call nmoe's chunked cross-entropy, not HF's
4. **Weight format**: nmoe uses different state_dict keys than HF

Reference: `/tmp/nmoe/nmoe/train.py` lines 263-310 for model/optimizer setup.

Key difference from FSDP backend:
- FSDP: `model = AutoModelForCausalLM.from_pretrained(...)`
- NMOE: `model = Transformer(cfg).cuda(); model.init_weights()`

### Adding Backend Selection to Config

```python
# In TrainerConfig (rollouts/training/configs.py)
@dataclass(frozen=True)
class TrainerConfig:
    backend: str = "fsdp"  # "fsdp" | "nmoe" | "megatron"
    ...

# In _setup_training_backend (rollouts/training/grpo.py)
if config.trainer.backend == "fsdp":
    backend = create_pytorch_backend(...)
elif config.trainer.backend == "nmoe":
    from ..training.backends.nmoe_backend import NmoeTrainingBackend
    backend = NmoeTrainingBackend(...)
elif config.trainer.backend == "megatron":
    ...
```

## Overview

The training system separates **orchestration** from **loss computation**.

SFT and GRPO are nearly identical - same forward/backward, same optimizer step.
The differences are:
1. **Loss weighting**: GRPO multiplies by advantages, SFT doesn't
2. **Data source**: GRPO generates rollouts online, SFT reads from dataset
3. **Weight sync**: GRPO syncs to inference engine, SFT doesn't need to

```python
# SFT loss
loss = cross_entropy(logits, labels) * loss_mask

# GRPO loss
loss = cross_entropy(logits, labels) * loss_mask * advantages
```

The core training loop handles:
- Batch fetching (from rollout manager OR data loader)
- Weight sync to inference engine (if present)
- Checkpointing
- Metrics logging

Loss functions are passed to the backend. Data source and weight sync are injected.

## File Structure

```
rollouts/training/
├── train.py              # Generic training loop (~200 lines)
├── losses.py             # Loss functions: grpo_loss, sft_loss (exists, keep)
├── configs.py            # TrainConfig, CheckpointConfig, etc. (exists)
├── weight_sync.py        # WeightSyncer protocol + implementations
├── grpo.py               # GRPO orchestration: grpo_train(), GRPOConfig
├── backends/
│   ├── protocol.py       # TrainingBackend protocol (exists)
│   ├── fsdp.py           # FSDP backend (exists)
│   ├── nmoe_backend.py   # NMOE backend (exists)
│   └── ...
└── types.py              # Batch, TrainFuture, etc. (exists)
```

Note: No `algorithms/` folder needed. Loss functions go in `losses.py` (already exists).
GRPO vs SFT is just different loss + different data source, not different "algorithms".

## Core Abstractions

### train() - The Training Loop

```python
# rollouts/training/train.py

def train(
    config: TrainConfig,
    backend: TrainingBackend,
    batch_iterator: Iterator[Batch],
    weight_syncer: WeightSyncer | None,
    process_batch: Callable[[int, Batch, TrainingBackend], dict],
    *,
    output_dir: Path,
    logger: Logger,
) -> TrainResult:
    """Algorithm-agnostic training loop.

    Args:
        config: Training configuration (num_steps, checkpoint_every, etc.)
        backend: Training backend (FSDP, NMOE, etc.) - owns model/optimizer
        batch_iterator: Yields batches (from rollout manager or data loader)
        weight_syncer: Syncs weights to inference engine (None for SFT/pretrain)
        process_batch: Algorithm-specific step function
        output_dir: Where to save checkpoints and logs
        logger: Structured logger for wide events

    Returns:
        TrainResult with final metrics and checkpoint path
    """
    for step in range(config.num_steps):
        batch = next(batch_iterator)

        # Algorithm-specific processing
        step_metrics = process_batch(step, batch, backend)

        # Weight sync (if doing RL with separate inference)
        if weight_syncer and should_sync(step, config):
            weight_syncer.sync()

        # Checkpoint
        if should_checkpoint(step, config):
            backend.checkpoint(output_dir / f"step_{step}")

        # Log wide event
        log_step(logger, step, step_metrics, config)

    return TrainResult(...)
```

### TrainingBackend Protocol

```python
# rollouts/training/backends/protocol.py (exists, unchanged)

class TrainingBackend(Protocol):
    def forward_backward(self, batch: dict[str, Any]) -> TrainFuture[dict]:
        """Compute loss and gradients."""
        ...

    def optim_step(self) -> TrainFuture[dict]:
        """Apply gradients."""
        ...

    def get_weights(self) -> TrainFuture[dict[str, Tensor]]:
        """Get state dict for weight sync."""
        ...

    def checkpoint(self, path: Path) -> None:
        """Save checkpoint."""
        ...
```

### WeightSyncer Protocol

```python
# rollouts/training/weight_sync.py

class WeightSyncer(Protocol):
    """Syncs weights from trainer to inference engine."""

    def sync(self) -> None:
        """Push current weights to inference."""
        ...

    def close(self) -> None:
        """Cleanup (destroy process groups, etc.)."""
        ...


class NCCLWeightSyncer:
    """GPU-to-GPU broadcast via NCCL.

    Fast, but requires:
    - Same TP on trainer and inference, OR
    - Trainer TP=1 (gather then broadcast)
    """

    def __init__(
        self,
        backend: TrainingBackend,
        inference_engine: InferenceEngine,
        config: WeightSyncConfig,
    ):
        self.backend = backend
        self.inference = inference_engine
        self.version = 0
        self._setup_nccl_group()

    def sync(self) -> None:
        weights = self.backend.get_weights().result()
        self._broadcast(weights)
        self.version += 1

    def close(self) -> None:
        dist.destroy_process_group(self._group)


class FilesystemWeightSyncer:
    """Checkpoint-based sync via shared filesystem.

    Slow, but works for:
    - Mismatched TP between trainer and inference
    - Cross-node without NCCL connectivity
    """

    def __init__(
        self,
        backend: TrainingBackend,
        inference_engine: InferenceEngine,
        sync_dir: Path,
    ):
        ...

    def sync(self) -> None:
        path = self.sync_dir / f"weights_v{self.version}"
        self.backend.checkpoint(path)
        self.inference.load_weights(path)
        self.version += 1
```

### Loss Functions (in losses.py)

```python
# rollouts/training/losses.py (exists, simplified view)

def grpo_loss(logits, batch):
    """GRPO loss: cross_entropy * loss_mask * advantages."""
    ce = cross_entropy(logits, batch["labels"], reduction="none")
    return (ce * batch["loss_mask"] * batch["advantages"]).sum() / batch["loss_mask"].sum()

def sft_loss(logits, batch):
    """SFT loss: cross_entropy * loss_mask (no advantages)."""
    ce = cross_entropy(logits, batch["labels"], reduction="none")
    return (ce * batch["loss_mask"]).sum() / batch["loss_mask"].sum()
```

The backend receives the loss function at construction time:

```python
backend = create_backend(config, loss_fn=grpo_loss)  # or sft_loss
```

## Usage Examples

### GRPO Training

```python
# examples/rl/reverse_text/train.py

from rollouts.training.train import train
from rollouts.training.losses import grpo_loss
from rollouts.training.weight_sync import NCCLWeightSyncer

def grpo_train(config: GRPOConfig, prompts: list, score_fn: Callable):
    # Setup - loss function passed to backend
    backend = create_backend(config.trainer, loss_fn=grpo_loss)
    inference = create_inference_engine(config.inference)
    weight_syncer = NCCLWeightSyncer(backend, inference, config.weight_sync)

    # Data comes from rollout manager (online generation)
    rollout_manager = PipelinedRolloutManager(
        prompts=prompts,
        score_fn=score_fn,
        inference=inference,
        config=config.rollout,
    )

    # Train
    result = train(
        config=config.train,
        backend=backend,
        batch_iterator=rollout_manager,
        weight_syncer=weight_syncer,
        output_dir=config.output_dir,
    )

    # Cleanup
    weight_syncer.close()
    inference.shutdown()

    return result
```

### SFT Training (no rollouts, no weight sync)

```python
# examples/sft/train.py

from rollouts.training.train import train
from rollouts.training.losses import sft_loss

def sft_train(config: SFTConfig, dataset: Dataset):
    # Setup - different loss function, same backend
    backend = create_backend(config.trainer, loss_fn=sft_loss)

    # Data comes from data loader (offline dataset)
    data_loader = DataLoader(dataset, batch_size=config.batch_size)

    result = train(
        config=config.train,
        backend=backend,
        batch_iterator=iter(data_loader),
        weight_syncer=None,  # No inference engine for SFT
        output_dir=config.output_dir,
    )

    return result
```

The only differences:
1. `grpo_loss` vs `sft_loss`
2. `rollout_manager` vs `data_loader`
3. `weight_syncer` vs `None`

## Migration Plan

### Phase 1: Create train.py with generic loop (1 hour)

1. Create `rollouts/training/train.py` with generic `train()` function
2. Extract the core loop from `_grpo_train_async` (lines 1012-1207)
3. `train()` takes: `backend`, `batch_iterator`, `weight_syncer`, `config`
4. Run existing GRPO test to verify nothing broke

### Phase 2: Merge the two step functions (30 min)

1. Delete `_process_training_step_no_sync`
2. Add `weight_syncer: WeightSyncer | None` param to `_process_training_step`
3. Weight sync logic: `if weight_syncer: weight_syncer.sync()`
4. Run tests

### Phase 3: Clean up weight sync (30 min)

1. Add `WeightSyncer` protocol to `weight_sync.py`
2. Rename `PipelineWeightSyncManager` to `NCCLWeightSyncer`
3. Existing code already has the right structure, just needs protocol
4. Run tests

### Phase 4: Move TITO helpers out (30 min)

1. Move `_trajectory_to_samples_*` to `examples/rl/tito/` or `rollouts/training/tito.py`
2. These are task-specific, not core training infrastructure
3. Run tests

## Reference Implementations

Look at these for patterns:

- **NMOE** (preferred style): `/tmp/nmoe/nmoe/train.py` - flat train() function, ~880 lines
- **Slime weight sync**: `/tmp/slime/slime/backends/fsdp_utils/update_weight_utils.py` - UpdateWeight protocol
- **Miles actor**: `/tmp/miles/miles/backends/fsdp_utils/actor.py` - FSDPTrainRayActor

## Design Principles

From `/Users/chiraagbalu/research/docs/code_style/`:

Key docs to reference:
- `code_philosophy_reference.md` - classes vs functions, when to use what
- `logging_sucks.md` - wide events pattern for logging
- `tiger_style.md` - fail fast, assertions, explicit control flow

1. **Classes for resources, functions for computation**
   - `TrainingBackend`: class (owns model, optimizer, device)
   - `WeightSyncer`: class (owns NCCL group)
   - `compute_advantages()`: function (pure math)
   - `train()`: function (orchestrates objects)

2. **Protocol over inheritance**
   - `TrainingBackend(Protocol)` not `class BaseTrainingBackend`
   - Implementations don't inherit, they just match the interface

3. **Inject algorithms, don't hardcode**
   - `train()` takes `process_batch` as a parameter
   - Easy to add PPO, DPO, etc. without touching train.py

4. **Wide events for logging**
   - One log line per step with all context
   - Memory stats, timing, metrics in single event
