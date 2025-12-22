# Functional Training Refactor

**DRI:**
**Claude:** (this conversation)

## Context

Replace stateful `TrainingBackend` class with pure functions. The current backend is a convenience bundle, not a necessity - inference engines only need a checkpoint path (string), not backend state.

## Out of Scope
- Distributed training changes (FSDP wrapping stays the same)
- Inference engine refactoring
- miniray integration (separate design)

## Solution

**Input:** Model, optimizer, batch, config passed explicitly to each function
**Output:** Metrics dict from each operation, checkpoint path from save

## Usage

### Current (Stateful)
```python
backend = PyTorchTrainingBackend(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    checkpoint_dir=checkpoint_dir,
    device=device,
)

for step in range(num_steps):
    fwd_metrics = await backend.forward_backward(batch).result()
    opt_metrics = await backend.optim_step().result()

    if step % sync_every == 0:
        ckpt_path = await backend.save_checkpoint(step, metrics)
        await sync_weights_to_engines(engines, str(ckpt_path))
```

### Proposed (Functional)
```python
for step in range(num_steps):
    fwd_metrics = forward_backward(model, optimizer, batch, loss_fn, device, num_minibatches)
    opt_metrics = optim_step(optimizer, max_grad_norm)

    if step % sync_every == 0:
        ckpt_path = save_checkpoint(model, optimizer, step, checkpoint_dir, metrics)
        await sync_weights_to_engines(engines, str(ckpt_path))
```

---

## Details

### Why This Works

**Key insight from nmoe/miles comparison:**

| Codebase | Approach | Why |
|----------|----------|-----|
| **nmoe** | Pure functions, state as local vars | SPMD - all processes identical |
| **miles** | Stateful Ray actors | Heterogeneous roles (rollout vs training) |
| **rollouts** | Stateful backend class | ??? |

Our RL case needs both inference + training, so some boundary exists. But that boundary is **checkpoint path on disk**, not method calls:

```python
# What inference engines actually need:
await engine.update_weights_from_checkpoint("/checkpoints/step_0100")
# Just a string! No backend.get_weights(), no state sync.
```

### State Elimination Analysis

| Current State | Used By | Alternative |
|---------------|---------|-------------|
| `model` | forward_backward, get_weights, save_checkpoint | Pass as argument |
| `optimizer` | forward_backward, optim_step, save_checkpoint | Pass as argument |
| `loss_fn` | forward_backward | Pass as argument |
| `device` | forward_backward (batch move) | Pass as argument |
| `checkpoint_dir` | save_checkpoint | Pass as argument |
| `weight_version` | save_checkpoint metadata | Use step number |
| `current_step` | optim_step return | Caller already knows |
| `_poisoned` | All methods | Just let it crash (nmoe style) |

### Async Considerations

**Current reality:** TrainFuture resolves immediately (no real async)

```python
# pytorch.py:240-242
future = TrainFuture(operation="forward_backward")
future.set_result(result)  # <-- Immediate!
return future
```

**Truly async operations:**
- `save_checkpoint` - disk I/O (use `trio.to_thread.run_sync`)
- `sync_weights_to_engines` - HTTP calls (already async)

**Proposed:**
- `forward_backward()` - sync (GPU-bound, not I/O bound)
- `optim_step()` - sync (GPU-bound)
- `save_checkpoint()` - async wrapper optional, or just sync
- Weight sync - stays async

### Implementation

```python
# rollouts/training/functional.py

def forward_backward(
    model: nn.Module,
    optimizer: Optimizer,
    batch: dict[str, torch.Tensor],
    loss_fn: Callable,
    device: torch.device,
    num_minibatches: int = 1,
) -> dict[str, float]:
    """Compute loss and gradients. Pure function."""
    optimizer.zero_grad()

    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    batch_size = batch["input_ids"].shape[0]
    micro_batch_size = batch_size // num_minibatches

    total_loss = 0.0
    for i in range(num_minibatches):
        start, end = i * micro_batch_size, (i + 1) * micro_batch_size
        micro_batch = {k: v[start:end] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        output = model(micro_batch["input_ids"])
        logits = output.logits if hasattr(output, "logits") else output
        loss = loss_fn(logits=logits, batch=micro_batch)

        scaled_loss = loss / num_minibatches
        scaled_loss.backward()
        total_loss += loss.item()

    grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

    return {
        "loss": total_loss / num_minibatches,
        "grad_norm": grad_norm,
        "num_minibatches": num_minibatches,
    }


def optim_step(
    optimizer: Optimizer,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    """Clip gradients and step optimizer. Pure function."""
    grad_norm_clipped = None
    if max_grad_norm is not None:
        grad_norm_clipped = torch.nn.utils.clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g["params"]],
            max_grad_norm,
        )

    optimizer.step()

    result = {"lr": optimizer.param_groups[0]["lr"]}
    if grad_norm_clipped is not None:
        result["grad_norm_clipped"] = float(grad_norm_clipped)
    return result


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
    checkpoint_dir: Path,
    metrics: dict[str, float] = {},
    rank: int = 0,
) -> Path:
    """Save checkpoint to disk. Returns checkpoint path."""
    ckpt_dir = checkpoint_dir / f"step_{step:04d}"

    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / "pytorch_model.bin")
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.bin")

        metadata = {
            "step": step,
            "timestamp": time.time(),
            "metrics": metrics,
        }
        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    if dist.is_initialized():
        dist.barrier()

    return ckpt_dir
```

### Reference: nmoe's Approach

From `nmoe/zero2.py:1-8`:
```python
"""ZeRO-2 for replicated (dense) parameters.

Elegant minimal implementation:
- step_dense_adamw: RS(AVG) grads -> AdamW shard update -> AG params

State lives in caller-provided dict to keep this module stateless.
Works seamlessly for single GPU (no-op) and multi-GPU (ZeRO-2).
"""
```

From `nmoe/train.py:36-116`:
```python
def train(cfg: Config):
    rank, world = runtime.init(cfg.seed)

    # State as local variables, not class attributes
    model = Transformer(cfg).cuda()
    optimizer, dense_groups = build_optimizer(model, cfg)
    zero2_state = {}  # Passed to step(), not hidden in class

    for step_num in range(start_step, cfg.steps):
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        model.zero_grad(set_to_none=True)
        loss.backward()
        step(model, optimizer, dense_groups, zero2_state, cfg, world)  # Explicit state
```

### Migration Path

1. **Add functional.py** with pure functions alongside existing backend
2. **Update rl_loop.py** to use functional style (keep backend for backwards compat)
3. **Update grpo.py** entry point
4. **Deprecate TrainingBackend** once stable
5. **Delete TrainingBackend** in future version

### Open Questions

- [ ] Keep `TrainingBackend` as thin wrapper over functions for backwards compat?
- [ ] How to handle FSDP state dict gathering (currently in backend)?
- [ ] Should `save_checkpoint` be async or is sync fine?
- [ ] How does this interact with miniray? (separate design doc)

### Files

**Read:**
- `/tmp/nmoe/nmoe/train.py:36-160` - Functional training loop
- `/tmp/nmoe/nmoe/opt.py:289-332` - `step()` as pure function
- `/tmp/nmoe/nmoe/zero2.py:97-208` - Stateless ZeRO-2
- `/tmp/miles/train.py` - Ray actor orchestration (contrast)

**Modify:**
- `rollouts/training/functional.py` - New file with pure functions
- `rollouts/training/loops/rl_loop.py` - Use functional style
- `rollouts/training/loops/sft_loop.py` - Use functional style
- `rollouts/training/grpo.py` - Update entry point

**Eventually Delete:**
- `rollouts/training/backends/pytorch.py` - Or keep as thin wrapper
- `rollouts/training/backends/protocol.py` - Protocol becomes unnecessary
