# Tinker API Alignment Plan

## Goal

Align rollouts training API with Tinker's design: loss function passed per-call, not baked into backend at construction.

## Why

1. **Explicit > implicit** - reading code shows exactly what loss is used
2. **Flexibility** - different losses per step (A/B testing, curriculum)
3. **Tinker parity** - enables future remote training backend
4. **Code style compliance** - "make implicit assumptions explicit"

## Current State

```python
# Construction: loss baked in
loss_fn = _make_loss_fn(config.trainer, grpo_loss, grpo_loss_clipped, grpo_loss_masked)
backend = create_pytorch_backend(..., loss_fn=loss_fn)

# Usage: loss implicit
fb_future = backend.forward_backward(training_batch)  # Uses self.loss_fn
```

## Target State

```python
# Construction: no loss
backend = create_pytorch_backend(...)

# Usage: loss explicit
loss_fn = _make_loss_fn(config.trainer, grpo_loss, grpo_loss_clipped, grpo_loss_masked)
fb_future = backend.forward_backward(training_batch, loss_fn=loss_fn)
```

## Files to Change

### 1. Protocol (`rollouts/training/backends/protocol.py`)

```python
# Before
def forward_backward(self, batch: dict[str, Any]) -> TrainFuture[dict[str, float]]: ...

# After
def forward_backward(
    self,
    batch: dict[str, Any],
    loss_fn: Callable[[torch.Tensor, dict], torch.Tensor | tuple[torch.Tensor, dict]],
) -> TrainFuture[dict[str, float]]: ...
```

### 2. PyTorch Backend (`rollouts/training/backends/pytorch.py`)

- Remove `loss_fn` from `__init__` and dataclass fields
- Add `loss_fn` parameter to `forward_backward()`
- Update line 211: `loss_result = loss_fn(logits, micro_batch)`
- Update docstrings

### 3. FSDP Backend (`rollouts/training/backends/fsdp.py`)

- Same pattern as pytorch.py
- Remove `loss_fn` from construction
- Add to `forward_backward()`

### 4. Factory (`rollouts/training/backends/pytorch_factory.py`)

- Remove `loss_fn` parameter from `create_pytorch_backend()`
- Update docstrings/examples

### 5. GRPO Trainer (`rollouts/training/grpo.py`)

Lines 238-249, 458, 576:
```python
# Before
loss_fn = _make_loss_fn(config.trainer, ...)
backend = create_pytorch_backend(..., loss_fn=loss_fn)
...
fb_future = backend.forward_backward(training_batch)

# After
backend = create_pytorch_backend(...)
loss_fn = _make_loss_fn(config.trainer, ...)
...
fb_future = backend.forward_backward(training_batch, loss_fn=loss_fn)
```

### 6. Training Loops

**sft_loop.py:67**
```python
# Before
fwd_metrics = await backend.forward_backward(batch).result()

# After - need to pass sft_loss
from ..losses import sft_loss
fwd_metrics = await backend.forward_backward(batch, loss_fn=sft_loss).result()
```

**rl_loop.py:81**
```python
# Similar - caller must provide loss_fn or accept it as parameter
```

### 7. Stub Backends (optional, for consistency)

- `jax_backend.py`
- `torch_func.py`
- `torchax_backend.py`

Update signatures to match protocol.

## Migration Checklist

- [ ] Update `TrainingBackend` protocol
- [ ] Update `PyTorchTrainingBackend.forward_backward()` signature
- [ ] Remove `loss_fn` from `PyTorchTrainingBackend.__init__`
- [ ] Update `FSDPTrainingBackend` similarly
- [ ] Update `create_pytorch_backend()` factory
- [ ] Update `grpo.py` - both sync and async paths
- [ ] Update `sft_loop.py`
- [ ] Update `rl_loop.py`
- [ ] Update stub backends for consistency
- [ ] Update docstrings and examples
- [ ] Run tests

## Future: forward_backward_custom

After this change, adding Tinker's `forward_backward_custom()` is trivial - it's the same as `forward_backward()` since we already accept arbitrary callables.

Could add string-based dispatch later if wanted:
```python
def forward_backward(self, batch, loss_fn: str | Callable):
    if isinstance(loss_fn, str):
        loss_fn = LOSS_REGISTRY[loss_fn]  # "grpo" -> grpo_loss
    ...
```

## Other Gaps (Separate PRs)

1. **DRO loss** - add to `losses.py`
2. **CISPO loss** - add to `losses.py`
3. **Declarative metrics** - suffix-based reduction (`metric:mean`, `metric:sum`)
4. **Checkpoint lifecycle** - list/delete/publish/TTL API
5. **Unify TrainerConfig** - two classes with same name, confusing
