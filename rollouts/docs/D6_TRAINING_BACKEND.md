# D6 Training Backend Ladder

This doc describes the intended evolution of the training backend interface and implementations ("D6").

The core goal is to make **pretrain / LoRA / SFT / RL** share the same *training backend surface* while still
allowing different execution strategies (plain PyTorch, FSDP, torch.func functional training, JAX, etc).

## Definitions

### Shared Surface Area (What Callers Rely On)

Our orchestration code (SFT/GRPO/train loops) should depend on the `TrainingBackend` protocol:

- `forward_backward(batch) -> metrics`
- `optim_step() -> step_metrics`
- `get_weights() -> state_dict` (for inference sync + checkpointing)
- `load_weights(state_dict)`

The surface is intentionally small: it is the "waist" that lets us swap implementations.

Backends may also provide checkpoint helpers (`save_checkpoint`, `load_checkpoint`), but the orchestration code
should not require them if `get_weights/load_weights` + an external checkpointer is sufficient.

### What "Works With Our Training Setup" Means

A model/backbone is "compatible" with our training stack if:

- It can be represented as an `nn.Module` for integration with HF/PEFT/FSDP and for producing a `state_dict`.
- The training step can be expressed as a function of `(weights, batch) -> (loss, metrics)` plus an optimizer update.
- Weight sync to the inference engine can consume the produced weights (either a HF-shaped `state_dict`, or we have
  an adapter that converts keys).

This is why we keep an `nn.Module` surface even if we train "functionally" internally.

## D6v1: `PyTorchTrainingBackend` (Current)

Location: `rollouts/training/backends/pytorch.py`

Properties:

- Stateful `nn.Module` + `torch.optim.Optimizer`
- Supports microbatching, gradient accumulation, clipping, checkpointing
- Designed to be compatible with FSDP and PEFT-style wrapping (LoRA)

This backend is the default because it aligns with the PyTorch ecosystem and is easiest to debug.

## D6v2: `TorchFuncTrainingBackend` (Planned)

Location: `rollouts/training/backends/torch_func.py` (currently stub)

Intent:

- Keep **`nn.Module` as the shared interoperability surface**
  - HF loading, PEFT/LoRA wrapping, FSDP compatibility, `state_dict` conventions, SGLang reload.
- Use a **functional core** for the *step*:
  - represent parameters/buffers explicitly (`params`, `buffers`)
  - compute gradients via `torch.func` transforms
  - apply optimizer updates functionally (likely via `torchopt` or an internal functional optimizer)

### The Middle Ground: `nn.Module` Surface, Functional Core

We choose this because it buys us two things at once:

- **Interoperability**: most of our codepaths (checkpointing, LoRA injection/merge, inference reload) assume an
  `nn.Module` and/or an HF-shaped `state_dict`.
- **Mathematical clarity / testability**: a functional step is closer to "pure math":
  - easy to unit test `(params, batch) -> loss/grad/metrics`
  - easy to reason about state updates (optimizer state is explicit)
  - easier to share concepts with a future JAX backend

The `nn.Module` instance becomes a *template* used by `torch.func.functional_call(model, (params, buffers), ...)`.

### When To Move More Functional

Signals that we should push more logic into the functional core:

- We want `vmap`/per-example grads, higher-order grads, or other program transforms.
- We want deterministic/replayable training steps and easier correctness testing.
- We want to share more code with a JAX backend (same step signature, same optimizer state threading).
- We are repeatedly fighting implicit mutation (optimizer/model side effects) in debugging.

### When To Move More Toward `nn.Module` Coupling

Signals that we should lean into module-centric training:

- We need deep integration with FSDP/DTensor/sharded optimizers that expect in-place param updates.
- We need "standard PyTorch" tooling (hooks, profilers, existing third-party wrappers) more than functional transforms.
- We are bottlenecked on implementation complexity rather than conceptual cleanliness.

### Implementation Sketch (Non-Normative)

The intended shape of D6v2:

1. **Extract functional state** from a module:
   - `params`: a tree/tuple of tensors matching module parameters
   - `buffers`: a dict/tree matching module buffers
2. **Define** `loss_fn(params, buffers, batch) -> (loss, metrics)`
3. **Compute grads** with `torch.func.grad_and_value` (or similar) over `params`
4. **Apply updates** with a functional optimizer:
   - `(updates, opt_state) = optimizer.update(grads, opt_state, params)`
   - `params = apply_updates(params, updates)`
5. **Expose weights** as a `state_dict`:
   - convert `(params, buffers)` back into a dict keyed like `model.state_dict()`
6. **Load weights** by converting a `state_dict` into `(params, buffers)`

This keeps compatibility with the rest of the system, while making the "math core" explicit.

## D6v3: JAX Backend (Planned)

Location: `rollouts/training/backends/jax_backend.py` (currently stub)

The intent is to reuse the D6v2 step signature as closely as possible:

- explicit params
- explicit optimizer state
- pure step functions
- standard "get_weights/load_weights" boundary for inference sync and checkpointing

## D6v4: TorchAX Backend (Planned)

Location: `rollouts/training/backends/torchax_backend.py` (currently stub)

TorchAX is a potential intermediate: it may allow us to run a "JAX-y" step while preserving more PyTorch
ecosystem compatibility. This is exploratory.

## Relationship To Pretraining ("Nano Trainer")

The `rollouts/pretrain/` codepath is already close to the D6v2 mindset: explicit weights, explicit update loops,
minimal coupling to framework objects.

If we want to "unify pretraining with the RL trainer", the cleanest path is:

- Keep one orchestration loop shape (state threading, checkpointing, logging)
- Swap only the *loss* and *batch source* (dataset vs rollouts)
- Use a single backend surface (`TrainingBackend`) regardless of whether we are doing pretrain, SFT, or RL

That unification is easiest if we keep the module surface stable for all model families, while allowing the
implementation to be functional internally.

