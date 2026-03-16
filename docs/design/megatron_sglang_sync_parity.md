# Megatron x SGLang Sync Parity Checklist

This is the current parity checklist for the `rollouts` Megatron -> SGLang
NCCL hot-update path against the `miles` baseline we are actually running in
Modal.

## Scope

- Trainer-side reference:
  [`/tmp/miles/miles/backends/megatron_utils/update_weight/update_weight_from_distributed.py`](/tmp/miles/miles/backends/megatron_utils/update_weight/update_weight_from_distributed.py)
- Shared process-group helper:
  [`/tmp/miles/miles/utils/distributed_utils.py`](/tmp/miles/miles/utils/distributed_utils.py)
- Engine wrapper reference:
  [`/tmp/miles/miles/backends/sglang_utils/sglang_engine.py`](/tmp/miles/miles/backends/sglang_utils/sglang_engine.py)
- Receiver-side patches:
  [`/tmp/miles/docker/patch/v0.5.0rc0-cu126/sglang.patch`](/tmp/miles/docker/patch/v0.5.0rc0-cu126/sglang.patch)
  [`/tmp/miles/docker/patch/v0.5.5.post1/sglang.patch`](/tmp/miles/docker/patch/v0.5.5.post1/sglang.patch)
  [`/tmp/miles/docker/patch/v0.5.6/sglang.patch`](/tmp/miles/docker/patch/v0.5.6/sglang.patch)

Current runtime image:
- [`examples/rl/base_config.py`](/Users/chiraagbalu/research/rollouts/examples/rl/base_config.py)
  pins `radixark/miles:nightly-dev-20260113a`

Current failure witness:
- [`run_20260316-073301/run.jsonl`](/Users/chiraagbalu/research/rollouts/results/rl/run_20260316-073301/run.jsonl)
- first witness tensor `model.norm.weight`
- sender-side `ncclInternalError`
- SGLang-side `Cuda failure 'invalid argument'`

## High-Level Read

We are no longer missing the Megatron export denotation.

The remaining gap is in the hot-update runtime contract:

- trainer-side publication orchestration
- rollout-engine concurrency control
- patched SGLang communicator/update behavior

`miles` owns that as one coherent state machine.
We still have a hybrid:

- our control plane
- our sender wrapper
- their patched SGLang receiver image

That is the main reason they do not hit the same failure mode.

## Checklist

### Runtime Baseline

- `miles` uses a pinned image + pinned SGLang patch stack.
  Status: matched for Modal witness runs
  Evidence:
  [`examples/rl/base_config.py`](/Users/chiraagbalu/research/rollouts/examples/rl/base_config.py)

- Exact SGLang patch level inside the live image is not yet explicitly recorded
  in our run journal.
  Status: missing observability

### Trainer-Side NCCL Group Construction

- `miles` uses a dedicated helper based on PyTorch rendezvous +
  `PrefixStore(group_name, ...)`.
  Status: matched
  Evidence:
  [`distributed_utils.py`](/tmp/miles/miles/utils/distributed_utils.py)
  [`weight_sync.py`](/Users/chiraagbalu/research/rollouts/rollouts/inference/weight_sync.py)

- `miles` resolves a real node IP for the weight-sync group.
  Status: matched
  Evidence:
  [`update_weight_from_distributed.py`](/tmp/miles/miles/backends/megatron_utils/update_weight/update_weight_from_distributed.py)
  [`megatron_worker.py`](/Users/chiraagbalu/research/rollouts/rollouts/training/megatron_worker.py)

- Ambient `MASTER_ADDR=127.0.0.1` is ignored for our weight-sync rendezvous.
  Status: matched

### Trainer-Side Export + Publication Shape

- `miles` uses a backend-local Megatron export/update layer before publication.
  Status: matched enough for dense witness path
  Evidence:
  [`export_runtime.py`](/Users/chiraagbalu/research/rollouts/rollouts/training/backends/megatron/export_runtime.py)
  [`inference_export.py`](/Users/chiraagbalu/research/rollouts/rollouts/training/backends/megatron/inference_export.py)

- `miles` only lets the PP/DP/TP source rank own actual publication side
  effects; other ranks only participate in collectives.
  Status: matched
  Evidence:
  [`update_weight_from_distributed.py`](/tmp/miles/miles/backends/megatron_utils/update_weight/update_weight_from_distributed.py)
  [`megatron_worker.py`](/Users/chiraagbalu/research/rollouts/rollouts/training/megatron_worker.py)

- `miles` sends metadata first, then ordered tensor broadcasts.
  Status: matched

- `miles` uses plain `dist.broadcast(param.data, ...)`.
  Status: matched now
  Evidence:
  [`update_weight_from_distributed.py`](/tmp/miles/miles/backends/megatron_utils/update_weight/update_weight_from_distributed.py)
  [`megatron_worker.py`](/Users/chiraagbalu/research/rollouts/rollouts/training/megatron_worker.py)

### Trainer-Side Update Lifecycle

- `miles` does:
  `pause_generation -> flush_cache -> broadcast -> continue_generation`
  Status: matched
  Evidence:
  [`update_weight_from_distributed.py`](/tmp/miles/miles/backends/megatron_utils/update_weight/update_weight_from_distributed.py)
  [`megatron_worker.py`](/Users/chiraagbalu/research/rollouts/rollouts/training/megatron_worker.py)

- `miles` guards each bucket broadcast with a `rollout_engine_lock`.
  Status: missing
  Why it matters:
  The lock makes the update channel a single-owner effect and is explicitly
  described as deadlock prevention in `miles`.

- `miles` performs update in bucketed chunks.
  Status: partially matched
  Notes:
  We now have explicit payloads and witness limiting, but we do not yet mirror
  the same bucket/lock structure as `miles`.

### Receiver-Side SGLang Contract

- `miles` patched SGLang routes `init_weights_update_group` through
  `*_communicator(...)`.
  Status: matched indirectly via image

- `miles` patched SGLang routes `update_weights_from_distributed` under
  `model_update_lock.writer_lock`.
  Status: matched indirectly via image
  Evidence:
  [`v0.5.0rc0-cu126/sglang.patch`](/tmp/miles/docker/patch/v0.5.0rc0-cu126/sglang.patch)

- Newer `miles` patches add `PyNcclCommunicator.nccl_pause()` /
  `nccl_resume()` and use them when pausing CUDA-graph-related memory
  occupation.
  Status: not exercised by our codepath
  Evidence:
  [`v0.5.5.post1/sglang.patch`](/tmp/miles/docker/patch/v0.5.5.post1/sglang.patch)
  [`v0.5.6/sglang.patch`](/tmp/miles/docker/patch/v0.5.6/sglang.patch)

- Our current SGLang launch env does not set `AMEM_ENABLE=1`.
  Status: missing compared with the newer pause/resume path
  Evidence:
  [`weight_sync.py`](/Users/chiraagbalu/research/rollouts/rollouts/training/weight_sync.py)

This is the most plausible still-missing runtime effect.

### Runtime Topology

- `miles` uses Ray actors for training and rollout engines.
  Status: intentionally different

- Our stack uses:
  - Argus / Modal launcher
  - miniray worker transport
  - our own sender abstraction
  - `miles` image for the receiver
  Status: different by design

This difference may matter even if all local helper functions look similar,
because communicator ownership and concurrency semantics are tied to process
topology, not just function signatures.

### Control-Plane Observability

- We now have a 1-tensor witness before rollout.
  Status: better than before
  Evidence:
  [`grpo.py`](/Users/chiraagbalu/research/rollouts/rollouts/training/grpo.py)

- Witness failure does not yet emit a clean terminal run event.
  Status: missing
  Why it matters:
  The run semantically fails, but the journal does not promptly transition to a
  terminal state. That makes `argus monitor --wait-for-event` less useful than
  it should be.

## Current Best Explanation

The remaining failure is probably not:

- Megatron export
- rank ownership
- loopback rendezvous
- sender async/sync semantics

The strongest remaining hypotheses are:

1. We still lack the newer SGLang-side communicator pause/resume effect
   (`AMEM_ENABLE=1` + `nccl_pause()/nccl_resume()` patches).
2. We still lack `miles`' rollout-engine lock semantics on the trainer side.
3. Our hybrid runtime topology is violating a communicator ownership invariant
   that does not exist inside the native `miles` Ray actor graph.

## Next Patches To Try

1. Add explicit parity for the `miles` rollout-engine lock semantics in our
   Megatron hot-update path.

2. Run one experiment with `AMEM_ENABLE=1` in the SGLang launch env, since the
   newer `miles` receiver patches make that the gate for `nccl_pause()` /
   `nccl_resume()`.

3. Record the exact SGLang patch/version surface in the run journal so we know
   which receiver contract is actually live.

4. Emit a structured `training_preflight_weight_sync_witness_failed` event and
   force orderly shutdown when the witness fails.

## Notes

- I could not verify `slime` locally in this pass because there is no usable
  `/tmp/slime` checkout in this environment.
- The checklist above is therefore grounded in the `miles` code and patches
  that our Modal image is explicitly pinned to.
