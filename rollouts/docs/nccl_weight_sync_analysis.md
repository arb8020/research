# NCCL Weight Sync Analysis

## The Problem

Training runs fail at step 1 with:
```
ncclUnhandledCudaError: Call to CUDA function failed.
```

This happens during `update_weights_from_distributed`, **not** during NCCL group initialization.

## Root Cause

**Rollouts runs HTTP notification and NCCL broadcast concurrently, causing a race condition.**

In `rollouts/training/backends/pytorch.py:1006-1010`:
```python
async with trio.open_nursery() as nursery:
    for endpoint in self._nccl_inference_endpoints:
        nursery.start_soon(notify_inference_server, endpoint)  # HTTP POST
    nursery.start_soon(broadcast_weights)  # NCCL broadcasts
```

Both tasks start at the same time. But:
- The HTTP endpoint (`/update_weights_from_distributed`) on SGLang **blocks** until all NCCL broadcasts complete
- If the trainer starts broadcasting before SGLang has received the HTTP request and started waiting for broadcasts (or vice versa), NCCL collectives fail

Additionally, rollouts uses **synchronous** broadcasts while Miles and SGLang both use `async_op=True`.

## Reference Implementations

### Miles (radixark/miles)

Location: `/tmp/miles/miles/backends/fsdp_utils/update_weight_utils.py:231-258`

```python
# FIRST: Send Ray remote calls to all engines (async, non-blocking)
refs = [
    engine.update_weights_from_distributed.remote(
        names=..., dtypes=..., shapes=..., group_name=...,
    )
    for engine in self.rollout_engines
]

# SECOND: Do broadcasts (with async_op=True)
handles = []
for _name, param in named_tensors:
    param_data = param.data.contiguous()
    handles.append(dist.broadcast(param_data, 0, group=self._model_update_groups, async_op=True))

# THIRD: Wait for all broadcasts
for handle in handles:
    handle.wait()

# FOURTH: Wait for Ray refs to confirm SGLang received everything
ray.get(refs)
```

Key points:
1. Fire-and-forget remote calls via `engine.update_weights_from_distributed.remote()` - these don't block
2. Async NCCL broadcasts with `async_op=True`
3. Wait for broadcast handles to complete
4. Then wait for remote refs to confirm SGLang finished

### QED-Nano (PrimeIntellect)

Location: `/tmp/QED-Nano/training/pipelinerl/finetune_loop.py:200-210`

```python
# FIRST: Send HTTP requests (async via ThreadPoolExecutor)
futures = self.request_weight_updates(message)  # Returns list of futures

# SECOND: Do broadcasts (blocking, one per parameter)
for name, parameter in named_parameters.items():
    with deepspeed.zero.GatheredParameters([parameter]):
        if get_accelerator().is_main_process:
            dist.broadcast(parameter.data.bfloat16(), src=0, group=self.actor_update_group)

# THIRD: Wait for HTTP futures
for future in futures:
    future.result()
```

Same pattern: HTTP fires first (in background), then broadcasts, then wait for HTTP.

### SGLang (receiver side)

Location: `/tmp/sglang/python/sglang/srt/model_executor/model_runner.py:1363-1378`

```python
# Allocate buffers and start async receives
handles = []
for name, dtype, shape in zip(names, dtypes, shapes):
    weight = torch.empty(shape, dtype=target_dtype, device=self.device)
    handles.append(
        torch.distributed.broadcast(
            weight, src=0, group=self._model_update_group[group_name],
            async_op=True,  # ASYNC receive
        )
    )
    weights.append((name, weight))

# Wait for all receives
for handle in handles:
    handle.wait()
```

SGLang expects **both sides to use async broadcasts** and wait on handles after all are issued.

## The Fix

Change `sync_weights_nccl` in `pytorch.py` to follow the Miles/QED-Nano pattern:

1. **Start HTTP requests first** (fire-and-forget via thread pool, don't await response yet)
2. **Then do NCCL broadcasts** (use `async_op=True` like Miles/SGLang)
3. **Wait for broadcast handles** to complete
4. **Then wait for HTTP responses** to confirm

This ensures:
- Both sides enter their broadcast loops before any actual NCCL ops happen
- All broadcasts are issued async, then waited on together
- No deadlock or race between HTTP and NCCL

## Code Changes Required

```python
async def sync_weights_nccl(self) -> None:
    # ... setup code ...

    # STEP 1: Fire HTTP requests (don't await yet)
    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(self._nccl_inference_endpoints))
    http_futures = []
    for endpoint in self._nccl_inference_endpoints:
        future = executor.submit(
            requests.post,
            f"{endpoint}/update_weights_from_distributed",
            json={
                "names": [p["name"] for p in param_info],
                "shapes": [p["shape"] for p in param_info],
                "dtypes": [p["dtype"] for p in param_info],
                "group_name": "weight_sync",
            },
            timeout=300.0,
        )
        http_futures.append(future)

    # STEP 2: Do NCCL broadcasts (async_op=True)
    def _do_broadcast():
        handles = []
        for name, param in state_dict.items():
            param_data = param.data.contiguous()
            if param_data.device.type != "cuda":
                param_data = param_data.cuda()
            handle = dist.broadcast(param_data, src=0, group=self._nccl_process_group, async_op=True)
            handles.append(handle)

        # STEP 3: Wait for all broadcasts
        for handle in handles:
            handle.wait()

    await trio.to_thread.run_sync(_do_broadcast)

    # STEP 4: Wait for HTTP responses
    for future in http_futures:
        future.result()
    executor.shutdown(wait=False)
```

## Additional Observations

### Miles ReloadableProcessGroup

Miles has a `ReloadableProcessGroup` wrapper (`miles/utils/reloadable_process_group.py`) that:
- Wraps all NCCL process groups
- Tracks group metadata (ranks)
- Allows destroying and recreating groups on demand

This could be useful for error recovery if NCCL groups become invalid.

### Timing Diagram

**Current (broken) rollouts behavior:**
```
Trainer:     [--HTTP POST--]  [--broadcast--]
                    |              |
                    v              v
             (concurrent, race condition!)
                    |              |
SGLang:      [wait for HTTP...][--receive--]
                               ↑
                         (may miss broadcasts)
```

**Correct behavior (Miles/QED-Nano):**
```
Trainer:     [HTTP fire]  [--broadcast (async)--]  [wait handles]  [wait HTTP]
                  |                |                     |              |
                  v                v                     v              v
             (sequential, synchronized)
                  |                |                     |
SGLang:      [recv HTTP]  [--receive (async)--]   [wait handles]  [return HTTP]
```

## Files to Modify

- `rollouts/training/backends/pytorch.py`: Fix `sync_weights_nccl()` method

---

## Appendix: Miles vs QED-Nano Pipeline Approaches

**Miles does NOT do PipelineRL the way QED-Nano does.**

| Aspect | Miles (train_async.py) | QED-Nano (PipelineRL) |
|--------|------------------------|----------------------|
| **Sampling overlap** | Next rollout starts while training current batch | Continuous sampling, never stops |
| **Weight sync timing** | **Blocks** - waits for current generation to finish before update | **Non-blocking** - broadcasts while sampling continues |
| **Staleness handling** | `update_weights_interval` param, syncs before update | Samples tagged with `weight_version`, trains on potentially stale data |
| **Max lag** | Not tracked (implicit 1-step) | Explicit `max_lag` parameter to reject too-stale samples |

### Miles `train_async.py` pattern

```python
# Lines 63-67
if (rollout_id + 1) % args.update_weights_interval == 0:
    # sync generate before update weights to prevent update weight in the middle of generation
    rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
    rollout_data_next_future = None  # <-- STOPS next generation
    actor_model.update_weights()     # <-- BLOCKS until weight sync complete
```

Miles explicitly **stops** the next generation and **blocks** on weight update. This is "async sampling but sync weight sync" - overlapping generation with training, but NOT doing in-flight weight updates.

### QED-Nano PipelineRL pattern

```python
# finetune_loop.py
# Training and sampling run continuously
# Weight updates happen in-flight without stopping inference
if training_metrics.samples - training_metrics.last_broadcasted_version >= args.weight_update_interval:
    weight_update_manager.send_weight_update(training_metrics.samples)
    # This broadcasts weights while SGLang continues serving requests
```

QED-Nano **never stops inference** - weights are broadcast while sampling continues. Samples are tagged with `model_version` and filtered by `max_lag`.

### Miles `--true-on-policy-mode`

This is **different** from async weight sync - it's about **deterministic inference** (matching logprobs between trainer and inference engine). Uses `--sglang-rl-on-policy-target fsdp` which means SGLang loads weights from the FSDP trainer's memory directly via shared CUDA memory.

### Summary

- **Miles**: Async sampling (overlap generate N+1 with train N), but **blocking weight sync**
- **QED-Nano/PipelineRL**: True async everything - sampling never stops, weights broadcast in-flight
- Both use the same NCCL weight sync mechanism, but with different orchestration patterns
