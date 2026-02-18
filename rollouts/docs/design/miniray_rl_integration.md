# MiniRay RL Integration

How to wire miniray primitives to slime-style RL training.

## What exists

miniray already has:
- `Worker` - local fork + socketpair
- `RemoteWorker` - TCP to remote nodes
- `Cluster` - SSH launch + connect
- `SharedMemory` - memfd/mmap for large data
- `NCCLConfig` - distributed training env setup
- `wait_any` - select() on multiple workers

## What we need to build

### 1. Rollout buffer layout

Define how rollouts sit in SharedMemory so training workers can read without copies.

```python
@dataclass
class RolloutBufferLayout:
    """Memory layout for rollouts in SharedMemory."""
    max_rollouts: int
    max_seq_len: int

    # Offsets into shared memory
    input_ids_offset: int      # [max_rollouts, max_seq_len] int32
    completion_ids_offset: int # [max_rollouts, max_seq_len] int32
    logprobs_offset: int       # [max_rollouts, max_seq_len] float32
    rewards_offset: int        # [max_rollouts] float32
    lengths_offset: int        # [max_rollouts] int32

    # Cursor for producer/consumer
    write_idx_offset: int      # int32 - next write position
    read_idx_offset: int       # int32 - next read position

def create_rollout_buffer(max_rollouts: int, max_seq_len: int) -> tuple[SharedMemory, RolloutBufferLayout]:
    """Allocate shared memory for rollout buffer."""
    ...

def write_rollout(shm: SharedMemory, layout: RolloutBufferLayout, rollout: dict) -> None:
    """Write one rollout to buffer (called by inference coordinator)."""
    ...

def read_rollout_batch(shm: SharedMemory, layout: RolloutBufferLayout, batch_size: int) -> dict:
    """Read batch from buffer (called by training worker)."""
    ...
```

### 2. Weight sync

Get updated weights from training workers to inference endpoints.

Options:
1. **Save/load** - training saves checkpoint, inference reloads (simple, slow)
2. **NCCL broadcast** - if inference on same cluster (fast, requires NCCL setup)
3. **HTTP push** - serialize weights, POST to Modal endpoint (medium)

For Modal inference, probably option 3:

```python
async def sync_weights_to_modal(
    weights: dict[str, Tensor],
    endpoints: list[str],
) -> None:
    """Push weights to Modal inference endpoints."""
    # Serialize to safetensors bytes
    buf = io.BytesIO()
    save_file(weights, buf)

    # POST to each endpoint
    async with httpx.AsyncClient() as client:
        await gather([
            client.post(f"{ep}/update_weights", content=buf.getvalue())
            for ep in endpoints
        ])
```

### 3. Training worker function

What runs on each GPU:

```python
def grpo_training_worker(handle: Worker) -> None:
    """Training worker that reads from shared buffer."""
    # Setup
    config = handle.recv(max_size=4096)
    setup_nccl_env(NCCLConfig(**config["nccl"]))
    dist.init_process_group(backend="nccl")

    model = load_model(config["model_path"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    # Attach to shared buffer
    shm_fd = config["buffer_fd"]  # Passed via fork
    shm = SharedMemory(fd=shm_fd, size=config["buffer_size"])
    layout = RolloutBufferLayout(**config["buffer_layout"])

    while True:
        msg = handle.recv(max_size=1024)

        if msg["cmd"] == "shutdown":
            break

        if msg["cmd"] == "train":
            # Read batch from shared buffer
            batch = read_rollout_batch(shm, layout, batch_size=config["batch_size"])

            # GRPO step
            loss = grpo_step(model, optimizer, batch)

            handle.send({"loss": loss.item()})

        if msg["cmd"] == "get_weights":
            # Return weights for sync
            handle.send({"weights": {k: v.cpu() for k, v in model.state_dict().items()}})
```

### 4. The training loop

```python
async def rl_train(
    model_path: str,
    inference_endpoints: list[str],
    prompts: list[dict],
    cluster: Cluster,
    steps: int = 100,
    batch_size: int = 32,
):
    # Create shared buffer
    shm, layout = create_rollout_buffer(max_rollouts=1024, max_seq_len=2048)

    # Launch training workers
    workers = cluster.start(work_fn="rollouts.training.grpo_worker.grpo_training_worker")

    # Send config to workers
    for i, w in enumerate(workers):
        w.send({
            "nccl": {"master_addr": ..., "rank": i, "world_size": len(workers), ...},
            "model_path": model_path,
            "lr": 1e-6,
            "batch_size": batch_size,
            "buffer_fd": shm.fd,
            "buffer_size": shm.size,
            "buffer_layout": asdict(layout),
        })

    for step in range(steps):
        # 1. Generate rollouts
        rollouts = await generate_rollouts(inference_endpoints, prompts[step])

        # 2. Write to buffer
        for rollout in rollouts:
            write_rollout(shm, layout, rollout)

        # 3. Train
        for w in workers:
            w.send({"cmd": "train"})
        losses = [w.recv(max_size=1024)["loss"] for w in workers]
        print(f"step {step}: loss={sum(losses)/len(losses):.4f}")

        # 4. Sync weights (every N steps)
        if step % 10 == 0:
            workers[0].send({"cmd": "get_weights"})
            weights = workers[0].recv(max_size=100*1024*1024)["weights"]
            await sync_weights_to_modal(weights, inference_endpoints)

    shutdown_workers(workers)
    shm.close()
```

## Open questions

1. **Buffer backpressure** - what if inference is faster than training? Need to block or drop.

2. **Multi-node buffer** - SharedMemory is single-node. For multi-node, either:
   - Each node has local buffer, training reads from local
   - RDMA for cross-node shared memory
   - Coordinator gathers rollouts, scatters to workers

3. **Modal weight sync latency** - HTTP round-trip for weights could be slow. Options:
   - Async sync (inference uses stale weights for 1-2 steps)
   - Modal volumes (write to mounted storage)
   - Keep inference on same cluster as training

4. **Fault tolerance** - what if a worker dies? Currently: crash. Future: restart.

## Implementation order

1. `RolloutBufferLayout` + read/write functions
2. `grpo_training_worker`
3. `rl_train` loop (single node first)
4. Weight sync to Modal
5. Multi-node

## Non-goals (for now)

- Dynamic task graphs (submit tasks that spawn subtasks)
- Fault tolerance / task re-execution
- Autoscaling
- PPO (needs critic model, more complex)
