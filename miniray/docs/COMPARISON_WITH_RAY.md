# MiniRay vs Ray: Concrete Example

## The Task: Distributed RL Training (RLHF/PPO)

Let's say you want to train a language model with reinforcement learning across 8 GPUs on 2 nodes (4 GPUs each). You need:

1. **Actor model** - generates responses (needs GPU, FSDP sharded)
2. **Critic model** - scores responses (needs GPU, FSDP sharded)
3. **Training loop** - coordinate rollouts + training steps

This is exactly what SLIME does with Ray. Let's compare!

---

## How SLIME Uses Ray

### Architecture

```python
# SLIME's Ray-based approach
import ray

# 1. Create placement groups (Ray's resource allocation)
@ray.remote(num_gpus=1)
class FSDPTrainRayActor:
    def __init__(self, world_size, rank, master_addr, master_port):
        # Set up NCCL environment
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)

    def init(self, args):
        # Initialize torch.distributed
        dist.init_process_group(backend="nccl")

        # Create FSDP model
        self.model = FSDP(model, ...)

    def train(self, rollout_data_ref):
        # Training step
        batch = ray.get(rollout_data_ref)  # Fetch from object store
        loss = self.forward_backward(batch)
        return loss

# 2. Create actor group
class RayTrainGroup:
    def __init__(self, num_nodes, num_gpus_per_node, placement_group):
        world_size = num_nodes * num_gpus_per_node

        # Spawn Ray actors across nodes
        self._actor_handlers = []
        for rank in range(world_size):
            actor = FSDPTrainRayActor.options(
                num_gpus=1,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_bundle_index=rank,
                ),
            ).remote(world_size, rank, master_addr, master_port)
            self._actor_handlers.append(actor)

    def async_train(self, rollout_data_ref):
        # Send training command to all actors
        return [actor.train.remote(rollout_data_ref) for actor in self._actor_handlers]

# 3. Main training loop
ray.init(address="auto")  # Connect to Ray cluster

actor_group = RayTrainGroup(num_nodes=2, num_gpus_per_node=4, pg=pg)
critic_group = RayTrainGroup(num_nodes=2, num_gpus_per_node=4, pg=pg)

for step in range(num_steps):
    # Generate rollouts
    rollout_data = generate_rollouts(...)

    # Put in Ray object store
    rollout_ref = ray.put(rollout_data)

    # Train actor and critic in parallel
    actor_results = actor_group.async_train(rollout_ref)
    critic_results = critic_group.async_train(rollout_ref)

    # Wait for completion
    ray.get(actor_results + critic_results)
```

### What Ray Provides

✅ **Actor system**: `@ray.remote` classes that live on remote nodes
✅ **Object store**: Plasma store for sharing data (rollout batches)
✅ **Placement groups**: Resource allocation across nodes
✅ **Scheduling**: Automatic placement of actors on GPUs
✅ **Fault tolerance**: Actors restart on failure

### Ray's Complexity

❌ **Object store overhead**: `ray.put()` / `ray.get()` serialization
❌ **Placement group complexity**: Bundle indices, scheduling strategies
❌ **Hidden state**: Actors maintain state, hard to debug
❌ **Magic initialization**: `ray.init()` connects to cluster head
❌ **Large dependency**: Entire Ray framework + GCS server

---

## How We Use MiniRay

### Architecture

```python
# MiniRay's simpler approach
from miniray import Cluster, NodeConfig, Worker, NCCLConfig, setup_nccl_env
import torch.distributed as dist

# 1. Define work function (runs on each GPU)
def training_work_fn(handle):
    """This runs in each worker process."""
    # Receive NCCL config from coordinator
    config = NCCLConfig(**handle.recv())
    setup_nccl_env(config)

    # Initialize torch.distributed
    dist.init_process_group(backend="nccl", init_method="env://")

    # Create FSDP model
    from rollouts.training.backends.fsdp import FSDPTrainingBackend
    backend = FSDPTrainingBackend(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        checkpoint_dir=Path("checkpoints"),
    )

    # Message loop
    handle.send({"status": "ready"})

    while True:
        msg = handle.recv()

        if msg["cmd"] == "forward_backward":
            # Training step
            batch = msg["batch"]
            metrics = backend.forward_backward(batch)
            handle.send(metrics)

        elif msg["cmd"] == "optim_step":
            metrics = backend.optim_step()
            handle.send(metrics)

        elif msg["cmd"] == "shutdown":
            break

# 2. Launch cluster
cluster = Cluster(nodes=[
    NodeConfig("192.168.1.10", num_workers=4),  # Node 1: 4 GPUs
    NodeConfig("192.168.1.11", num_workers=4),  # Node 2: 4 GPUs
])

# Start workers (SSH + TCP)
actor_workers = cluster.start(work_fn="module.actor_training_work_fn")
critic_workers = cluster.start(work_fn="module.critic_training_work_fn")

# 3. Send NCCL configs
from miniray import create_nccl_configs

actor_configs = create_nccl_configs(
    master_addr="192.168.1.10",
    nodes=[("node1", 4), ("node2", 4)],
    master_port=29500,
)
critic_configs = create_nccl_configs(
    master_addr="192.168.1.10",
    nodes=[("node1", 4), ("node2", 4)],
    master_port=29501,  # Different port!
)

for worker, config in zip(actor_workers, actor_configs):
    worker.send(config.__dict__)
for worker, config in zip(critic_workers, critic_configs):
    worker.send(config.__dict__)

# Wait for ready
for worker in actor_workers + critic_workers:
    status = worker.recv()
    assert status["status"] == "ready"

# 4. Training loop
for step in range(num_steps):
    # Generate rollouts
    batch = generate_rollouts(...)

    # Send to all workers (explicit!)
    for worker in actor_workers:
        worker.send({"cmd": "forward_backward", "batch": batch})
    for worker in critic_workers:
        worker.send({"cmd": "forward_backward", "batch": batch})

    # Collect metrics
    actor_metrics = [w.recv() for w in actor_workers]
    critic_metrics = [w.recv() for w in critic_workers]

    # Optimizer step
    for worker in actor_workers + critic_workers:
        worker.send({"cmd": "optim_step"})

    opt_metrics = [w.recv() for w in actor_workers + critic_workers]

    print(f"Step {step}: actor_loss={avg(actor_metrics)}, critic_loss={avg(critic_metrics)}")

# 5. Cleanup
cluster.stop()
```

### What MiniRay Provides

✅ **TCP workers**: `RemoteWorker` connects via simple TCP socket
✅ **JSON messaging**: Explicit, human-readable communication
✅ **SSH deployment**: `cluster.start()` SSHs to nodes and launches servers
✅ **NCCL setup**: Helper to configure torch.distributed for multi-node
✅ **Minimal**: ~600 lines, stdlib + PyTorch only

### MiniRay's Simplicity

✅ **No object store**: Send data directly via `worker.send(batch)`
✅ **No placement groups**: You specify nodes + ports explicitly
✅ **Explicit state**: Message loop is visible, easy to debug
✅ **Simple deployment**: SSH + TCP, no cluster manager
✅ **Tiny dependency**: Just install MiniRay package

---

## Side-by-Side Comparison

### Toy Task: Train on 2 nodes, 8 GPUs total

| Step | Ray (SLIME) | MiniRay |
|------|-------------|---------|
| **Setup cluster** | `ray start --head` on head node<br>`ray start --address=...` on workers<br>Complex cluster config | SSH access to nodes<br>That's it |
| **Define actor** | `@ray.remote` decorator<br>Placement groups<br>Bundle indices | `def work_fn(handle):` function<br>Simple message loop |
| **Launch workers** | `actor.remote(...)` with scheduling strategy | `cluster.start(work_fn="module.work_fn")` |
| **Send data** | `ref = ray.put(data)`<br>`actor.method.remote(ref)` | `worker.send({"cmd": "train", "data": data})` |
| **Get results** | `ray.get(refs)` | `worker.recv()` |
| **Debug** | Actors hide state, need Ray dashboard | Print in work function, see immediately |
| **Code lines** | ~200 (actor) + ~100 (group) + Ray setup | ~100 (work_fn) + ~50 (cluster launch) |

### Real SLIME Example (Simplified)

```python
# SLIME: Create actor group
pg = ray.util.placement_group(bundles=[{"GPU": 1}] * 8)
actor_group = RayTrainGroup(
    args=args,
    num_nodes=2,
    num_gpus_per_node=4,
    pg=(pg, reordered_bundle_indices),  # Complex placement logic
    wandb_run_id=wandb_run_id,
)

# Initialize actors
ray.get(actor_group.async_init(args, role="actor"))

# Training step
rollout_ref = ray.put(rollout_data)  # Serialize to object store
results = actor_group.async_train(rollout_id, rollout_ref)
ray.get(results)  # Wait and deserialize
```

### MiniRay Equivalent

```python
# MiniRay: Create cluster
cluster = Cluster(nodes=[
    NodeConfig("node1", num_workers=4),
    NodeConfig("node2", num_workers=4),
])

# Launch workers
workers = cluster.start(work_fn="module.training_work_fn")

# Send NCCL configs
for worker, config in zip(workers, nccl_configs):
    worker.send(config.__dict__)

# Wait for ready
for worker in workers:
    assert worker.recv()["status"] == "ready"

# Training step
for worker in workers:
    worker.send({"cmd": "train", "batch": rollout_data})  # Direct send

# Get results
results = [worker.recv() for worker in workers]  # Direct recv
```

---

## When to Use Each?

### Use MiniRay

✅ **Learning distributed systems** - Understand every line
✅ **Research clusters** - 1-10 nodes, single institution
✅ **Debugging** - See exactly what's happening
✅ **Simple tasks** - Train model, run experiments
✅ **Full control** - Explicit communication, no magic

### Use Ray

✅ **Production** - Need fault tolerance, autoscaling
✅ **Complex workflows** - Many actors, complex dependencies
✅ **Multi-tenant** - Shared cluster with resource isolation
✅ **Large scale** - 100+ nodes
✅ **Don't want to maintain** - Let Ray handle infrastructure

---

## Key Insight

> **Ray is like Kubernetes for ML**: Powerful orchestration, but complex.
>
> **MiniRay is like SSH + tmux**: Simple primitives that compose.

For research and learning, MiniRay's explicit approach helps you understand distributed systems. Once you know how it works, Ray's abstractions make more sense!

**"Much simpler semantics than Ray."** 🚀
