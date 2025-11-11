# MiniRay: Simple Distributed Workers

**Heinrich's Pattern + TCP = Distributed System**

MiniRay is a ~600 line distributed computing system built for learning and research. It provides multi-node worker spawning with simpler semantics than Ray.

## Quick Start

### Single Node

```python
# Launch worker server
python -m rollouts.training.miniray.worker_server \
    --port 10000 \
    --work-fn my_module.work_fn

# Connect and use
from rollouts.training.miniray import RemoteWorker
worker = RemoteWorker("localhost", 10000)
worker.send({"cmd": "hello"})
print(worker.recv())
```

### Multi-Node Cluster

```python
from rollouts.training.miniray import Cluster, NodeConfig

# Define cluster
cluster = Cluster(nodes=[
    NodeConfig("node1.local", num_workers=4),
    NodeConfig("node2.local", num_workers=4),
])

# Launch workers (SSH + TCP)
workers = cluster.start(work_fn="module.train_fn")

# Use them (same API as local Worker!)
for worker in workers:
    worker.send({"cmd": "train", "batch": batch})

results = [w.recv() for w in workers]
cluster.stop()
```

### Distributed FSDP Training

```python
# 2 nodes, 8 GPUs, FSDP across nodes
cluster = Cluster(nodes=[
    NodeConfig("192.168.1.10", num_workers=4),
    NodeConfig("192.168.1.11", num_workers=4),
])

workers = cluster.start(work_fn="module.training_work_fn")

# Send NCCL configs
from rollouts.training.miniray import create_nccl_configs
configs = create_nccl_configs(
    master_addr="192.168.1.10",
    nodes=[("node1", 4), ("node2", 4)],
)

for worker, config in zip(workers, configs):
    worker.send(config.__dict__)

# Training loop (FSDP syncs across nodes!)
for batch in batches:
    for worker in workers:
        worker.send({"cmd": "forward_backward", "batch": batch})

    metrics = [worker.recv() for worker in workers]
    print(f"loss={sum(m['loss'] for m in metrics) / len(metrics):.4f}")
```

## Architecture

```
Coordinator ──SSH──> Node1: WorkerServer ──fork──> Workers (4 GPUs)
            └─SSH──> Node2: WorkerServer ──fork──> Workers (4 GPUs)
                          │                              │
                          └──────────NCCL────────────────┘
                          (Multi-node gradient sync)
```

## Components

| Component | File | Lines | What It Does |
|-----------|------|-------|--------------|
| **RemoteWorker** | `remote_worker.py` | ~150 | TCP client, connects to remote workers |
| **WorkerServer** | `worker_server.py` | ~200 | TCP server, spawns workers on connection |
| **Cluster** | `cluster.py` | ~200 | SSH to nodes, orchestrate workers |
| **NCCL Setup** | `nccl.py` | ~100 | Configure torch.distributed for multi-node |

**Total:** ~650 lines

## vs Ray

| Feature | MiniRay | Ray |
|---------|---------|-----|
| **Multi-node** | ✅ | ✅ |
| **Lines of code** | ~600 | ~50k+ |
| **Dependencies** | stdlib + PyTorch | Ray framework |
| **Semantics** | Simple (TCP + fork) | Complex (actors, object store) |
| **Learning curve** | 30 min | Days |
| **Understandable** | Every line | Black box |

## When to Use

**Use MiniRay:**
- ✅ Learning distributed systems
- ✅ Research codebases (1-10 nodes)
- ✅ You want full control
- ✅ Single-institution clusters

**Use Ray:**
- Production at scale
- Need fault tolerance
- Multi-tenant clusters
- Don't want to maintain infra

## Documentation

- **Full docs:** `docs/MINIRAY.md`
- **Example:** `examples/train_distributed_miniray.py`
- **Design:** `docs/MULTI_GPU_DESIGN.md`

## Philosophy

> "If you replace the UDS with a TCP socket you have the beginning of a distributed system here with much simpler semantics than Ray."
> — Heinrich Kuttler

MiniRay follows:
- **Tiger Style:** Explicit state, clear errors
- **Heinrich:** Simple primitives > complex abstractions
- **Casey Muratori:** Minimal coupling, gradual granularity

## License

Same as rollouts project.
