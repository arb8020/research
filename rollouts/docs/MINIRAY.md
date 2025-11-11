

# MiniRay: Simple Distributed Computing

**TCP + Heinrich's Pattern = Distributed Workers**

MiniRay is a minimal distributed computing system built on simple primitives:
- TCP sockets (replaces UDS for multi-node)
- `os.fork()` (process spawning)
- JSON messaging (no pickle complexity)
- NCCL (for GPU gradient synchronization)

**Simpler semantics than Ray, same power for distributed training.**

---

## Why MiniRay?

### The Problem with Ray

Ray is powerful but complex:
- Actor system with hidden state
- Plasma object store (opaque)
- Complex scheduling (placement groups)
- Too much magic for learning

### MiniRay's Philosophy

> "If you replace the UDS with a TCP socket you have the beginning of a distributed system here with much simpler semantics than Ray."
> — Heinrich Kuttler

**MiniRay provides:**
- ✅ Multi-node worker spawning
- ✅ Same API as local Worker pattern
- ✅ NCCL setup for distributed training
- ✅ Simple, understandable code (~600 lines)
- ✅ No hidden magic

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│              Coordinator (Node 1)                    │
│  ┌────────────────────────────────────┐             │
│  │  Cluster.start()                   │             │
│  │  - SSH to each node                │             │
│  │  - Launch WorkerServer             │             │
│  │  - Connect RemoteWorker            │             │
│  └────────────────────────────────────┘             │
└──────────────────────────────────────────────────────┘
              │                         │
              │ TCP                     │ TCP
              ↓                         ↓
┌──────────────────────────┐  ┌──────────────────────────┐
│      Node 1              │  │      Node 2              │
│  ┌────────────────┐      │  │  ┌────────────────┐      │
│  │ WorkerServer   │      │  │  │ WorkerServer   │      │
│  │  :10000        │      │  │  │  :10000        │      │
│  └────────────────┘      │  │  └────────────────┘      │
│         │                │  │         │                │
│         │ os.fork()      │  │         │ os.fork()      │
│         ↓                │  │         ↓                │
│  ┌────────────────┐      │  │  ┌────────────────┐      │
│  │ Worker (GPU 0) │◄─┐   │  │  │ Worker (GPU 0) │◄─┐   │
│  │ Worker (GPU 1) │  │   │  │  │ Worker (GPU 1) │  │   │
│  │ Worker (GPU 2) │  │   │  │  │ Worker (GPU 2) │  │   │
│  │ Worker (GPU 3) │  │   │  │  │ Worker (GPU 3) │  │   │
│  └────────────────┘  │   │  │  └────────────────┘  │   │
│                      │   │  │                      │   │
│   torch.distributed  │   │  │   torch.distributed  │   │
│   NCCL group         │   │  │   NCCL group         │   │
└──────────────────────┼───┘  └──────────────────────┼───┘
                       │                             │
                       └─────────NCCL (GPU)──────────┘
                       (Multi-node gradient sync)
```

---

## Components

### 1. RemoteWorker (TCP Client)

**File:** `rollouts/training/miniray/remote_worker.py`

**What it does:** Connects to remote WorkerServer via TCP

**API:** Same as local `Worker`!

```python
# Local worker (single node)
from rollouts.training.worker import Worker
worker = Worker(work_fn)

# Remote worker (multi-node)
from rollouts.training.miniray import RemoteWorker
worker = RemoteWorker("node2.local", 10000)

# Same API!
worker.send({"cmd": "train", "batch": batch})
result = worker.recv()
```

---

### 2. WorkerServer (TCP Server)

**File:** `rollouts/training/miniray/worker_server.py`

**What it does:** Listens on a port, spawns workers on connection

**Runs on:** Each compute node

```bash
# On node1
python -m rollouts.training.miniray.worker_server \
    --port 10000 \
    --workers 4 \
    --work-fn my_module.train_fn

# On node2
python -m rollouts.training.miniray.worker_server \
    --port 10000 \
    --workers 4 \
    --work-fn my_module.train_fn
```

**What happens:**
1. Listens on port 10000
2. Accepts 4 connections
3. Spawns worker process (`os.fork()`) for each connection
4. Worker runs `my_module.train_fn(handle)`

---

### 3. Cluster (Orchestrator)

**File:** `rollouts/training/miniray/cluster.py`

**What it does:** SSH to nodes, launch servers, connect workers

**API:**
```python
from rollouts.training.miniray import Cluster, NodeConfig

# Define cluster
cluster = Cluster(nodes=[
    NodeConfig("node1.local", num_workers=4),
    NodeConfig("node2.local", num_workers=4),
])

# Launch workers (SSH + connect)
workers = cluster.start(work_fn="module.train_fn")

# Use them (same API as local!)
for worker in workers:
    worker.send({"cmd": "train", "batch": batch})

results = [w.recv() for w in workers]

# Cleanup
cluster.stop()
```

---

### 4. NCCL Setup (Multi-Node Training)

**File:** `rollouts/training/miniray/nccl.py`

**What it does:** Configures torch.distributed for multi-node

**Problem:** torch.distributed needs these env vars for multi-node:
- `MASTER_ADDR`: IP of rank 0 node
- `MASTER_PORT`: Port for coordination
- `WORLD_SIZE`: Total number of processes
- `RANK`: Global rank of this process
- `LOCAL_RANK`: Rank within this node

**Solution:**
```python
from rollouts.training.miniray import NCCLConfig, setup_nccl_env

# In each worker
def train_fn(handle):
    # Receive config from coordinator
    config = NCCLConfig(**handle.recv())

    # Set up NCCL environment
    setup_nccl_env(config)

    # Initialize torch.distributed
    dist.init_process_group(backend="nccl", init_method="env://")

    # Now FSDP works across nodes!
    model = FSDP(model, ...)
```

---

## Quick Start

### Single Node (Local Testing)

```python
# Terminal 1: Launch worker server
python -m rollouts.training.miniray.worker_server \
    --port 10000 \
    --work-fn my_module.work_fn

# Terminal 2: Connect and use
from rollouts.training.miniray import RemoteWorker

worker = RemoteWorker("localhost", 10000)
worker.send({"cmd": "echo", "data": "hello"})
print(worker.recv())
```

### Multi-Node (Cluster)

```python
# On coordinator node
from rollouts.training.miniray import Cluster, NodeConfig

cluster = Cluster(nodes=[
    NodeConfig("node1", num_workers=4, base_port=10000),
    NodeConfig("node2", num_workers=4, base_port=10000),
])

workers = cluster.start(work_fn="my_module.train_fn")

# Use workers
for worker in workers:
    worker.send({"cmd": "train"})

results = [w.recv() for w in workers]
cluster.stop()
```

---

## Complete Example: Distributed FSDP Training

See `examples/train_distributed_miniray.py` for full example.

### Work Function (Runs on Each Worker)

```python
def training_work_fn(handle):
    """Training worker for distributed FSDP."""
    # 1. Receive NCCL config
    config = NCCLConfig(**handle.recv())
    setup_nccl_env(config)

    # 2. Initialize torch.distributed
    dist.init_process_group(backend="nccl", init_method="env://")

    # 3. Create FSDP backend
    from rollouts.training.backends.fsdp import FSDPTrainingBackend

    backend = FSDPTrainingBackend(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        checkpoint_dir=Path("checkpoints"),
    )

    # 4. Training loop
    handle.send({"status": "ready"})

    while True:
        msg = handle.recv()

        if msg["cmd"] == "forward_backward":
            metrics = backend.forward_backward(msg["batch"])
            handle.send(metrics)

        elif msg["cmd"] == "optim_step":
            metrics = backend.optim_step()
            handle.send(metrics)

        elif msg["cmd"] == "shutdown":
            break
```

### Coordinator Script

```python
async def main():
    # 1. Define cluster
    cluster = Cluster(nodes=[
        NodeConfig("192.168.1.10", num_workers=4),
        NodeConfig("192.168.1.11", num_workers=4),
    ])

    # 2. Launch workers
    workers = cluster.start(work_fn="my_module.training_work_fn")

    # 3. Send NCCL configs
    nccl_configs = create_nccl_configs(
        master_addr="192.168.1.10",
        nodes=[("node1", 4), ("node2", 4)],
    )

    for worker, config in zip(workers, nccl_configs):
        worker.send(config.__dict__)

    # Wait for ready
    for worker in workers:
        assert worker.recv()["status"] == "ready"

    # 4. Training loop
    for step in range(num_steps):
        # Forward/backward
        for worker in workers:
            worker.send({"cmd": "forward_backward", "batch": batch})

        metrics = [worker.recv() for worker in workers]
        avg_loss = sum(m["loss"] for m in metrics) / len(metrics)

        # Optimizer step
        for worker in workers:
            worker.send({"cmd": "optim_step"})

        opt_metrics = [worker.recv() for worker in workers]

        print(f"Step {step}: loss={avg_loss:.4f}")

    # 5. Cleanup
    cluster.stop()
```

---

## Comparison: MiniRay vs Ray

| Feature | MiniRay | Ray |
|---------|---------|-----|
| **Lines of code** | ~600 | Library (~50k+) |
| **Dependencies** | Python stdlib + PyTorch | Ray + all its deps |
| **Learning curve** | 30 minutes | Days |
| **Semantics** | Simple (TCP + fork) | Complex (actors, object store) |
| **Multi-node** | ✅ Yes | ✅ Yes |
| **NCCL setup** | Explicit (you set env vars) | Automatic |
| **Fault tolerance** | ❌ No (manual restart) | ✅ Yes |
| **Object store** | ❌ No (use NFS or add memfd) | ✅ Yes (Plasma) |
| **GPU scheduling** | ❌ Manual | ✅ Automatic |
| **Understandability** | 🔥 You understand every line | 🤷 Black box |

**When to use MiniRay:**
- Learning distributed systems ✅
- Single-institution clusters (1-10 nodes) ✅
- Research codebases ✅
- You want full control ✅

**When to use Ray:**
- Production deployments at scale
- Need fault tolerance
- Multi-tenant clusters
- Don't want to maintain infrastructure

---

## Adding Features

### memfd (Zero-Copy Data Passing)

Currently, MiniRay uses JSON for messages (fine for small data).
For large batches, add memfd + NFS:

```python
# rollouts/training/miniray/shared_memory.py
class RemoteWorkerWithShmem(RemoteWorker):
    def __init__(self, host, port):
        super().__init__(host, port)

        # Create shared memory region
        self.shmem_fd = os.memfd_create(f"worker_{host}_{port}")
        os.ftruncate(self.shmem_fd, 100_000_000)  # 100MB
        self.shmem = mmap.mmap(self.shmem_fd, 100_000_000)

    def send_large(self, data: bytes):
        """Zero-copy send via shared memory"""
        self.shmem.seek(0)
        self.shmem.write(data)
        self.send({"type": "shmem", "size": len(data)})

    def recv_large(self) -> bytes:
        """Zero-copy recv from shared memory"""
        msg = self.recv()
        assert msg["type"] == "shmem"
        self.shmem.seek(0)
        return self.shmem.read(msg["size"])
```

### marshal (Faster Serialization)

Replace JSON with marshal for 5-10x speedup:

```python
# In RemoteWorker
def send(self, msg):
    data = marshal.dumps(msg)
    self.w.write(struct.pack('I', len(data)))
    self.w.write(data)
    self.w.flush()

def recv(self):
    length = struct.unpack('I', self.r.read(4))[0]
    data = self.r.read(length)
    return marshal.loads(data)
```

### Function Registry (Ray-like @remote)

```python
class RemoteWorkerWithRegistry(RemoteWorker):
    def __init__(self, host, port):
        super().__init__(host, port)
        self.functions = {}

    def register(self, name: str, fn: Callable):
        self.functions[name] = fn

    def call_remote(self, name: str, *args, **kwargs):
        self.send({"cmd": "call", "fn": name, "args": args, "kwargs": kwargs})
        return self.recv()

# Usage (like @ray.remote)
worker.register("train_step", train_step_fn)
result = worker.call_remote("train_step", batch)
```

---

## Troubleshooting

### "Connection refused" when connecting to worker

**Cause:** WorkerServer not running on remote node

**Fix:**
```bash
# On remote node, check if server is running
ps aux | grep worker_server

# If not, launch it
python -m rollouts.training.miniray.worker_server --port 10000 --work-fn ...
```

### "SSH permission denied"

**Cause:** No SSH key set up

**Fix:**
```bash
# Generate SSH key
ssh-keygen -t rsa

# Copy to remote node
ssh-copy-id user@node2
```

### NCCL hangs at initialization

**Cause:** MASTER_ADDR or firewall

**Fix:**
1. Check MASTER_ADDR is reachable: `ping 192.168.1.10`
2. Check port 29500 is open: `nc -zv 192.168.1.10 29500`
3. Disable firewall: `sudo ufw disable` (Ubuntu)

### Workers die silently

**Cause:** Exception in work function

**Fix:** Check stderr from WorkerServer:
```bash
# Launch server with visible stderr
python -m rollouts.training.miniray.worker_server ... 2>&1 | tee worker.log
```

---

## Files Created

```
rollouts/training/miniray/
├── __init__.py              # Package exports
├── remote_worker.py         # TCP client (RemoteWorker)
├── worker_server.py         # TCP server (WorkerServer)
├── cluster.py               # Cluster orchestration
└── nccl.py                  # NCCL multi-node setup

examples/
└── train_distributed_miniray.py  # Complete example

docs/
└── MINIRAY.md              # This document
```

**Total:** ~600 lines for full distributed system

---

## Design Principles

### Tiger Style
- Explicit state (no hidden globals)
- Clear error messages
- Assert preconditions

### Heinrich Kuttler
- Simple primitives (fork + TCP > complex abstractions)
- No pickle overhead (use JSON/marshal)
- Full control over serialization

### Casey Muratori
- Minimal coupling (protocols over inheritance)
- Gradual granularity (Worker → RemoteWorker → Cluster)
- No forced abstractions

---

## Future Work

**When you actually need Ray's features:**

1. **Fault tolerance** - Add supervision layer (~200 lines)
2. **Object store** - Add memfd + registry (~300 lines)
3. **Smart scheduling** - Add GPU affinity (~200 lines)

**Or:** Just use Ray at that point! MiniRay has taught you how it works. 😄

---

## References

- Heinrich's multiprocessing pattern: `/docs/code_style/multiprocessing_heinrich.md`
- SLIME's Ray architecture: `references/slime/`
- TorchForge's service pattern: `references/torchforge/`
- Ray design principles: `/docs/code_style/ray_design.txt`

---

## Conclusion

**MiniRay gives you:**
- ✅ Multi-node distributed training
- ✅ Simple, understandable code
- ✅ Same API as local Worker
- ✅ Full control (no black boxes)
- ✅ Perfect for learning

**"Much simpler semantics than Ray."** 🚀
