# MiniRay - Simple Distributed Computing

**Heinrich's Pattern + TCP = Distributed System**

MiniRay is a ~600 line distributed computing system built for learning and research. It provides multi-node worker spawning with simpler semantics than Ray.

## Philosophy

> "Python multiprocessing is overused garbage. You spend all your time serializing and deserializing pickled stuff."
> — Heinrich Kuttler

> "And if you replace the UDS with a tcp socket you have the beginning of a distributed system here with much simpler semantics than, say, Ray, which also does too much."
> — Heinrich Kuttler

MiniRay follows three principles:

1. **Simple primitives over abstractions**: `os.fork()` + `socketpair` for local, TCP sockets for distributed
2. **No pickle overhead**: JSON for messages (or marshal for speed)
3. **Explicit control**: You see and control everything, no magic

## Installation

```bash
# From research directory
uv pip install -e miniray/
```

## Core Components

### 1. Worker (Local Multiprocessing)

Heinrich's pattern: `os.fork()` + `socketpair` for clean process isolation.

```python
from miniray import Worker

def work(handle):
    rank, world_size = handle.recv()
    print(f"Worker {rank}/{world_size}")
    result = do_computation(rank)
    handle.send(result)

# Spawn workers
workers = [Worker(work) for _ in range(4)]

# Send work
for i, w in enumerate(workers):
    w.send({"rank": i, "world_size": 4})

# Collect results
results = [w.recv() for w in workers]

# Wait for completion
for w in workers:
    w.wait()
```

**Why this is better than multiprocessing:**
- ✅ No pickle overhead (JSON is explicit and fast)
- ✅ No hidden "watchdog process" nonsense
- ✅ Full control over serialization
- ✅ Can set env vars before fork
- ✅ Simple: ~170 lines of code

### 2. RemoteWorker (Distributed via TCP)

Same API as Worker, but over the network:

```python
from miniray import RemoteWorker

# Connect to remote worker server
worker = RemoteWorker("node2.local", 10000)

# Same API as local Worker!
worker.send({"cmd": "train", "batch": batch})
result = worker.recv()
```

### 3. Cluster (Multi-Node Orchestration)

Manage workers across multiple nodes:

```python
from miniray import Cluster, NodeConfig

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

## Design Inspiration

MiniRay is built on three design philosophies:

### Heinrich Kuttler (Simplicity)
- Avoid Python's `multiprocessing` module (too much magic)
- Use `os.fork()` + `socketpair` for local processes
- Use TCP sockets for distributed workers
- JSON for serialization (no pickle complexity)

**Key insight:**
```python
# Local: os.fork() + socketpair
sock, sock0 = socket.socketpair()

# Distributed: TCP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host, port))
```
Replace UDS with TCP → distributed system with simple semantics!

### Tiger Style (Explicitness)
- Explicit state (no hidden globals)
- Clear error messages
- Assert preconditions
- You control everything

### Casey Muratori (Minimal Coupling)
- Protocols over inheritance
- Gradual granularity: Worker → RemoteWorker → Cluster
- No forced abstractions
- Simple primitives compose

## Architecture

```
┌─────────────────────────────────────────────┐
│         Coordinator (Main Process)          │
│  ┌────────────────────────────────────┐    │
│  │  Cluster.start()                   │    │
│  │  - SSH to each node                │    │
│  │  - Launch WorkerServer             │    │
│  │  - Connect RemoteWorker            │    │
│  └────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
         │                         │
         │ TCP                     │ TCP
         ↓                         ↓
┌─────────────────┐      ┌─────────────────┐
│     Node 1      │      │     Node 2      │
│  WorkerServer   │      │  WorkerServer   │
│  (TCP server)   │      │  (TCP server)   │
│       │         │      │       │         │
│       │ fork    │      │       │ fork    │
│       ↓         │      │       ↓         │
│  Worker (GPU 0) │      │  Worker (GPU 0) │
│  Worker (GPU 1) │      │  Worker (GPU 1) │
│  Worker (GPU 2) │      │  Worker (GPU 2) │
│  Worker (GPU 3) │      │  Worker (GPU 3) │
└─────────────────┘      └─────────────────┘
```

## vs Ray

| Feature | MiniRay | Ray |
|---------|---------|-----|
| **Lines of code** | ~600 | Library (~50k+) |
| **Dependencies** | stdlib + PyTorch (optional) | Ray + all deps |
| **Learning curve** | 30 minutes | Days |
| **Semantics** | Simple (TCP + fork) | Complex (actors, object store) |
| **Multi-node** | ✅ Yes | ✅ Yes |
| **Understandable** | 🔥 Every line | 🤷 Black box |

**Use MiniRay when:**
- ✅ Learning distributed systems
- ✅ Research codebases (1-10 nodes)
- ✅ You want full control
- ✅ Single-institution clusters

**Use Ray when:**
- Production at scale
- Need fault tolerance
- Multi-tenant clusters
- Don't want to maintain infrastructure

## Advanced Features (Future)

MiniRay keeps it simple by default, but you can add:

### memfd (Zero-Copy Data Passing)
```python
# Currently: JSON messages (fine for small data)
worker.send({"cmd": "train", "batch": batch})

# Advanced: memfd for large batches (Heinrich's approach)
fd = os.memfd_create("shared_batch")
os.ftruncate(fd, size)
# Share fd across processes, consumers open /proc/self/fd/$fd
```

### marshal (Faster Serialization)
```python
# Currently: JSON (human-readable, compatible)
json.dump(msg, w)

# Advanced: marshal (5-10x faster)
data = marshal.dumps(msg)
w.write(struct.pack('I', len(data)))
w.write(data)
```

### NCCL (Multi-Node GPU Training)
```python
from miniray import NCCLConfig, setup_nccl_env

# In each worker
config = NCCLConfig(**handle.recv())
setup_nccl_env(config)

# Initialize torch.distributed
dist.init_process_group(backend="nccl", init_method="env://")

# Now FSDP works across nodes!
model = FSDP(model, ...)
```

## Design Notes

### Why Classes (Not Pure Functions)?
Worker/RemoteWorker are classes because they're **handles** (pid + socket + file objects). The class groups these related handles together. A pure functional design would require passing 4 separate variables everywhere, which is more awkward.

### When to Use Protocols?
Add protocols when you have **multiple implementations** of the same interface. Currently we have:
- Worker (local via fork)
- RemoteWorker (TCP-based)

Both share the same API (send/recv/wait). When we add more implementations (e.g., RayWorker, SlurmWorker), that's when to formalize a WorkerProtocol.

**Guideline**: Protocol ≈ "I have 3+ implementations, time to formalize the interface"

## Documentation

- `../README_MINIRAY.md` - Quick reference
- `../rollouts/docs/MINIRAY.md` - Complete guide
- `../rollouts/examples/train_distributed_miniray.md` - Multi-node FSDP example

## Files (~600 lines total)

```
miniray/
├── __init__.py           # Package exports
├── worker.py             # Local Worker (os.fork + socketpair) ~170 lines
├── remote_worker.py      # TCP-based RemoteWorker ~150 lines
├── worker_server.py      # TCP server (spawns workers) ~200 lines
├── cluster.py            # Multi-node orchestration ~200 lines
└── nccl.py              # NCCL multi-node setup ~100 lines
```

## License

Same as rollouts project.

---

**"Much simpler semantics than Ray."** 🚀
