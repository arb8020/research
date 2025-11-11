# Toy Example: Parallel Matrix Multiplication

The simplest distributed computing task: multiply large matrices in parallel.

## Task

You have 1000 matrix pairs to multiply. Each multiplication takes ~1 second on CPU.
You want to use 8 workers across 2 machines to speed it up.

---

## With Ray

```python
import ray
import numpy as np

# 1. Start Ray cluster (must run separately!)
# Terminal on node1: ray start --head --port=6379
# Terminal on node2: ray start --address="node1:6379"

# 2. Connect to cluster
ray.init(address="auto")

# 3. Define remote function
@ray.remote
def matrix_multiply(A, B):
    """Multiply matrices A and B."""
    return np.matmul(A, B)

# 4. Generate work
matrices = [(np.random.rand(1000, 1000), np.random.rand(1000, 1000))
            for _ in range(1000)]

# 5. Submit tasks to Ray
futures = []
for A, B in matrices:
    # ray.put() copies data to object store
    A_ref = ray.put(A)
    B_ref = ray.put(B)
    # Submit task
    future = matrix_multiply.remote(A_ref, B_ref)
    futures.append(future)

# 6. Get results
results = ray.get(futures)  # Blocks until all complete

print(f"Computed {len(results)} matrix products")
```

**What Ray does:**
- Plasma object store holds matrices
- Scheduler assigns tasks to 8 workers
- Automatic load balancing
- Fault tolerance (retries on failure)

**Complexity:**
- Must start Ray cluster separately
- Object store serialization overhead
- Can't see which worker does what
- Debugging requires Ray dashboard

---

## With MiniRay

```python
from miniray import Cluster, NodeConfig
import numpy as np

# 1. Define work function
def matrix_worker(handle):
    """Worker that multiplies matrices."""
    while True:
        msg = handle.recv()

        if msg["cmd"] == "multiply":
            A = np.array(msg["A"])
            B = np.array(msg["B"])
            result = np.matmul(A, B)
            handle.send({"result": result.tolist()})

        elif msg["cmd"] == "shutdown":
            break

# 2. Launch cluster (SSH + TCP)
cluster = Cluster(nodes=[
    NodeConfig("node1.local", num_workers=4),
    NodeConfig("node2.local", num_workers=4),
])

workers = cluster.start(work_fn="__main__.matrix_worker")

# 3. Generate work
matrices = [(np.random.rand(1000, 1000), np.random.rand(1000, 1000))
            for _ in range(1000)]

# 4. Simple round-robin scheduling
results = []
for i, (A, B) in enumerate(matrices):
    worker = workers[i % len(workers)]
    worker.send({
        "cmd": "multiply",
        "A": A.tolist(),
        "B": B.tolist(),
    })

# 5. Collect results
for i in range(len(matrices)):
    worker = workers[i % len(workers)]
    result = worker.recv()
    results.append(result["result"])

print(f"Computed {len(results)} matrix products")

# 6. Cleanup
for worker in workers:
    worker.send({"cmd": "shutdown"})
cluster.stop()
```

**What MiniRay does:**
- TCP sockets for communication
- You do the scheduling (round-robin here)
- Explicit send/recv
- No extra services needed

**Simplicity:**
- Just SSH access needed
- Direct data transfer (JSON)
- Print in worker to see what's happening
- Simple to debug

---

## Even Simpler: Local Workers (No Network)

Maybe you just want to use all 8 cores on one machine:

```python
from miniray import Worker
import numpy as np

# 1. Define work function
def matrix_worker(handle):
    """Worker that multiplies matrices."""
    while True:
        msg = handle.recv()

        if msg["cmd"] == "multiply":
            A = np.array(msg["A"])
            B = np.array(msg["B"])
            result = np.matmul(A, B)
            handle.send({"result": result.tolist()})

        elif msg["cmd"] == "shutdown":
            break

# 2. Spawn local workers (os.fork + socketpair)
workers = [Worker(matrix_worker) for _ in range(8)]

# 3. Generate work
matrices = [(np.random.rand(1000, 1000), np.random.rand(1000, 1000))
            for _ in range(1000)]

# 4. Distribute work
results = []
for i, (A, B) in enumerate(matrices):
    worker = workers[i % len(workers)]
    worker.send({
        "cmd": "multiply",
        "A": A.tolist(),
        "B": B.tolist(),
    })

# 5. Collect results
for i in range(len(matrices)):
    worker = workers[i % len(workers)]
    result = worker.recv()
    results.append(result["result"])

print(f"Computed {len(results)} matrix products")

# 6. Cleanup
for worker in workers:
    worker.send({"cmd": "shutdown"})
    worker.wait()
```

**Notice:** Same API for local and distributed! Just change:
- `Worker()` → `cluster.start()`
- Everything else stays the same!

---

## Comparison

| Feature | Ray | MiniRay (Cluster) | MiniRay (Local) |
|---------|-----|-------------------|-----------------|
| **Setup** | Ray cluster (GCS, object store) | SSH access | None! |
| **Code** | `@ray.remote` + `ray.get()` | `cluster.start()` + `send/recv` | `Worker()` + `send/recv` |
| **Scheduling** | Automatic | Manual (round-robin) | Manual (round-robin) |
| **Debugging** | Ray dashboard needed | Print statements work | Print statements work |
| **Lines of code** | ~30 | ~50 | ~45 |
| **Dependencies** | Ray (~500MB) | MiniRay (~60KB) | MiniRay (~60KB) |

---

## Key Takeaway

**Ray:** Great for complex workflows, automatic scheduling, production.

**MiniRay:** Great for learning, debugging, research. You control everything.

**The best part?** MiniRay's simplicity helps you understand what Ray is doing under the hood. Once you get it, Ray's abstractions make sense!

---

## Next Steps

Try this yourself:

1. **Start local:** Use `Worker()` on your laptop
2. **Go distributed:** Change to `Cluster()` when you have multiple machines
3. **Add GPU training:** Use the FSDP example in `examples/train_distributed_miniray.md`

**"Simple primitives that compose."** 🚀
