# Example: Distributed Training with MiniRay (2 nodes, 8 GPUs)

**⚠️ UNTESTED CONCEPTUAL EXAMPLE - Use as reference only**

This example shows how to use MiniRay for multi-node distributed training
with FSDP. It demonstrates:
- Cluster setup (2 nodes, 4 GPUs each)
- NCCL configuration for multi-node
- FSDP training across nodes
- Same API as local Worker pattern

## Requirements

- 2 nodes with 4 GPUs each
- SSH access to both nodes
- PyTorch with NCCL
- Synchronized code on both nodes (git, rsync, or NFS)

## Usage

```bash
# On coordinator node (node1):
python this_script.py  # (if you make it executable)
```

---

## Code

```python
"""
Tiger Style: Explicit distributed setup, clear flow.
Heinrich Kuttler: TCP + fork = distributed workers.
"""

import asyncio
from pathlib import Path

import torch
import torch.distributed as dist

from rollouts.training.miniray import Cluster, NodeConfig, NCCLConfig, setup_nccl_env


# ============================================================================
# Work Function (Runs on Each Worker)
# ============================================================================

def training_work_fn(handle):
    """Training worker function for distributed FSDP.

    This function runs in each worker process on each node.
    It receives NCCL config, initializes torch.distributed, and trains.

    Args:
        handle: Worker handle for communication (Heinrich API)
    """
    import os

    # === Step 1: Receive NCCL config from coordinator ===
    print(f"[Worker {os.getpid()}] Waiting for NCCL config...")
    config_dict = handle.recv()
    config = NCCLConfig(**config_dict)

    print(f"[Worker {os.getpid()}] Received config: rank {config.rank}/{config.world_size}")

    # === Step 2: Set up NCCL environment ===
    setup_nccl_env(config)

    # === Step 3: Initialize torch.distributed ===
    dist.init_process_group(
        backend="nccl",
        init_method="env://",  # Uses env vars we set
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"[Worker {rank}] torch.distributed initialized ({world_size} workers)")

    # === Step 4: Create model + FSDP ===
    from rollouts.training.backends.fsdp import FSDPTrainingBackend, FSDPConfig

    # Simple model for example
    import torch.nn as nn

    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def simple_loss(logits, labels, loss_mask=None, advantages=None):
        return nn.functional.mse_loss(logits, labels)

    # Create FSDP backend (shards model across nodes!)
    backend = FSDPTrainingBackend(
        model=model,
        optimizer=optimizer,
        loss_fn=simple_loss,
        checkpoint_dir=Path("checkpoints"),
        config=FSDPConfig(
            sharding_strategy="FULL_SHARD",
            mixed_precision=True,
        ),
    )

    print(f"[Worker {rank}] FSDP backend initialized")

    # === Step 5: Training message loop ===
    handle.send({"status": "ready"})

    while True:
        msg = handle.recv()
        cmd = msg.get("cmd")

        if cmd == "forward_backward":
            # Forward + backward pass
            batch = msg["batch"]

            # Convert to tensors
            input_ids = torch.tensor(batch["input_ids"], device=f"cuda:{config.local_rank}")
            labels = torch.tensor(batch["labels"], device=f"cuda:{config.local_rank}")

            # FSDP training step (syncs gradients across nodes!)
            result = backend.forward_backward({
                "input_ids": input_ids,
                "labels": labels,
            })

            metrics = result.result() if hasattr(result, 'result') else result
            handle.send(metrics)

        elif cmd == "optim_step":
            # Optimizer step
            result = backend.optim_step()
            metrics = result.result() if hasattr(result, 'result') else result
            handle.send(metrics)

        elif cmd == "shutdown":
            print(f"[Worker {rank}] Shutting down...")
            dist.destroy_process_group()
            break

    print(f"[Worker {rank}] Exited")


# ============================================================================
# Coordinator (Main Script)
# ============================================================================

async def main():
    """Main training loop (runs on coordinator node)."""
    import os

    print("=== MiniRay Distributed Training Example ===\n")

    # === Step 1: Define cluster ===
    cluster = Cluster(nodes=[
        NodeConfig(
            host="192.168.1.10",  # Node 1 (replace with your IP)
            num_workers=4,        # 4 GPUs
            base_port=10000,
        ),
        NodeConfig(
            host="192.168.1.11",  # Node 2 (replace with your IP)
            num_workers=4,        # 4 GPUs
            base_port=10000,
        ),
    ])

    print(f"Cluster config:")
    print(f"  Node 1: {cluster.nodes[0].host} (4 GPUs)")
    print(f"  Node 2: {cluster.nodes[1].host} (4 GPUs)")
    print(f"  Total:  8 workers\n")

    # === Step 2: Launch workers ===
    print("Launching worker servers...")
    workers = cluster.start(
        work_fn="__main__.training_work_fn",  # Adjust module path!
        wait_time=5.0,
    )

    print(f"Connected to {len(workers)} workers\n")

    # === Step 3: Send NCCL configs ===
    print("Sending NCCL configs to workers...")

    # Create NCCL configs for all workers
    from rollouts.training.miniray.nccl import create_nccl_configs

    nccl_configs = create_nccl_configs(
        master_addr="192.168.1.10",  # Node 1 is master
        nodes=[
            ("node1", 4),  # 4 GPUs on node 1
            ("node2", 4),  # 4 GPUs on node 2
        ],
        master_port=29500,
    )

    # Send config to each worker
    for worker, config in zip(workers, nccl_configs):
        worker.send(config.__dict__)

    # Wait for all workers to be ready
    print("Waiting for workers to initialize...")
    for i, worker in enumerate(workers):
        status = worker.recv()
        print(f"  Worker {i}: {status['status']}")

    print("\nAll workers ready!\n")

    # === Step 4: Training loop ===
    print("Starting training loop...\n")

    num_steps = 10
    batch_size = 8

    for step in range(num_steps):
        # Create dummy batch
        batch = {
            "input_ids": torch.randint(0, 256, (batch_size, 512)).tolist(),
            "labels": torch.randn(batch_size, 256).tolist(),
        }

        # Send forward_backward to all workers
        for worker in workers:
            worker.send({"cmd": "forward_backward", "batch": batch})

        # Collect metrics (FSDP syncs gradients automatically!)
        metrics = [worker.recv() for worker in workers]
        avg_loss = sum(m["loss"] for m in metrics) / len(metrics)
        avg_grad_norm = sum(m["grad_norm"] for m in metrics) / len(metrics)

        # Send optim_step to all workers
        for worker in workers:
            worker.send({"cmd": "optim_step"})

        opt_metrics = [worker.recv() for worker in workers]
        lr = opt_metrics[0]["lr"]

        print(f"Step {step:3d}: loss={avg_loss:.4f}, grad_norm={avg_grad_norm:.2f}, lr={lr:.2e}")

    print("\n=== Training Complete ===\n")

    # === Step 5: Cleanup ===
    print("Shutting down cluster...")
    cluster.stop()
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Important Notes

### Before Running

1. **Update IP addresses:** Replace `192.168.1.10` and `192.168.1.11` with your actual node IPs
2. **Check SSH access:** Ensure passwordless SSH is set up:
   ```bash
   ssh-copy-id user@node2
   ```
3. **Sync code:** All nodes need the same codebase:
   ```bash
   # Option 1: Git
   git push && ssh node2 "cd /path/to/code && git pull"

   # Option 2: rsync
   rsync -avz --exclude .git . user@node2:/path/to/code/
   ```
4. **Open ports:** Ensure firewall allows:
   - Ports 10000-10003 (worker servers)
   - Port 29500 (NCCL coordination)
5. **Test locally first:** Single node before multi-node

### Module Path

The `work_fn` argument needs the correct Python module path:
- If running as script: `"__main__.training_work_fn"`
- If module: `"my_package.training.training_work_fn"`

### Troubleshooting

**"Connection refused":**
```bash
# Check if worker_server is accessible
nc -zv node2 10000
```

**NCCL hangs:**
```bash
# Check master node is reachable
ping 192.168.1.10

# Check NCCL port
nc -zv 192.168.1.10 29500
```

**Workers die silently:**
```bash
# Check worker_server logs on remote node
ssh node2 "tail -f /tmp/worker_server.log"
```

---

## TODO Before Production

- [ ] Test on actual 2-node cluster
- [ ] Add timeout handling for network failures
- [ ] Add retry logic for worker deaths
- [ ] Validate NCCL initialization thoroughly
- [ ] Add proper checkpoint saving/loading
- [ ] Test with real model (not dummy Sequential)
- [ ] Add monitoring/logging infrastructure
- [ ] Handle heterogeneous GPU configs
- [ ] Add graceful shutdown on errors

---

## Why This is Conceptual

This example is **untested** because:
- Requires 2-node cluster setup (not everyone has this)
- SSH configuration is environment-specific
- Network topology varies by institution
- NCCL behavior depends on hardware/drivers

**Use as a template, not production code!**
