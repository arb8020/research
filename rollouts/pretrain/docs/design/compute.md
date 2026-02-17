# Compute Reference

> How to get GPUs for pretraining experiments.

## Quick Reference

| Provider | GPU | $/hr per GPU | 8x Cost | Max Runtime |
|----------|-----|--------------|---------|-------------|
| Modal | H100 80GB | $3.95 | $31.60/hr | 24 hours |
| Modal | H200 141GB | $4.54 | $36.32/hr | 24 hours |
| Modal | A100 80GB | $2.50 | $20.00/hr | 24 hours |
| RunPod | H100 80GB | ~$3.50 | ~$28/hr | Unlimited |
| LambdaLabs | H100 80GB | $2.49 | $19.92/hr | Unlimited |

---

## Modal (Primary)

### Setup

```bash
pip install modal
modal setup  # Creates ~/.modal.toml with auth
```

### Usage

```python
import modal

app = modal.App("pretrain")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "tiktoken")
)

@app.function(
    image=image,
    gpu="H100:8",  # 8x H100 (80GB each)
    timeout=24 * 60 * 60,  # 24 hour max
)
def train(config: dict):
    # Training code here
    pass

# Run
modal run train.py
```

### GPU Options

```python
# Single GPU
gpu="H100"
gpu="A100"
gpu="L40S"

# Multi-GPU (same node)
gpu="H100:8"   # 8x H100
gpu="A100:4"   # 4x A100

# Blackwell (when available)
gpu="B200:8"
```

### Code Sync

Modal mounts local code automatically. For large repos, use git bundle:

```python
# In run script
import subprocess
subprocess.run(["git", "bundle", "create", "/tmp/repo.bundle", "HEAD"])

# Mount bundle
@app.function(
    mounts=[modal.Mount.from_local_file("/tmp/repo.bundle", "/root/repo.bundle")]
)
def train():
    subprocess.run(["git", "clone", "/root/repo.bundle", "/root/pretrain"])
    # ...
```

---

## broker (Multi-Provider Abstraction)

Location: `/Users/chiraagbalu/research/broker`

### Usage

```python
from broker import ProvisionRequest, get_provider

request = ProvisionRequest(
    gpu_type="H100",
    gpu_count=8,
    max_price_per_gpu=5.0,
)

provider = get_provider("modal")  # or "runpod", "lambdalabs", "vast"
instance = await provider.provision(request)

# SSH access
await instance.run_command("nvidia-smi")

# Cleanup
await instance.terminate()
```

### Supported Providers

- **Modal**: `broker/providers/modal.py`
- **RunPod**: `broker/providers/runpod.py` (GraphQL API)
- **LambdaLabs**: `broker/providers/lambdalabs.py`
- **Vast.ai**: `broker/providers/vast.py`
- **DigitalOcean**: `broker/providers/digitalocean.py`

---

## miniray (Multi-Node)

Location: `/Users/chiraagbalu/research/miniray`

For 2x8x (16 GPU) training across 2 nodes:

```python
from miniray import Cluster, NodeConfig

cluster = Cluster(nodes=[
    NodeConfig("node1.example.com", num_workers=8, base_port=10000),
    NodeConfig("node2.example.com", num_workers=8, base_port=10000),
])

# Launch training workers
cluster.launch_servers(work_fn="pretrain.train:worker_main")

# Connect and send config
workers = cluster.connect_to_servers()
for i, worker in enumerate(workers):
    worker.send({
        "rank": i,
        "world_size": 16,
        "master_addr": "node1.example.com",
        "config": config_dict,
    })

# Wait for completion
for worker in workers:
    result = worker.recv()
```

### NCCL Setup (from miniray/nccl.py)

```python
from miniray.nccl import NCCLConfig, setup_nccl_env

config = NCCLConfig(
    master_addr="192.168.1.10",
    master_port=29500,
    world_size=16,
    rank=rank,
    local_rank=local_rank,
)

# Sets MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK, LOCAL_RANK, CUDA_VISIBLE_DEVICES
setup_nccl_env(config)

import torch.distributed as dist
dist.init_process_group(backend="nccl", init_method="env://")
```

---

## Local Development

### Single GPU

```bash
python -m pretrain.train --config configs/tiny.toml
```

### 8x GPU (Single Node)

```bash
torchrun --standalone --nproc_per_node=8 -m pretrain.train --config configs/small.toml
```

### Environment Variables

```bash
# torchrun sets these automatically
RANK=0          # Global rank
WORLD_SIZE=8    # Total processes
LOCAL_RANK=0    # Rank on this node

# Optional
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NCCL_DEBUG=INFO
```

---

## Cost Estimation

### Training Time Estimates

For reference, modded-nanogpt achieves:
- ~170k tok/s on 8x H100 (GPT-2 124M)
- ~45 minutes for 10B tokens

Scaling:
- 1B params: ~10x slower per token
- 100B tokens: ~10x longer

### Budget Examples

| Experiment | Tokens | Est. Time | Cost (8xH100) |
|------------|--------|-----------|---------------|
| Tiny test | 1B | ~1 hour | ~$32 |
| Small run | 10B | ~10 hours | ~$320 |
| Medium run | 100B | ~100 hours | ~$3,200 |

---

## Reproducible Environments

### Docker

```dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

RUN pip install numpy tiktoken

COPY . /app
WORKDIR /app

CMD ["torchrun", "--standalone", "--nproc_per_node=8", "-m", "pretrain.train"]
```

### NixOS (if available)

```nix
# flake.nix
{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }: {
    devShells.x86_64-linux.default = nixpkgs.legacyPackages.x86_64-linux.mkShell {
      packages = with nixpkgs.legacyPackages.x86_64-linux; [
        python311
        python311Packages.torch-bin
        python311Packages.numpy
      ];
    };
  };
}
```

---

## Monitoring

### From Modal

```bash
modal app logs pretrain
```

### From SSH (broker/miniray)

```python
# miniray includes a logs server
from miniray import LogsServer

server = LogsServer(port=9100, watch_dir="/output")
server.serve_forever()

# Client
from miniray import RemoteWorker
monitor = RemoteWorker("node1.example.com", 9100)
monitor.send({"cmd": "tail", "file": "metrics/step_00001000.parquet"})
```

### Query Metrics

```bash
# Install duckdb CLI
pip install duckdb

# Query parquet files
duckdb -c "
SELECT step, value as loss
FROM 'output/metrics/*.parquet'
WHERE tag = 'loss'
ORDER BY step
"
```
