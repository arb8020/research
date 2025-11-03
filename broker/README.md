# Broker

Unified GPU provisioning across RunPod, Prime Intellect, and Lambda Labs.

## Installation

```bash
uv pip install -e .
```

## Quick Start

```bash
# Setup credentials
broker init
# Edit .env with your API keys

# Search for GPUs
broker search --gpu-type "H100"

# Provision instance
broker create --gpu-type "H100" --name my-job
```

## Python API

```python
from broker import GPUClient

# Initialize with credentials
gpu_client = GPUClient(
    credentials={"runpod": "your-key"},
    ssh_key_path="~/.ssh/id_ed25519"
)

# Search using query DSL
query = (
    (gpu_client.gpu_type.contains("H100")) &
    (gpu_client.price_per_hour <= 2.0) &
    (gpu_client.vram_gb >= 80)
)

offers = gpu_client.search(query, sort=lambda x: x.price_per_hour)

# Provision instance
instance = gpu_client.create(
    query=query,
    gpu_count=8,
    container_disk_gb=250,
    sort=lambda x: x.price_per_hour,
    reverse=False  # cheapest first
)

# Wait for SSH
instance.wait_until_ssh_ready(timeout=900)
print(f"SSH: {instance.ssh_connection_string()}")

# Cleanup
gpu_client.terminate_instance(instance.id, instance.provider)
```

## Query DSL

Build complex queries with pandas-style syntax:

```python
# Price and GPU filtering
query = (
    gpu_client.gpu_type.contains("A100") &
    (gpu_client.price_per_hour <= 2.0)
)

# Multi-GPU with memory requirements
query = (
    (gpu_client.vram_gb >= 80) &
    (gpu_client.memory_gb >= 128) &
    (gpu_client.manufacturer == "Nvidia")
)

# Provider-specific
query = (
    (gpu_client.provider == "runpod") &
    (gpu_client.cloud_type == CloudType.COMMUNITY)
)
```

## CLI Reference

```bash
broker search --gpu-type "RTX 4090"            # Search GPUs
broker create --gpu-type "H100" --name my-job  # Provision
broker list                                    # List instances
broker status <instance-id>                    # Get status
broker ssh <instance-id>                       # SSH connection
broker info <instance-id>                      # System info
broker terminate <instance-id>                 # Terminate
```

## Configuration

Create `.env`:
```bash
RUNPOD_API_KEY=your_key
PRIME_API_KEY=your_key
LAMBDA_API_KEY=your_key
SSH_KEY_PATH=~/.ssh/id_ed25519
```
