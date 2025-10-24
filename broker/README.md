# Broker

Provision and manage cloud GPU instances across multiple providers.

## CLI Reference

```bash
broker init                                    # Create .env template
broker search --gpu-type "RTX 4090"            # Search available GPUs
broker create --gpu-type "H100" --name my-job  # Provision instance
broker list                                    # List your instances
broker status <instance-id>                    # Get instance status
broker ssh <instance-id>                       # Get SSH connection string
broker info <instance-id>                      # Get system info (GPU/CPU/memory)
broker terminate <instance-id>                 # Terminate instance
```

**Global options:** `--credentials`, `--ssh-key`, `--quiet`, `--json`, `--debug`

## Python API Reference

```python
from broker import search, create, list_instances, terminate

# Search for GPUs
offers = search(
    gpu_type="H100",
    max_price_per_hour=2.0,
    gpu_count=1,
    provider="runpod"  # optional: runpod, primeintellect, lambdalabs
)

# Provision instance
instance = create(
    gpu_type="H100",
    name="my-job",
    credentials={"runpod": "your-api-key"}
)
print(f"SSH: {instance.ssh_connection}")

# List instances
instances = list_instances(credentials={"runpod": "your-api-key"})

# Terminate
terminate(instance.instance_id, credentials={"runpod": "your-api-key"})
```

## Setup

Create `.env` file:
```bash
RUNPOD_API_KEY=your_key
PRIME_API_KEY=your_key
LAMBDA_API_KEY=your_key
SSH_KEY_PATH=~/.ssh/id_ed25519
```

## Implementation Details

- **Providers**: RunPod, Prime Intellect, Lambda Labs
- **Query syntax**: Pandas-style queries supported (e.g., `gpus.gpu_type.contains("A100") & gpus.price_per_hour < 2.0`)
- **SSH keys**: Automatically injected into instances for secure access
- **Validation**: GPU types, providers, and pricing validated before provisioning
