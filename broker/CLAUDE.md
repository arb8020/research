# broker

Unified GPU cloud provisioning across RunPod, Prime Intellect, Lambda Labs, Vast.ai, and DigitalOcean.

## Auth setup

```bash
broker auth status          # check configured credentials
broker auth login runpod    # add/update a provider key
broker auth switch <profile> # switch credential profile
```

## Key CLI commands

```bash
# Find available GPUs
broker search --gpu-type A100
broker search --gpu-type H100 --max-price 3.0

# See what's running (use this before provisioning to avoid duplicates)
broker list

# Provision
broker create --gpu-type A100 --name my-job

# Manage instances
broker status <instance-id>
broker ssh <instance-id>          # get SSH connection string
broker exec <instance-id> "nvidia-smi"
broker logs <instance-id>         # fetch pod logs
broker terminate <instance-id>
broker cleanup                    # terminate ALL running instances
```

## Python API

```python
from broker.client import GPUClient
from broker.credentials import get_credentials

client = GPUClient(credentials=get_credentials())

# Search
offers = await client.search(client.gpu_type.contains("A100"))

# Provision
instance = await client.create(
    client.gpu_type.contains("A100"),
    gpu_count=1,
    name="my-job",
    exposed_ports=[9100],
)
await instance.wait_until_ssh_ready(timeout=600)

# Use
result = instance.exec("nvidia-smi")
print(result.stdout)

# Clean up
await instance.terminate()
```

## Credentials

Precedence: CLI flag > env vars > `~/.broker/credentials.toml`

Env vars: `RUNPOD_API_KEY`, `PRIME_API_KEY`, `LAMBDA_API_KEY`, `VAST_API_KEY`

## Key gotchas

- Always run `broker list` before provisioning — previous attempts may have left pods running (especially on primeintellect which can get stuck in `pending`)
- `--provider runpod` to force a specific provider when the default picks a stuck one
- `AccountError` = credentials/billing issue (not retryable); `ProvisionError` = this attempt failed (may retry)
- SSH readiness takes 1-10 min after provisioning depending on provider
