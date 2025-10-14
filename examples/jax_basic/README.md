# JAX Basic Integration Test

End-to-end integration test validating the full broker → bifrost → GPU workflow.

## What it does

1. Uses **broker** to provision a cheap GPU instance
2. Uses **bifrost** to deploy and run a JAX script on the GPU
3. Verifies GPU is detected and can perform computation
4. Cleans up by terminating the instance

## Prerequisites

```bash
# Set environment variables
export RUNPOD_API_KEY="your-runpod-key"
export VAST_API_KEY="your-vast-key"  # optional
export SSH_KEY_PATH="~/.ssh/id_ed25519"  # optional, defaults to this
```

## Running

```bash
# From repository root
uv run python examples/jax_basic/run_integration_test.py
```

## Cost

Estimated cost: $0.10-0.30 per run (uses cheapest available GPU, terminates immediately)

## What gets validated

- ✅ Broker can search and provision GPUs
- ✅ SSH connection establishment
- ✅ Bifrost code deployment
- ✅ GPU driver/CUDA functionality
- ✅ Python environment bootstrap with uv
- ✅ JAX GPU detection and computation
