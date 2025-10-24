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
export PRIME_API_KEY="your-prime-key"  # optional
export LAMBDA_API_KEY="your-lambda-key"  # optional
export SSH_KEY_PATH="~/.ssh/id_ed25519"  # optional, defaults to this
```

## Running

```bash
# Full integration test (broker + bifrost + JAX)
uv run python examples/jax_basic/run_integration_test.py
uv run python examples/jax_basic/run_integration_test.py --provider lambdalabs  # specific provider

# Lambda Labs provider test (search + provision + nvidia-smi)
uv run python examples/jax_basic/test_lambda_integration.py
uv run python examples/jax_basic/test_lambda_integration.py --skip-create  # only test search
uv run python examples/jax_basic/test_lambda_integration.py --gpu-type H100  # filter by GPU type
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
