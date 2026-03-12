# Inference Serving Examples

Demo scripts for provisioning GPU instances and deploying SGLang servers.

## Setup

```bash
# Set required environment variables
export RUNPOD_API_KEY=...  # For GPU provisioning
export HF_TOKEN=...        # For gated models
```

## Scripts

### 1. provision_and_serve.py

Provisions a GPU and deploys an SGLang inference server.

```bash
# Provision new RunPod instance and deploy
python examples/provision_and_serve.py

# Use specific GPU type
python examples/provision_and_serve.py --gpu-type H100 --gpu-count 2

# Use specific model
python examples/provision_and_serve.py --model meta-llama/Llama-3.1-8B-Instruct

# Provision with existing network volume
python examples/provision_and_serve.py \
  --recipe recipes/examples/qwen3_0.6b_4090.py \
  --provider runpod \
  --persistent-volume-id <your-persistent-volume-id> \
  --persistent-volume-location <matching-datacenter-id> \
  --hf-cache-dir /workspace/.cache/huggingface

# Use existing instance (skip provisioning)
python examples/provision_and_serve.py --node-id runpod:abc123

# Use static SSH connection
python examples/provision_and_serve.py --ssh root@gpu.example.com:22
```

### 2. download_model_cache.py

Preloads model weights to a remote HF cache without starting SGLang.

```bash
# Download Qwen3-0.6B weights to mounted network volume cache
python examples/download_model_cache.py \
  --model Qwen/Qwen3-0.6B \
  --provider runpod \
  --persistent-volume-id <your-persistent-volume-id> \
  --persistent-volume-location <matching-datacenter-id> \
  --hf-cache-dir /workspace/.cache/huggingface

# Reuse existing node
python examples/download_model_cache.py --node-id runpod:abc123 --hf-cache-dir /workspace/.cache/huggingface

# Optional: write structured logs to a known path
python examples/download_model_cache.py \
  --model Qwen/Qwen3-0.6B \
  --provider runpod \
  --persistent-volume-id <your-persistent-volume-id> \
  --persistent-volume-location <matching-datacenter-id> \
  --hf-cache-dir /workspace/.cache/huggingface \
  --log-dir results/model-cache-smoke

# Tail progress while it runs
tail -f results/model-cache-smoke/events.jsonl
tail -f results/model-cache-smoke/download_output.log
```

### 3. query_server.py

Queries a deployed SGLang/vLLM server.

```bash
# Interactive mode (auto-detects model)
python examples/query_server.py --url http://gpu.example.com:30000/v1

# Single query
python examples/query_server.py --url http://gpu:30000/v1 --prompt "What is 2+2?"

# With streaming
python examples/query_server.py --url http://gpu:30000/v1 --prompt "Tell me a story" --stream

# Specify model explicitly
python examples/query_server.py --url http://gpu:30000/v1 --model "Qwen/Qwen2.5-7B-Instruct"
```

### 4. test_runpod_network_volume_reuse.py

End-to-end validation that network volume cache survives pod termination:
preload on pod A, terminate, then offline cache-only verify on fresh pod B.

```bash
python examples/test_runpod_network_volume_reuse.py \
  --persistent-volume-id <your-persistent-volume-id> \
  --persistent-volume-location <matching-datacenter-id> \
  --model Qwen/Qwen3-0.6B \
  --hf-cache-dir /workspace/.cache/huggingface \
  --log-dir results/network-volume-reuse-smoke

# Tail structured status + streamed output
tail -f results/network-volume-reuse-smoke/events.jsonl
tail -f results/network-volume-reuse-smoke/download_output.log
```

## Full Workflow

```bash
# 1. (Optional) Preload weights once to network volume
python examples/download_model_cache.py --model Qwen/Qwen3-0.6B --provider runpod --persistent-volume-id ... --persistent-volume-location ... --hf-cache-dir /workspace/.cache/huggingface

# 2. Provision and deploy
python examples/provision_and_serve.py --model Qwen/Qwen2.5-7B-Instruct
# Output: Server URL: http://xxx.xxx.xxx.xxx:30000
# 3. Provision and deploy from preloaded cache
python examples/provision_and_serve.py --recipe recipes/examples/qwen3_0.6b_4090.py --provider runpod --persistent-volume-id ... --persistent-volume-location ... --hf-cache-dir /workspace/.cache/huggingface

# 4. Query the server
python examples/query_server.py --url http://xxx.xxx.xxx.xxx:30000/v1 --prompt "Hello!"

# 5. When done, terminate the instance
# (command shown in provision_and_serve.py output)
```
