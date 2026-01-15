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

# Use existing instance (skip provisioning)
python examples/provision_and_serve.py --node-id runpod:abc123

# Use static SSH connection
python examples/provision_and_serve.py --ssh root@gpu.example.com:22
```

### 2. query_server.py

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

## Full Workflow

```bash
# 1. Provision and deploy
python examples/provision_and_serve.py --model Qwen/Qwen2.5-7B-Instruct
# Output: Server URL: http://xxx.xxx.xxx.xxx:30000

# 2. Query the server
python examples/query_server.py --url http://xxx.xxx.xxx.xxx:30000/v1 --prompt "Hello!"

# 3. When done, terminate the instance
# (command shown in provision_and_serve.py output)
```
