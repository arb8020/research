"""Smoke test config for KernelBench with local SGLang endpoint.

Runs problems against a local SGLang server for testing self-hosted models.

Prerequisites:
    1. Start SGLang server:
       python -m sglang.launch_server --model Qwen/Qwen2.5-Coder-3B-Instruct --port 30000

    2. Or use the helper in run_eval.py (TODO)

Usage:
    python run_eval.py configs/sglang_smoke.py
    python run_eval.py configs/sglang_smoke.py --limit 5
"""

from pathlib import Path

from rollouts.core import Endpoint

# Endpoint configuration - local SGLang server
endpoint = Endpoint(
    model="sglang/Qwen/Qwen2.5-Coder-3B-Instruct",  # Model loaded in SGLang
    base_url="http://localhost:30000/v1",
    api_format="openai-chat",  # SGLang uses OpenAI-compatible API
)

# Dataset configuration
levels = [1]  # Level 1 = easiest problems
backend = "CUDA"
max_samples = 5

# Evaluation configuration
max_turns = 4
max_concurrent = 2  # SGLang can handle some parallelism

# Output
output_dir = Path("results/kernelbench/sglang_smoke")
eval_name = "kernelbench_sglang_smoke"
verbose = True

# No remote sandboxes - use local subprocess for GPU scoring
sandbox_configs = []
