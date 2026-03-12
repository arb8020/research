"""Smoke test config for KernelBench with API models.

Runs a few Level 1 problems against Claude to verify the pipeline works.

Usage:
    python run_eval.py configs/api_smoke.py
    python run_eval.py configs/api_smoke.py --model openai/gpt-4o
"""

from pathlib import Path

from rollouts.core import Endpoint

# Endpoint configuration
endpoint = Endpoint(
    model="anthropic/claude-sonnet-4-20250514",
    base_url="https://api.anthropic.com/v1",
    api_format="anthropic-messages",
)

# Dataset configuration
levels = [1]  # Level 1 = easiest problems
backend = "CUDA"
max_samples = 3  # Just a few for smoke test

# Evaluation configuration
max_turns = 4  # Allow 4 refinement attempts
max_concurrent = 1  # Sequential for smoke test

# Output
output_dir = Path("results/kernelbench/api_smoke")
eval_name = "kernelbench_api_smoke"
verbose = True

# No remote sandboxes - use local subprocess for GPU scoring
# (requires CUDA-capable GPU on this machine)
sandbox_configs = []
