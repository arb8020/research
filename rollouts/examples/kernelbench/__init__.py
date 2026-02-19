"""KernelBench: Multi-turn RL for GPU kernel optimization.

Based on Kevin: Multi-Turn RL for Generating CUDA Kernels (arXiv:2507.11948)

Usage:
    # Evaluate with API model
    python run_eval.py configs/api_smoke.py

    # Evaluate with SGLang endpoint
    python run_eval.py configs/sglang_smoke.py

    # Run GRPO training
    python run_train.py configs/grpo_level1.py --provision --provider runpod
"""

from .dataset import load_kernelbench_dataset, load_kernelbench_prompts
from .scoring import kernelbench_score_fn

__all__ = [
    "load_kernelbench_dataset",
    "load_kernelbench_prompts",
    "kernelbench_score_fn",
]
