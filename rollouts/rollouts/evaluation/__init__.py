"""Evaluation utilities for rollouts.

Two evaluation paradigms:

1. Standard benchmarks (lm_eval wrapper):
   from rollouts.evaluation.lm_eval import run_lm_eval
   results = run_lm_eval(base_url="...", tasks=["mmlu", "hellaswag"])

2. Rollouts native eval (agentic tasks, RL training):
   from rollouts.evaluation import evaluate
   results = await evaluate(samples, eval_config)
"""

from .lm_eval import run_lm_eval

__all__ = ["run_lm_eval"]
