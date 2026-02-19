"""Evaluation utilities for rollouts.

Two evaluation paradigms:

1. Standard benchmarks (lm_eval wrapper):
   from rollouts.evaluation.lm_eval import run_lm_eval
   results = run_lm_eval(base_url="...", tasks=["mmlu", "hellaswag"])

2. Rollouts native eval (agentic tasks, RL training):
   from rollouts.evaluation import evaluate
   results = await evaluate(samples, eval_config)
"""

# Re-export from the core evaluation module (../evaluation.py)
# This is necessary because the evaluation/ directory shadows evaluation.py
import importlib.util
import sys
from pathlib import Path

# Load evaluation.py directly
_eval_module_path = Path(__file__).parent.parent / "evaluation.py"
_spec = importlib.util.spec_from_file_location("rollouts._evaluation_core", _eval_module_path)
_eval_module = importlib.util.module_from_spec(_spec)
sys.modules["rollouts._evaluation_core"] = _eval_module
_spec.loader.exec_module(_eval_module)

# Re-export key items
evaluate = _eval_module.evaluate
evaluate_sample = _eval_module.evaluate_sample
EvalReport = _eval_module.EvalReport
EvalRuntime = _eval_module.EvalRuntime

from .lm_eval import run_lm_eval

__all__ = [
    "run_lm_eval",
    "evaluate",
    "evaluate_sample",
    "EvalReport",
    "EvalRuntime",
]
