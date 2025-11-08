"""Rollouts: Lightweight framework for LLM evaluation with vision support.

Tiger Style evaluation framework inspired by llm-workbench/rollouts.
"""

from .rollout import (
    Message,
    Usage,
    Logprob,
    Logprobs,
    Choice,
    ChatCompletion,
    Rollout,
    Endpoint,
    generate,
    generate_exception,
    GSM8KSample,
    Sample,
)
from .evaluate import evaluate_dataset, evaluate_sample

__all__ = [
    # Core types
    "Message",
    "Usage",
    "Logprob",
    "Logprobs",
    "Choice",
    "ChatCompletion",
    "Rollout",
    "Endpoint",
    # Generate functions
    "generate",
    "generate_exception",
    # Evaluation
    "evaluate_dataset",
    "evaluate_sample",
    # Sample types
    "GSM8KSample",
    "Sample",
]

__version__ = "0.1.0"
