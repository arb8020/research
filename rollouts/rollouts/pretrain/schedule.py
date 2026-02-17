"""Learning rate scheduling for pretraining.

Pure functions for computing learning rate at each step.
"""

from __future__ import annotations

import math


def get_lr(step: int, config: dict) -> float:
    """Compute learning rate for a given step.

    Linear warmup, then cosine decay to min_lr_ratio of peak.

    Config keys:
        steps: int - Total training steps (required)
        lr: float - Peak learning rate (required)
        warmup_steps: int - Linear warmup steps (default: 100)
        min_lr_ratio: float - Floor as ratio of peak (default: 0.1)

    Returns:
        Learning rate for this step
    """
    total_steps = config["steps"]
    peak_lr = config["lr"]
    warmup_steps = config.get("warmup_steps", 100)
    min_lr_ratio = config.get("min_lr_ratio", 0.1)

    assert total_steps > 0
    assert peak_lr > 0
    assert warmup_steps >= 0
    assert 0 < min_lr_ratio <= 1

    # Warmup phase: linear 0 -> peak
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps

    # Decay phase: cosine peak -> floor
    decay_steps = total_steps - warmup_steps
    if decay_steps <= 0:
        return peak_lr

    progress = (step - warmup_steps) / decay_steps
    # Cosine decay from 1.0 to min_lr_ratio
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    lr_ratio = min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    return peak_lr * lr_ratio
