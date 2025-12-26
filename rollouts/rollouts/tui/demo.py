#!/usr/bin/env python3
"""Demo script to test the training monitor TUI.

Generates fake training logs and metrics to stdin.

Usage:
    python -m rollouts.tui.demo | python -m rollouts.tui.monitor
"""

import json
import math
import random
import time


def main() -> None:
    """Generate fake training logs."""
    step = 0

    # Simulate training loop
    while True:
        # Metrics every step
        loss = 1.0 * math.exp(-step / 100) + random.gauss(0, 0.05)
        reward = 0.3 + 0.5 * (1 - math.exp(-step / 50)) + random.gauss(0, 0.02)
        kl = 0.01 + random.gauss(0, 0.002)

        metrics = {
            "step": step,
            "timestamp": time.time(),
            "loss": max(0, loss),
            "reward": reward,
            "kl_div": max(0, kl),
        }
        print(json.dumps(metrics), flush=True)

        # Training log every 5 steps
        if step % 5 == 0:
            log = {
                "logger": "training",
                "level": "INFO",
                "message": f"Step {step}: loss={loss:.4f}, reward={reward:.4f}",
            }
            print(json.dumps(log), flush=True)

        # SGLang log every 10 steps
        if step % 10 == 0:
            log = {
                "logger": "sglang.server",
                "level": "INFO",
                "message": f"Processed batch {step // 10}, latency={random.uniform(0.1, 0.5):.3f}s",
            }
            print(json.dumps(log), flush=True)

        # Occasional warnings
        if random.random() < 0.05:
            log = {
                "logger": "training",
                "level": "WARNING",
                "message": f"High KL divergence at step {step}: {kl:.4f}",
            }
            print(json.dumps(log), flush=True)

        step += 1
        time.sleep(0.2)  # 5 steps per second


if __name__ == "__main__":
    main()
