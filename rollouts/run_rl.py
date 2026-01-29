#!/usr/bin/env python3
"""Run RL training.

Usage:
    uv run run_rl.py --config examples/rl/calculator/grpo_01_01.py
    uv run run_rl.py --config examples/rl/calculator/grpo_01_01.py --provision
    uv run run_rl.py --config examples/rl/calculator/grpo_01_01.py --node-id runpod:abc123
"""

from rollouts.run import main

if __name__ == "__main__":
    main()
