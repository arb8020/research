"""Debug config for SFT + RL pipeline - quick iteration on limited hardware.

Following experiment_config.md: Always include a debug config that allows
short runs on limited hardware.

This config runs BOTH:
1. SFT training: 100 steps on 1K SmolTalk samples
2. RL training: 50 steps on 500 GSM8K problems

Expected runtime: ~15-20 minutes on single GPU.

Tiger Style: ALL parameters explicit, no defaults.
"""

import sys
from pathlib import Path

# Add parent to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_config import (
    Config,
    DataConfig,
    DatasetSpec,
    ModelConfig,
    OutputConfig,
    RLConfig,
    SFTConfig,
    TargetConfig,
)

# Tiger Style: Construct config with ALL parameters explicit (no overrides, no defaults)
# Following qwen3_next pattern: configs/01_baseline.py
config = Config(
    target=TargetConfig(
        gpu_ranks=[4],  # Single GPU for debug (using GPU 4 from range 4-7)
        device_type="cuda",  # cuda|cpu|mps
    ),
    model=ModelConfig(
        name="Qwen/Qwen2.5-0.5B-Instruct",
        dtype="bfloat16",
        compile=False,  # Faster startup for debug
    ),
    data=DataConfig(
        # SFT data mixture (debug version)
        # Full nanochat SFT: ARC-Easy(2.3K) + ARC-Challenge(1.1K) + GSM8K(8K) + SmolTalk(10K) + identity(1K) + spelling(600)
        # Debug: Just 1K SmolTalk for fast iteration
        sft_mixture=[
            DatasetSpec(
                name="HuggingFaceTB/smol-smoltalk",
                split="train",
                subset=None,
                max_samples=1000,  # Just 1K for debug
                filepath=None,
                size=None,
                repeat=1,
            ),
        ],
        # RL data mixture (debug version)
        # Full nanochat RL: GSM8K main train (8K problems)
        # Debug: Just 500 problems
        rl_mixture=[
            DatasetSpec(
                name="openai/gsm8k",
                split="train",
                subset="main",
                max_samples=500,  # Just 500 for debug
                filepath=None,
                size=None,
                repeat=1,
            ),
        ],
        max_length=512,  # Shorter for debug
        shuffle_seed=42,
    ),
    sft=SFTConfig(
        # nanochat defaults with debug overrides
        num_epochs=1,
        num_iterations=100,  # Override epochs for debug (just 100 steps)
        batch_size=2,  # Smaller for lower memory
        target_examples_per_step=8,  # Smaller global batch
        unembedding_lr=0.004,  # nanochat default
        embedding_lr=0.2,  # nanochat default
        matrix_lr=0.02,  # nanochat default
        weight_decay=0.0,
        init_lr_frac=0.02,
        eval_every=25,  # More frequent for debug
        eval_steps=10,
        checkpoint_every=50,  # More frequent for debug
        log_every=10,
    ),
    rl=RLConfig(
        # nanochat defaults with debug overrides
        num_epochs=1,
        examples_per_step=8,  # Fewer examples per step for debug
        num_samples=8,  # Fewer samples per example for debug
        batch_size=4,  # Smaller batch for debug
        max_new_tokens=128,  # Shorter generations for debug
        temperature=1.0,
        top_k=50,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        init_lr_frac=0.05,
        eval_every=20,  # More frequent for debug
        eval_examples=100,  # Fewer eval examples
        save_every=20,
        baseline=0.0,
    ),
    output=OutputConfig(
        save_dir=Path("./results"),
        log_level="INFO",
        experiment_name="01_debug_sft_rl",
        use_wandb=False,
        wandb_project="integration_training",
        mode="sft+rl",  # Run both stages
        source_checkpoint=None,  # Will be set after SFT completes
    ),
)
