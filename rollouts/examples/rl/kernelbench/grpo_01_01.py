"""KernelBench GRPO baseline experiment.

Run with:
    # Uses hardware config from this file (default: RunPod A100)
    python rollouts/run_rl.py --config examples/rl/kernelbench/grpo_01_01.py

    # Override to run on Modal
    python rollouts/run_rl.py --config examples/rl/kernelbench/grpo_01_01.py --provider modal

    # Override to run locally
    python rollouts/run_rl.py --config examples/rl/kernelbench/grpo_01_01.py --local

Note:
    Unlike reverse_text which has an SFT-warmup model, this starts from
    a base model that hasn't been trained on kernel code. Initial rewards
    will likely be low (~0.1) until the model learns the task format.

    For better starting performance, consider:
    1. SFT warmup on working kernel examples first
    2. Using a code-specialized base model
    3. Starting with only Level 1 (easiest) problems
"""

from examples.rl.kernelbench.base_config import train  # noqa: F401 (used by runner)
from rollouts.training.configs import HardwareConfig
from rollouts.training.grpo import (
    CheckpointConfig,
    GRPOConfig,
    GRPOOutputConfig,
    InferenceConfig,
    ModelConfig,
    RolloutConfig,
    TrainerConfig,
)

# =============================================================================
# Hardware Configuration (what to provision)
# =============================================================================

hardware = HardwareConfig(
    gpu_type="A100",
    gpu_count=1,
    provider="runpod",  # Default to RunPod; override with --provider modal or --local
)

# =============================================================================
# Training Configuration
# =============================================================================

# Base model options:
# - "Nanbeige/Nanbeige4.1-3B" - Small, fast iteration, Chinese/English
# - "zai-org/GLM-4.7-Flash" - Larger, better capacity
# - "Qwen/Qwen2.5-Coder-3B-Instruct" - Code-specialized (may need different prompts)
DEFAULT_MODEL = "Nanbeige/Nanbeige4.1-3B"

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="kernelbench_grpo_01"),
    model=ModelConfig(name=DEFAULT_MODEL),
    checkpoint=CheckpointConfig(
        num_steps=50,
        checkpoint_every=10,
        sync_weights_every=1,  # On-policy
    ),
    rollout=RolloutConfig(
        batch_size=4,  # prompts per step
        n_samples_per_prompt=4,  # rollouts per prompt (4x4=16 total)
        temperature=0.7,
        max_seq_len=4096,  # Kernels can be long
        max_tokens=2048,  # Allow substantial responses
    ),
    trainer=TrainerConfig(
        lr=1e-6,  # Conservative for code generation
        num_minibatches=4,  # 16 total / 4 = micro_batch_size=4
        loss_type="masked",  # Importance sampling with ratio masking
    ),
    inference=InferenceConfig(
        mem_fraction=0.5,  # Leave room for kernel compilation during scoring
    ),
)
