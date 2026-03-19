"""GRPO RL on iGSM from clean pretrained checkpoint - Modal deployment.

Key experiment: Can RL learn error-correction behavior that LoRA couldn't?

Paper showed:
- pretrain_clean → LoRA_retry → FAILS
- pretrain_retry → WORKS

Our question:
- pretrain_clean → GRPO (outcome reward only) → ???

If GRPO can learn self-correction from outcome reward alone, it would suggest
RL can discover behaviors that supervised finetuning cannot.

Usage:
    python -m argus run --config examples/rl/igsm/grpo_from_clean_modal.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

from rollouts.training.configs import DepsConfig, HardwareConfig
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
# Hardware Configuration (Modal)
# =============================================================================

hardware = HardwareConfig(
    gpu_type="A100",
    gpu_count=1,
    provider="modal",
    deps=DepsConfig(
        pip_packages=(
            "torch>=2.4",
            "transformers>=5.0",
            "datasets",
            "accelerate",
            "safetensors",
            "sglang[all] @ git+https://github.com/sgl-project/sglang.git@main#subdirectory=python",
            "curl_cffi",
            "huggingface_hub>=1.4.0",
            "numpy",
            "tiktoken",
        ),
        pip_index_url="https://download.pytorch.org/whl/cu124",
        pip_extra_index_url="https://pypi.org/simple",
    ),
)

# =============================================================================
# Training Configuration
# =============================================================================

# Path to pretrained clean checkpoint (must be set before running)
PRETRAIN_CHECKPOINT = "output/igsm_clean/checkpoint_100000"
DIFFICULTY: Literal["easy", "med", "hard"] = "med"

config = GRPOConfig(
    model=ModelConfig(
        name=PRETRAIN_CHECKPOINT,
        dtype="bfloat16",
    ),
    output=GRPOOutputConfig(
        experiment_name=f"igsm_{DIFFICULTY}_grpo_from_clean",
        output_dir="output",
    ),
    checkpoint=CheckpointConfig(
        num_steps=1000,  # Total RL steps
        checkpoint_every=100,
        sync_weights_every=1,
        pipeline_mode="sync",
        weight_sync_mode="disk",
    ),
    rollout=RolloutConfig(
        n_samples_per_prompt=8,  # Paper used 8 for GSM8K-style
        temperature=0.8,
        max_seq_len=1024,
        max_tokens=512,
        batch_size=8,
    ),
    trainer=TrainerConfig(
        lr=1e-6,  # Low LR for RL on pretrained model
        loss_type="clipped",  # PPO-style clipping
        num_minibatches=8,
        max_grad_norm=1.0,
    ),
    inference=InferenceConfig(
        cuda_device_ids=(0,),
        port=30000,
        mem_fraction=0.7,
        tensor_parallel_size=1,
    ),
)


# =============================================================================
# Entry Point
# =============================================================================


def train(
    config: GRPOConfig | None = None,
    max_samples: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run GRPO training on iGSM."""
    if config is None:
        from examples.rl.igsm.grpo_from_clean_modal import config as default_config

        config = default_config

    # Ensure iGSM is available
    import subprocess

    igsm_path = Path("/tmp/iGSM")
    if not igsm_path.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/facebookresearch/iGSM.git", str(igsm_path)],
            check=True,
        )
        for subdir in ["tools", "data_gen", "data_gen/pretrain", "const"]:
            init_file = igsm_path / subdir / "__init__.py"
            if not init_file.exists():
                init_file.touch()

    sys.path.insert(0, str(igsm_path))
    sys.path.insert(0, str(Path(__file__).parent))

    from base_config import igsm_score_fn, load_igsm_prompts

    from rollouts.environments.no_tools import BasicEnvironment
    from rollouts.training.grpo import grpo_train
    from rollouts.training.scoring import FunctionScorer

    # Load prompts (different seed than pretrain for held-out problems)
    prompts = load_igsm_prompts(
        max_samples=max_samples or 1000,
        difficulty=DIFFICULTY,
        seed=12345,
        mode="train",
    )

    print(f"Loaded {len(prompts)} iGSM prompts")

    return grpo_train(
        config=config,
        prompts=prompts,
        scorer=FunctionScorer(igsm_score_fn),
        environment_cls=BasicEnvironment,
    )


if __name__ == "__main__":
    train()
