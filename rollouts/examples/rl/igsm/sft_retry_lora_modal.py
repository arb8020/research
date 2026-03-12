"""LoRA SFT on retry data - Modal deployment.

This is the paper's "FAIL" condition - LoRA cannot learn error correction.
The paper shows this skill must be learned during pretraining, not added via
parameter-efficient finetuning.

From: pretrained clean checkpoint
To: model with LoRA adapters (should NOT learn retry behavior)

Paper result: LoRA finetuning fails to improve accuracy

Usage:
    python -m argus run --config examples/rl/igsm/sft_retry_lora_modal.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rollouts.training.configs import DepsConfig, HardwareConfig, ModelConfig

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
            "transformers>=4.50",
            "peft>=0.7.0",
            "numpy",
            "tiktoken",
        ),
        pip_index_url="https://download.pytorch.org/whl/cu124",
    ),
)

# =============================================================================
# Training Configuration
# =============================================================================


@dataclass
class LoRASFTConfig:
    """Config for LoRA SFT on iGSM retry data."""

    # Model
    model_name: str = "output/igsm_clean/checkpoint_100000"  # Pretrained checkpoint
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32

    # Training
    num_steps: int = 20_000
    batch_size: int = 64  # Smaller batch for LoRA
    lr: float = 1e-4  # Higher LR for LoRA
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 500

    # Logging
    log_every: int = 100
    checkpoint_every: int = 5_000
    output_dir: str = "output/igsm_sft_retry_lora"

    # iGSM
    difficulty: str = "med"
    max_op: int = 15
    max_edge: int = 20
    retry_rate: float = 0.2
    retry_type: str = "strong"
    max_seq_len: int = 768


config = LoRASFTConfig()

# For compatibility with existing infrastructure
model_config = ModelConfig(
    name=config.model_name,
    use_lora=config.use_lora,
    lora_rank=config.lora_rank,
    lora_alpha=config.lora_alpha,
)

igsm_config = {
    "difficulty": config.difficulty,
    "max_op": config.max_op,
    "max_edge": config.max_edge,
    "retry_rate": config.retry_rate,
    "retry_type": config.retry_type,
}


# =============================================================================
# Entry Point
# =============================================================================


def train(
    config: LoRASFTConfig | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run LoRA SFT training.

    This uses the rollouts SFT infrastructure with PEFT LoRA.
    """
    if config is None:
        from examples.rl.igsm.sft_retry_lora_modal import config as default_config

        config = default_config

    import subprocess
    import sys

    # Ensure iGSM is available
    igsm_path = Path("/tmp/iGSM")
    if not igsm_path.exists():
        print(f"Cloning iGSM to {igsm_path}...")
        subprocess.run(
            ["git", "clone", "https://github.com/facebookresearch/iGSM.git", str(igsm_path)],
            check=True,
        )
        # Create __init__.py files
        for subdir in ["tools", "data_gen", "data_gen/pretrain", "const"]:
            init_file = igsm_path / subdir / "__init__.py"
            if not init_file.exists():
                init_file.touch()

    sys.path.insert(0, str(igsm_path))

    # Import training components
    # Build iGSM retry data loader
    from rollouts.synthetic.igsm_retry import build_igsm_retry_loader
    from rollouts.training.backends.pytorch_factory import (
        create_pytorch_backend,
        parse_dtype,
    )
    from rollouts.training.loops.sft_loop import run_sft_training
    from rollouts.training.metrics import JSONLLogger
    from rollouts.training.types import SFTTrainingConfig, TrainingSample

    print("Building iGSM retry data loader...")
    loader = build_igsm_retry_loader(
        difficulty=config.difficulty,
        seq_len=config.max_seq_len,
        batch_size=config.batch_size,
        retry_rate=config.retry_rate,
        retry_type=config.retry_type,
        device="cuda",
        seed=42,
    )

    # Generate training samples
    print("Generating training samples...")
    samples = []
    for _ in range(config.num_steps * config.batch_size // 100):  # Generate batches worth
        input_ids, labels = loader.next()
        # Convert to samples
        for i in range(input_ids.shape[0]):
            samples.append(
                TrainingSample(
                    tokens=input_ids[i].tolist(),
                    loss_mask=[1.0] * len(input_ids[i]),  # Train on all tokens
                    response_length=len(input_ids[i]),
                )
            )

    print(f"Generated {len(samples)} training samples")

    # Create backend with LoRA
    print("Creating PyTorch backend with LoRA...")
    backend = create_pytorch_backend(
        model_name=config.model_name,
        checkpoint_dir=Path(config.output_dir),
        dtype=parse_dtype("bfloat16"),
        use_lora=config.use_lora,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lr=config.lr,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
    )

    # Create metrics logger
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_logger = JSONLLogger(output_path)

    # Create SFT config
    sft_config = SFTTrainingConfig(
        num_steps=config.num_steps,
        batch_size=config.batch_size,
        log_every=config.log_every,
        checkpoint_every=config.checkpoint_every,
    )

    # Run training
    print("Starting LoRA SFT training...")
    import trio

    metrics = trio.run(run_sft_training, backend, samples, sft_config, metrics_logger)

    print(f"Training complete! Final loss: {metrics[-1]['loss']:.4f}")
    return {
        "status": "completed",
        "output_dir": config.output_dir,
        "final_loss": metrics[-1]["loss"],
    }


if __name__ == "__main__":
    train()
