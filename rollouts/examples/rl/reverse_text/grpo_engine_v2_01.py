"""Reverse Text GRPO with engine_v2 (rollouts native inference).

Minimal 2-GPU setup: 1 GPU for inference (engine_v2), 1 GPU for training.
This tests the rollouts-native inference engine instead of SGLang.

Architecture:
    GPU 0: Inference engine (engine_v2)
    GPU 1: Training (FSDP/DDP)

Run with:
    python rollouts/run_rl.py --config examples/rl/reverse_text/grpo_engine_v2_01.py
"""

from examples.rl.reverse_text.base_config import train as _base_train
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
# Hardware Configuration
# =============================================================================

hardware = HardwareConfig(
    gpu_type="RTX A5000",
    gpu_count=2,
    provider="runpod",
)

# =============================================================================
# Training Configuration
# =============================================================================

DEFAULT_MODEL = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="reverse_text_engine_v2"),
    model=ModelConfig(name=DEFAULT_MODEL),
    checkpoint=CheckpointConfig(
        num_steps=100,
        checkpoint_every=20,
        sync_weights_every=1,
        pipeline_mode="stop_and_sync",  # Simpler for initial testing
        weight_sync_mode="disk",  # Start with disk sync
    ),
    rollout=RolloutConfig(
        batch_size=8,
        n_samples_per_prompt=16,
        temperature=1.0,
        max_seq_len=512,
        max_tokens=128,
    ),
    trainer=TrainerConfig(
        cuda_device_ids=(1,),  # GPU 1 for training
        lr=3e-6,
        num_minibatches=32,
        loss_type="masked",
    ),
    inference=InferenceConfig(
        backend="engine_v2",  # Use rollouts native inference
        cuda_device_ids=(0,),  # GPU 0 for inference
        port=30000,
        mem_fraction=0.8,
        tensor_parallel_size=1,
    ),
)


def train(config: GRPOConfig | None = None, **kwargs: object) -> dict:
    """Run 2-GPU training with engine_v2."""
    return _base_train(config=config, **kwargs)


if __name__ == "__main__":
    train()
