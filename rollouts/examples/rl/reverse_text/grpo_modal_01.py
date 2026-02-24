"""Reverse Text GRPO on Modal.

Single A100 setup running on Modal sandbox.

Run with:
    python -m rollouts.run --config examples/rl/reverse_text/grpo_modal_01.py
"""

from examples.rl.reverse_text.base_config import train as _base_train
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
# Hardware Configuration
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
            "peft",
            "huggingface_hub>=1.4.0",
        ),
        pip_index_url="https://download.pytorch.org/whl/cu124",
        pip_extra_index_url="https://pypi.org/simple",
    ),
)

# =============================================================================
# Training Configuration
# =============================================================================

DEFAULT_MODEL = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="reverse_text_modal"),
    model=ModelConfig(name=DEFAULT_MODEL),
    checkpoint=CheckpointConfig(
        num_steps=100,
        checkpoint_every=20,
        sync_weights_every=1,
        pipeline_mode="sync",  # Simpler for single GPU
        weight_sync_mode="disk",
    ),
    rollout=RolloutConfig(
        batch_size=8,
        n_samples_per_prompt=8,
        temperature=1.0,
        max_seq_len=1024,
        max_tokens=256,
    ),
    trainer=TrainerConfig(
        cuda_device_ids=(0,),
        lr=3e-6,
        num_minibatches=16,
        loss_type="masked",
    ),
    inference=InferenceConfig(
        cuda_device_ids=(0,),  # Shared GPU
        port=30000,
        mem_fraction=0.7,
        tensor_parallel_size=1,
    ),
)


def train(config: GRPOConfig | None = None, **kwargs: object) -> dict:
    """Run training on Modal."""
    return _base_train(config=config, **kwargs)


if __name__ == "__main__":
    train()
