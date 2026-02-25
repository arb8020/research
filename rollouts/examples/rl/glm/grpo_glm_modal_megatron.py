"""GLM-4.7-Flash GRPO training on Modal with Megatron backend.

Uses Megatron instead of TorchTitan to avoid torch.nn.attention.varlen dependency.

Run with:
    python -m rollouts.modal_runner --config examples/rl/glm/grpo_glm_modal_megatron.py
"""

from examples.rl.glm.base_config import train as _base_train
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
# Hardware Configuration for Modal (Megatron backend)
# =============================================================================

# Dependencies - no torchtitan, uses Megatron instead
GLM_DEPS = DepsConfig(
    python_version="3.12",
    system_packages=(
        "bash",
        "curl",
        "git",
        "build-essential",
        "libnuma1",
        "tmux",
    ),
    pip_packages=(
        # PyTorch with CUDA 12.8
        "torch>=2.5.0",
        "torchvision",
        "torchaudio",
        "flashinfer-python",
        # HF stack
        "hf-transfer",
        "datasets>=4.4.1",
        "accelerate>=0.20.0",
        "peft>=0.7.0",
        # Megatron deps (apex removed - requires compilation)
        "einops",
        # Utils
        "openai",
        "anthropic",
        "dacite",
        "aiohttp",
        "trio",
        "httpx",
        "markdownify",
    ),
    pip_index_url="https://download.pytorch.org/whl/nightly/cu128",
    pip_extra_index_url="https://pypi.org/simple",
    bootstrap_commands=(
        # Install uv first
        "curl -LsSf https://astral.sh/uv/install.sh | sh && . ~/.local/bin/env",
        # Install sglang first (pins older transformers)
        "~/.local/bin/uv pip install --system 'sglang[all] @ git+https://github.com/sgl-project/sglang.git@main#subdirectory=python'",
        # Force-upgrade transformers/hf_hub on top
        "~/.local/bin/uv pip install --system --upgrade 'transformers>=5.0.0' 'huggingface-hub>=1.4.0'",
        # Clone Megatron-LM for megatron.core imports
        "git clone --depth 1 https://github.com/NVIDIA/Megatron-LM.git /root/Megatron-LM",
    ),
)

hardware = HardwareConfig(
    gpu_type="A100-80GB",
    gpu_count=8,
    provider="modal",
    deps=GLM_DEPS,
    use_torchrun=False,  # Training script handles DDP internally
)

# =============================================================================
# Training Configuration
# =============================================================================

MODEL_NAME = "zai-org/GLM-4.7-Flash"

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="glm_4.7_flash_grpo_modal_megatron"),
    model=ModelConfig(
        name=MODEL_NAME,
        dtype="bfloat16",
    ),
    trainer=TrainerConfig(
        backend="megatron",
        lr=1e-6,
        weight_decay=0.01,
        max_grad_norm=1.0,
        num_minibatches=8,
        loss_type="vanilla",
        # GPU assignment: inference on GPU 0, training on GPUs 1-7
        cuda_device_ids=(1, 2, 3, 4, 5, 6, 7),
        # Megatron parallelism settings
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        sequence_parallel=False,
        activation_checkpointing=True,
    ),
    inference=InferenceConfig(
        backend="sglang",
        cuda_device_ids=(0,),
        mem_fraction=0.9,
        startup_timeout=600.0,
    ),
    rollout=RolloutConfig(
        batch_size=4,
        n_samples_per_prompt=8,
        temperature=0.7,
        max_seq_len=2048,
        max_tokens=256,
    ),
    checkpoint=CheckpointConfig(
        num_steps=50,
        checkpoint_every=10,
        sync_weights_every=1,
        weight_sync_mode="disk",
    ),
)


def train(config: GRPOConfig | None = None, **kwargs: object) -> dict:
    """Run GLM GRPO training with Megatron backend."""
    return _base_train(config=config, **kwargs)


if __name__ == "__main__":
    train(config)
