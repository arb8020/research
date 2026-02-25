"""GLM-4.7-Flash GRPO training on Modal with 8x A100-80GB.

Modal sandbox-based training with:
- 8x A100-80GB: 1 for inference, 7 for training (FSDP)
- ~62GB per training GPU with FSDP/7
- Fast iteration (~30s cold start with cached image)

Run with:
    python -m rollouts.modal_runner --config examples/rl/glm/grpo_glm_modal.py

Note:
    GLM-4.7-Flash is a 30B MoE model with 3.6B active parameters.
    Requires 8x A100-80GB because FSDP needs 7+ GPUs to fit optimizer states.
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
# Hardware Configuration for Modal
# =============================================================================

# Dependencies for Modal image build
# Uses PyTorch nightly (cu128) for B200/Blackwell support
# torchtitan v0.2.2 requires PyTorch nightly (torch.nn.attention.varlen)
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
        # PyTorch nightly with CUDA 12.8 for B200
        "torch",
        "torchvision",
        "torchaudio",
        "flashinfer-python",
        # HF stack
        "hf-transfer",
        "datasets>=4.4.1",
        "accelerate>=0.20.0",
        "peft>=0.7.0",
        # Training
        "torchtitan @ git+https://github.com/pytorch/torchtitan.git@v0.2.2",
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
        # Install uv first (uv_pip_install installs it, but we need it for bootstrap)
        "curl -LsSf https://astral.sh/uv/install.sh | sh && . ~/.local/bin/env",
        # Install sglang first (pins older transformers) - use uv for speed
        "~/.local/bin/uv pip install --system 'sglang[all] @ git+https://github.com/sgl-project/sglang.git@main#subdirectory=python'",
        # Force-upgrade transformers/hf_hub on top (fixes sglang's pinned versions)
        "~/.local/bin/uv pip install --system --upgrade 'transformers>=5.0.0' 'huggingface-hub>=1.4.0'",
        # Clone Megatron-LM for megatron.core imports
        "git clone --depth 1 https://github.com/NVIDIA/Megatron-LM.git /root/Megatron-LM",
    ),
)

hardware = HardwareConfig(
    gpu_type="A100-80GB",  # 8x A100-80GB: 1 for inference, 7 for training
    gpu_count=8,
    provider="modal",
    deps=GLM_DEPS,
    use_torchrun=False,  # TorchTitan handles multi-GPU FSDP internally
)

# =============================================================================
# Training Configuration
# =============================================================================

# GLM-4.7-Flash: 30B MoE, 3.6B active
MODEL_NAME = "zai-org/GLM-4.7-Flash"

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="glm_4.7_flash_grpo_modal"),
    model=ModelConfig(
        name=MODEL_NAME,
        dtype="bfloat16",
    ),
    trainer=TrainerConfig(
        backend="torchtitan",
        torchtitan_model="glm",
        torchtitan_model_size="4.7-flash",
        lr=1e-6,
        weight_decay=0.01,
        max_grad_norm=1.0,
        num_minibatches=8,
        loss_type="vanilla",
        # GPU assignment: inference on GPU 0, training on GPUs 1-7 (7 GPUs for FSDP)
        cuda_device_ids=(1, 2, 3, 4, 5, 6, 7),
    ),
    inference=InferenceConfig(
        backend="sglang",
        cuda_device_ids=(0,),
        mem_fraction=0.9,  # GLM MoE needs more VRAM
        startup_timeout=600.0,  # 10 min - GLM-4.7 has 48 shards (~60GB) to download
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
        weight_sync_mode="disk",  # NCCL weight sync not yet supported for TorchTitan
    ),
)


def train(config: GRPOConfig | None = None, **kwargs: object) -> dict:
    """Run GLM GRPO training."""
    return _base_train(config=config, **kwargs)


if __name__ == "__main__":
    train(config)
