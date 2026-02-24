"""GLM-4.7-Flash GRPO training on Modal.

Modal sandbox-based training with:
- Fast iteration (~30s cold start with cached image)
- Persistent HF cache via Modal Volume (no re-downloading models)
- PyTorch nightly for B200/Blackwell support

Run with:
    python -m rollouts.modal_runner --config examples/rl/glm/grpo_glm_modal.py

Note:
    GLM-4.7-Flash is a 30B MoE model with 3.6B active parameters.
    First run downloads model (~60GB), subsequent runs use cached volume.
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
        # Install sglang first (pins older transformers)
        "pip install 'sglang[all] @ git+https://github.com/sgl-project/sglang.git@main#subdirectory=python'",
        # Force-upgrade transformers/hf_hub on top (fixes sglang's pinned versions)
        "pip install --upgrade 'transformers>=5.0.0' 'huggingface-hub>=1.4.0'",
        # Clone Megatron-LM for megatron.core imports
        "git clone --depth 1 https://github.com/NVIDIA/Megatron-LM.git /root/Megatron-LM",
    ),
)

hardware = HardwareConfig(
    gpu_type="A100",  # Start with A100 for testing, switch to B200 later
    gpu_count=2,  # 1 for inference, 1 for training
    provider="modal",
    deps=GLM_DEPS,
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
        # GPU assignment: inference on GPU 0, training on GPU 1
        cuda_device_ids=(1,),
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
