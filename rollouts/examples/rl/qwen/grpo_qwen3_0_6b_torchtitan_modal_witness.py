"""Modal witness run for TorchTitan + Prime-style reverse-text RL.

This keeps our system and contract-native RL path, but borrows the known-good
Prime-CI reverse-text sync recipe:

- model: PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT
- lr: 3e-6
- loss_type: masked
- batch_size: 8
- n_samples_per_prompt: 8
- sync pipeline semantics

It is the preferred first remote witness over the generic RunPod config because
Modal reduces provisioning drift, while still exercising the real trainer path.
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

QWEN_TORCHTITAN_MODAL_DEPS = DepsConfig(
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
        "torch>=2.10.0",
        "torchvision",
        "torchaudio",
        "sglang[all] @ git+https://github.com/sgl-project/sglang.git@main#subdirectory=python",
        "transformers>=5.0.0",
        "huggingface-hub>=1.4.0",
        "datasets>=4.4.1",
        "accelerate>=0.20.0",
        "peft>=0.7.0",
        "hf-transfer",
        "torchtitan @ git+https://github.com/pytorch/torchtitan.git@v0.2.2",
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
    pip_prerelease=True,
    bootstrap_commands=(),
)

hardware = HardwareConfig(
    gpu_type="A100",
    gpu_count=2,
    provider="modal",
    deps=QWEN_TORCHTITAN_MODAL_DEPS,
    use_torchrun=False,
)

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="qwen3_0_6b_torchtitan_modal_witness"),
    model=ModelConfig(
        name="PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT",
        dtype="bfloat16",
    ),
    trainer=TrainerConfig(
        backend="torchtitan",
        torchtitan_model="qwen3",
        torchtitan_model_size="0.6B",
        lr=3e-6,
        weight_decay=0.0,
        max_grad_norm=1.0,
        num_minibatches=8,
        loss_type="masked",
        cuda_device_ids=(1,),
    ),
    inference=InferenceConfig(
        backend="sglang",
        cuda_device_ids=(0,),
        mem_fraction=0.45,
        startup_timeout=600.0,
    ),
    rollout=RolloutConfig(
        batch_size=8,
        n_samples_per_prompt=8,
        temperature=1.0,
        max_seq_len=512,
        max_tokens=128,
    ),
    checkpoint=CheckpointConfig(
        num_steps=8,
        log_every=1,
        checkpoint_every=4,
        sync_weights_every=1,
        weight_sync_mode="disk",
        pipeline_mode="sync",
        max_lag=0,
        pipeline_queue_size=0,
    ),
)


def train(config: GRPOConfig | None = None, **kwargs: object) -> dict:
    return _base_train(config=config or globals()["config"], **kwargs)
