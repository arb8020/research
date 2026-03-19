"""Modal witness run for TorchTitan + vLLM reverse-text RL.

This keeps the contract-native RL path and the Prime-style reverse-text recipe,
but swaps the earlier unsupported TorchTitan+SGLang shared env attempt for a
TorchTitan+vLLM stack closer to `/tmp/torchforge`.
"""

from examples.rl.reverse_text.base_config import train as _base_train
from rollouts.training.configs import DepsConfig, HardwareConfig
from rollouts.training.grpo import (
    CheckpointConfig,
    GRPOConfig,
    GRPOOutputConfig,
    InferenceConfig,
    ModelConfig,
    ResourceWatchdogConfig,
    RolloutConfig,
    TrainerConfig,
)

QWEN_TORCHTITAN_MODAL_BASE_DEPS = DepsConfig(
    python_version="3.12",
    system_packages=(
        "bash",
        "curl",
        "git",
        "build-essential",
        "libnuma1",
        "tmux",
    ),
    pip_index_url="https://download.pytorch.org/whl/cu128",
    pip_extra_index_url="https://pypi.org/simple",
    pip_prerelease=True,
    bootstrap_commands=(),
)

QWEN_TORCHTITAN_VLLM_TRAINER_DEPS = DepsConfig(
    python_version="3.12",
    pip_packages=(
        "torch==2.9.0",
        "torchtitan==0.2.0",
        "torchmonarch==0.2.0",
        "torchstore @ git+https://github.com/meta-pytorch/torchstore.git@no-monarch-2026.01.05",
        "datasets>=2.21.0",
        "tokenizers",
        "accelerate>=0.20.0",
        "peft>=0.7.0",
        "hf-transfer",
        "openai",
        "anthropic",
        "dacite",
        "aiohttp",
        "trio",
        "httpx",
        "markdownify",
    ),
)

QWEN_TORCHTITAN_VLLM_INFERENCE_DEPS = DepsConfig(
    python_version="3.12",
    pip_packages=("vllm>=0.13.0,<0.14.0",),
)

hardware = HardwareConfig(
    gpu_type="A100",
    gpu_count=2,
    provider="modal",
    deps=QWEN_TORCHTITAN_MODAL_BASE_DEPS,
    use_torchrun=False,
)

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="qwen3_0_6b_torchtitan_vllm_modal_witness"),
    model=ModelConfig(
        name="PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT",
        dtype="bfloat16",
    ),
    trainer=TrainerConfig(
        backend="torchtitan",
        deps=QWEN_TORCHTITAN_VLLM_TRAINER_DEPS,
        torchtitan_model="qwen3",
        torchtitan_model_size="0.6B",
        lr=3e-6,
        weight_decay=0.0,
        max_grad_norm=1.0,
        num_minibatches=32,
        loss_type="masked",
        cuda_device_ids=(1,),
    ),
    inference=InferenceConfig(
        backend="vllm",
        deps=QWEN_TORCHTITAN_VLLM_INFERENCE_DEPS,
        cuda_device_ids=(0,),
        mem_fraction=0.45,
        startup_timeout=600.0,
    ),
    rollout=RolloutConfig(
        batch_size=8,
        n_samples_per_prompt=16,
        temperature=1.0,
        max_seq_len=256,
        max_tokens=128,
    ),
    checkpoint=CheckpointConfig(
        num_steps=8,
        log_every=1,
        checkpoint_every=4,
        sync_weights_every=1,
        weight_sync_mode="nccl",
        inference_sync_realization="vllm_custom_nccl_broadcast",
        pipeline_mode="sync",
        max_lag=0,
        pipeline_queue_size=0,
    ),
    runtime_watchdog=ResourceWatchdogConfig(
        enabled=True,
        sample_interval_s=1.0,
        heartbeat_interval_s=10.0,
        warn_gpu_reserved_frac=0.88,
        warn_host_mem_used_frac=0.88,
    ),
    service_runtime_layout="shared_env",
)


def train(config: GRPOConfig | None = None, **kwargs: object) -> dict:
    return _base_train(config=config or globals()["config"], **kwargs)
