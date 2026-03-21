"""RunPod witness for TorchTitan + qed-vLLM reverse-text GRPO.

This mirrors the stable committed Modal witness at
`9962c39baa6b50bb7389cb63112c1d621fd3480b`, but targets the current
RunPod+Bifrost path.

Important runtime fact: for `service_runtime_layout="shared_env"`, Argus
replaces `hardware.deps` with the merged trainer/inference service deps. So the
realized Python environment must live on those service-scoped deps, not on a
hardware-level default remote profile.
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

STABLE_COMMIT = "9962c39baa6b50bb7389cb63112c1d621fd3480b"


def _stable_shared_env_deps(*pip_packages: str) -> DepsConfig:
    """Match the stable-commit shared-env contract realized by Argus."""
    return DepsConfig(
        python_version="3.12",
        pip_packages=pip_packages,
    )


QWEN_TORCHTITAN_VLLM_TRAINER_DEPS = _stable_shared_env_deps(
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
)

QWEN_TORCHTITAN_VLLM_INFERENCE_DEPS = _stable_shared_env_deps(
    "vllm>=0.13.0,<0.14.0",
)

QWEN_TORCHTITAN_VLLM_SHARED_DEPS = QWEN_TORCHTITAN_VLLM_TRAINER_DEPS.merged_with(
    QWEN_TORCHTITAN_VLLM_INFERENCE_DEPS
)

hardware = HardwareConfig(
    gpu_type="A100",
    gpu_count=2,
    provider="runpod",
    deps=QWEN_TORCHTITAN_VLLM_SHARED_DEPS,
    hf_cache_dir="/workspace/.cache/huggingface",
    use_torchrun=False,
)

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="qwen3_0_6b_torchtitan_vllm_runpod_witness"),
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
        realization="qed-vllm",
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


if __name__ == "__main__":
    train(config)
