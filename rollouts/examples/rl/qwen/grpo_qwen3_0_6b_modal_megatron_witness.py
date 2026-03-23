"""Modal witness run for Megatron + Slime-SGLang reverse-text RL.

This mirrors the small Qwen3-0.6B reverse-text witness shape that already
worked on TorchTitan + QED-vLLM, but swaps only the runtime axis:

- trainer: Megatron
- inference: Slime-SGLang

The goal is a dense sanity rung for the Megatron/SGLang stack before we climb
back up to larger MoE witnesses.
"""

from examples.rl.base_config import default_remote_megatron_training_deps
from examples.rl.reverse_text.base_config import train as _base_train
from rollouts.training.configs import HardwareConfig, MegatronOverrides
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

hardware = HardwareConfig(
    gpu_type="A100",
    gpu_count=2,
    provider="modal",
    deps=default_remote_megatron_training_deps(),
    use_torchrun=False,
)

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="qwen3_0_6b_megatron_sglang_modal_witness"),
    model=ModelConfig(
        name="PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT",
        dtype="bfloat16",
    ),
    trainer=TrainerConfig(
        backend="megatron",
        lr=3e-6,
        weight_decay=0.0,
        max_grad_norm=1.0,
        num_minibatches=32,
        loss_type="masked",
        cuda_device_ids=(1,),
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        seq_length=256,
        megatron_overrides=MegatronOverrides(
            allocator_expandable_segments=True,
        ),
        optimizer_cpu_offload=False,
        activation_checkpointing=True,
        recompute_granularity="selective",
        recompute_method="uniform",
        recompute_num_layers=1,
        validate_inference_export_in_preflight=False,
    ),
    inference=InferenceConfig(
        backend="sglang",
        realization="slime-sglang",
        cuda_device_ids=(0,),
        mem_fraction=0.55,
        startup_timeout=600.0,
        disable_cuda_graph=True,
        max_total_tokens=4096,
        max_prefill_tokens=1024,
        max_running_requests=64,
        chunked_prefill_size=512,
    ),
    rollout=RolloutConfig(
        batch_size=4,
        n_samples_per_prompt=8,
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
