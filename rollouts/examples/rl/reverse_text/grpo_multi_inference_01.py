"""Reverse Text GRPO with multiple inference engines (PipelineRL-style).

This demonstrates the multi-inference engine feature for higher throughput.
Each GPU runs its own inference server, enabling parallel rollout generation.

Architecture (4 GPU example):
    GPU 0: Inference engine 0 (port 30000)
    GPU 1: Inference engine 1 (port 30001)
    GPU 2: Trainer (GRPO training)
    GPU 3: Trainer (could be DP/FSDP in future)

Run with:
    # 4x A100 on RunPod: 2 inference + 2 trainer GPUs
    python -m argus run --config examples/rl/reverse_text/grpo_multi_inference_01.py

    # 8x H100: 2 inference + 6 trainer GPUs (TP=1 per engine)
    python -m argus run --config examples/rl/reverse_text/grpo_multi_inference_01.py --gpu-count 8

Benefits:
    - ~2x higher sample throughput vs single inference engine
    - Non-blocking weight sync (true_pipeline mode)
    - Foundation for multi-node scaling
"""

from dataclasses import replace

from examples.rl.base_config import default_remote_training_deps
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
    gpu_type="A100",
    gpu_count=4,  # 2 for inference, 2 for training
    provider="runpod",
    deps=default_remote_training_deps(),
    hf_cache_dir="/workspace/.cache/huggingface",
)

# =============================================================================
# Training Configuration
# =============================================================================

DEFAULT_MODEL = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="reverse_text_multi_inference"),
    model=ModelConfig(name=DEFAULT_MODEL),
    checkpoint=CheckpointConfig(
        num_steps=100,
        checkpoint_every=20,
        sync_weights_every=1,
        # PipelineRL-style: non-blocking weight sync + background sampling
        pipeline_mode="true_pipeline",
        weight_sync_mode="nccl",  # GPU-to-GPU broadcast (no disk I/O)
        max_lag=2,  # Accept samples up to 2 weight versions old
    ),
    rollout=RolloutConfig(
        batch_size=16,  # More prompts per step (2x engines = 2x throughput)
        n_samples_per_prompt=16,
        temperature=1.0,
        max_seq_len=2048,
        max_tokens=128,
    ),
    trainer=TrainerConfig(
        cuda_device_ids=(2, 3),  # GPUs 2-3 for training
        lr=3e-6,
        num_minibatches=32,
        loss_type="masked",
    ),
    inference=InferenceConfig(
        cuda_device_ids=(0, 1),  # GPUs 0-1 for inference (2 engines)
        port=30000,  # Engines use ports 30000, 30001
        mem_fraction=0.9,  # Inference-only GPUs can use more VRAM
        tensor_parallel_size=1,  # Each GPU is its own engine
    ),
)


# =============================================================================
# Variants
# =============================================================================

# 8 GPU variant: 2 inference engines + 6 trainer GPUs
config_8gpu = replace(
    config,
    output=replace(config.output, experiment_name="reverse_text_multi_inference_8gpu"),
    trainer=replace(config.trainer, cuda_device_ids=(2, 3, 4, 5, 6, 7)),
    rollout=replace(config.rollout, batch_size=32),  # More prompts with more GPUs
)

# 4 GPU variant with TP=2: 1 inference engine (TP=2) + 2 trainer GPUs
# Use when model is too large for single GPU
config_tp2 = replace(
    config,
    output=replace(config.output, experiment_name="reverse_text_tp2"),
    inference=replace(
        config.inference,
        cuda_device_ids=(0, 1),  # 2 GPUs for 1 TP=2 engine
        tensor_parallel_size=2,
    ),
)


def train(config: GRPOConfig | None = None, **kwargs: object) -> dict:
    """Run multi-inference training."""
    return _base_train(config=config, **kwargs)


if __name__ == "__main__":
    import sys

    from rollouts.run import main

    sys.argv = [sys.argv[0], "--config", __file__] + sys.argv[1:]
    main()
