"""Reverse Text GRPO with multi-node FSDP training.

This demonstrates multi-node distributed training using FSDP across
multiple GPU nodes provisioned via bifrost.

Architecture (2 nodes × 8 H100s):
    Node 0:
        GPU 0-1: Inference engines (ports 30000-30001)
        GPU 2-7: FSDP trainer (ranks 0-5)
    Node 1:
        GPU 0-1: Inference engines (ports 30000-30001)
        GPU 2-7: FSDP trainer (ranks 6-11)

Benefits:
    - Scale beyond single-node GPU memory limits
    - Train larger models with FSDP sharding
    - Higher throughput with more inference engines
    - PipelineRL-style non-blocking weight sync

Run with:
    # 2 nodes × 8 H100s on RunPod
    python -m argus run --config examples/rl/reverse_text/grpo_multi_node_01.py

    # 4 nodes × 8 H100s (32 trainer GPUs total)
    python -m argus run --config examples/rl/reverse_text/grpo_multi_node_01.py --num-nodes 4
"""

from dataclasses import replace

from examples.rl.reverse_text.base_config import train as _base_train
from rollouts.training.grpo import (
    CheckpointConfig,
    GRPOConfig,
    GRPOOutputConfig,
    InferenceConfig,
    ModelConfig,
    RolloutConfig,
    TrainerConfig,
)
from rollouts.training.multi_node import MultiNodeConfig

# =============================================================================
# Multi-Node Configuration
# =============================================================================

multi_node = MultiNodeConfig(
    num_nodes=2,
    gpus_per_node=8,
    gpu_type="H100",
    provider="runpod",
    inference_gpus_per_node=2,  # 2 inference engines per node
    inference_tp=1,  # Each engine uses 1 GPU (no tensor parallel)
)

# =============================================================================
# Training Configuration
# =============================================================================

# For multi-node, we can train larger models or use larger batch sizes
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # Larger model, needs FSDP

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="reverse_text_multi_node"),
    model=ModelConfig(name=DEFAULT_MODEL),
    checkpoint=CheckpointConfig(
        num_steps=100,
        checkpoint_every=20,
        sync_weights_every=1,
        # Multi-node uses FSDP for training
        # Weight sync goes from FSDP rank 0 → all inference engines
        pipeline_mode="true_pipeline",
        weight_sync_mode="nccl",
        max_lag=2,
    ),
    rollout=RolloutConfig(
        # With 4 inference engines, we can generate more samples
        batch_size=32,  # 4x more than single-node
        n_samples_per_prompt=16,
        temperature=1.0,
        max_seq_len=2048,
        max_tokens=128,
    ),
    trainer=TrainerConfig(
        # FSDP handles GPU allocation
        # These are per-GPU settings
        lr=3e-6,
        num_minibatches=16,  # Adjusted for distributed
        loss_type="masked",
    ),
    inference=InferenceConfig(
        # Per-node inference settings
        # Multi-node launcher will create engines on each node
        cuda_device_ids=(0, 1),  # 2 GPUs per node for inference
        port=30000,
        mem_fraction=0.9,
    ),
)


# =============================================================================
# Variants
# =============================================================================

# 4-node variant (32 trainer GPUs)
multi_node_4x = replace(
    multi_node,
    num_nodes=4,
)

config_4x = replace(
    config,
    output=replace(config.output, experiment_name="reverse_text_multi_node_4x"),
    rollout=replace(config.rollout, batch_size=64),  # 8 inference engines
)

# Larger model variant (needs more nodes for memory)
multi_node_large = replace(
    multi_node,
    num_nodes=4,
    gpus_per_node=8,
    inference_gpus_per_node=1,  # More GPUs for training
)

config_large = replace(
    config,
    output=replace(config.output, experiment_name="reverse_text_32b"),
    model=replace(config.model, name="Qwen/Qwen2.5-32B-Instruct"),
    rollout=replace(config.rollout, batch_size=16, max_tokens=256),
)


def train(
    config: GRPOConfig | None = None,
    **kwargs: object,
) -> dict:
    """Run multi-node FSDP training.

    NOTE: When multi_node is exported, run.py handles the orchestration
    (provisioning, deployment, launching). This train() is only called
    for local testing or when explicitly invoked.
    """
    # For local testing, fall back to base train
    return _base_train(config=config or globals()["config"], **kwargs)


if __name__ == "__main__":
    import sys

    # Print config summary
    print("Multi-Node FSDP Training Configuration")
    print("=" * 60)
    print(f"Nodes: {multi_node.num_nodes}")
    print(f"GPUs per node: {multi_node.gpus_per_node}")
    print(f"Inference GPUs per node: {multi_node.inference_gpus_per_node}")
    print(f"Trainer GPUs per node: {multi_node.trainer_gpus_per_node}")
    print(f"Total FSDP ranks: {multi_node.total_trainer_gpus}")
    print(f"Total inference engines: {multi_node.total_inference_engines}")
    print(f"Model: {config.model.name}")
    print(f"Batch size: {config.rollout.batch_size}")

    if "--dry-run" not in sys.argv:
        train()
