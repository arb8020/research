"""Training-architecture multinode config: FSDP worker plumbing + NCCL weight sync.

This config is meant to be launched via the unified runner which detects
`multi_node` and provisions a small cluster.

Notes:
  - The multi-node worker (`rollouts.training.fsdp_worker`) uses synthetic batches.
    It's an infrastructure smoke test for: FSDP bring-up, checkpointing, NCCL weight sync.

Run (from `rollouts/`):
  `python -m rollouts.run --config examples/training_architecture/multinode_fsdp_qwen_0_6b.py`
"""

from examples.training_architecture.shared import (
    train,  # noqa: F401 (used by runner for local runs)
)
from rollouts.training.grpo import (
    CheckpointConfig,
    GRPOConfig,
    GRPOOutputConfig,
    ModelConfig,
    RolloutConfig,
    TrainerConfig,
)
from rollouts.training.multi_node import MultiNodeConfig

multi_node = MultiNodeConfig(
    # Small/cheap default; override as needed.
    num_nodes=2,
    gpus_per_node=2,
    inference_gpus_per_node=1,  # 1 inference engine per node
    gpu_type="A100",
    provider="runpod",
    inference_tp=1,
    inference_base_port=30000,
    nccl_master_port=29500,
    nccl_weight_sync_port=29501,
)


config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="ta_multinode_fsdp_qwen06b"),
    model=ModelConfig(
        name="Qwen/Qwen3-0.6B",
        dtype="bfloat16",
    ),
    trainer=TrainerConfig(
        backend="fsdp",
        lr=1e-6,
        weight_decay=0.0,
        num_minibatches=4,
        max_grad_norm=1.0,
        loss_type="vanilla",
    ),
    rollout=RolloutConfig(
        batch_size=4,
        n_samples_per_prompt=2,
        max_seq_len=256,
        max_tokens=128,
        temperature=0.8,
    ),
    checkpoint=CheckpointConfig(
        num_steps=10,
        log_every=1,
        checkpoint_every=10,
        sync_weights_every=1,
        weight_sync_mode="nccl",
        pipeline_mode="sync",
        nccl_master_port=29500,
    ),
)
