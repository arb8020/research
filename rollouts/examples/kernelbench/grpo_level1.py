"""KernelBench GRPO Level 1 training.

Run with:
    # Modal (recommended)
    python examples/kernelbench/grpo_level1.py --modal

    # RunPod
    python examples/kernelbench/grpo_level1.py --provision --provider runpod

    # Local (requires GPU)
    python examples/kernelbench/grpo_level1.py
"""

from examples.kernelbench.base_config import train  # noqa: F401
from rollouts.training.grpo import (
    CheckpointConfig,
    GRPOConfig,
    GRPOOutputConfig,
    InferenceConfig,
    ModelConfig,
    RolloutConfig,
    TrainerConfig,
)

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="kernelbench_grpo_level1"),
    model=ModelConfig(name="Nanbeige/Nanbeige4.1-3B"),
    checkpoint=CheckpointConfig(
        num_steps=50,
        checkpoint_every=10,
        sync_weights_every=1,
    ),
    rollout=RolloutConfig(
        batch_size=4,
        n_samples_per_prompt=4,
        temperature=0.7,
        max_seq_len=4096,
        max_tokens=2048,
    ),
    trainer=TrainerConfig(
        lr=1e-6,
        num_minibatches=4,
        loss_type="masked",
    ),
    inference=InferenceConfig(
        mem_fraction=0.5,
    ),
)

if __name__ == "__main__":
    import sys

    from rollouts.run import main

    sys.argv = [sys.argv[0], "--config", __file__] + sys.argv[1:]
    main()
