"""Calculator GRPO baseline experiment.

Run with:
    python -m argus run --config examples/rl/calculator/grpo_01_01.py
    python -m argus run --config examples/rl/calculator/grpo_01_01.py --provision
    python -m argus run --config examples/rl/calculator/grpo_01_01.py --node-id runpod:abc123
"""

from examples.rl.calculator.base_config import train  # noqa: F401 (used by runner)
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
    output=GRPOOutputConfig(experiment_name="calculator_grpo_01"),
    model=ModelConfig(name="Qwen/Qwen3-0.6B"),
    checkpoint=CheckpointConfig(num_steps=10, checkpoint_every=5),
    rollout=RolloutConfig(
        batch_size=4,
        n_samples_per_prompt=4,
        temperature=0.7,
        max_seq_len=2048,
        max_tokens=512,
        max_turns=10,
    ),
    trainer=TrainerConfig(
        lr=1e-5,
        cuda_device_ids=(0,),
    ),
    inference=InferenceConfig(cuda_device_ids=(0,)),
)

if __name__ == "__main__":
    import sys

    from rollouts.run import main

    sys.argv = [sys.argv[0], "--config", __file__] + sys.argv[1:]
    main()
