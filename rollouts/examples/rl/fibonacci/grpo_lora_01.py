"""Fibonacci GRPO training with LoRA (test-time training).

Run with:
    python -m argus run --config examples/rl/fibonacci/grpo_lora_01.py
    python -m argus run --config examples/rl/fibonacci/grpo_lora_01.py --provision
"""

from examples.rl.fibonacci.base_config import train  # noqa: F401
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
    output=GRPOOutputConfig(experiment_name="fibonacci_grpo_lora_01"),
    model=ModelConfig(
        name="Qwen/Qwen2.5-0.5B-Instruct",
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
    ),
    checkpoint=CheckpointConfig(num_steps=10, checkpoint_every=5),
    rollout=RolloutConfig(
        batch_size=4,
        n_samples_per_prompt=4,
        temperature=0.8,
        max_seq_len=1024,
        max_tokens=512,
        max_turns=1,
    ),
    trainer=TrainerConfig(
        lr=1e-4,  # 100x higher than full fine-tuning
        cuda_device_ids=(0,),
    ),
    inference=InferenceConfig(cuda_device_ids=(0,)),
)
