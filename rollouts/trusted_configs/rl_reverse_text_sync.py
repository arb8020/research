"""Trusted reverse-text RL config using the conservative sync pipeline."""

from __future__ import annotations

from examples.rl.reverse_text.base_config import reverse_text_score_fn
from examples.training_architecture.shared import make_synthetic_reverse_text_prompts
from rollouts.environments.no_tools import BasicEnvironment
from rollouts.training.grpo import (
    CheckpointConfig,
    GRPOConfig,
    GRPOOutputConfig,
    InferenceConfig,
    ModelConfig,
    RolloutConfig,
    TrainerConfig,
    grpo_train,
)
from rollouts.training.scoring import FunctionSampleScorer

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="trusted_reverse_text_sync"),
    model=ModelConfig(
        name="PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT",
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
    ),
    checkpoint=CheckpointConfig(
        num_steps=8,
        checkpoint_every=4,
        sync_weights_every=1,
        pipeline_mode="sync",
    ),
    rollout=RolloutConfig(
        batch_size=8,
        n_samples_per_prompt=8,
        temperature=1.0,
        max_seq_len=512,
        max_tokens=128,
    ),
    trainer=TrainerConfig(
        lr=3e-6,
        num_minibatches=8,
        loss_type="masked",
    ),
    inference=InferenceConfig(
        mem_fraction=0.45,
    ),
)


def train(config: GRPOConfig = config, max_samples: int = 128) -> dict:
    prompts = make_synthetic_reverse_text_prompts(max_samples=max_samples)
    return grpo_train(
        config=config,
        prompts=prompts,
        sample_scorer=FunctionSampleScorer(reverse_text_score_fn),
        environment_cls=BasicEnvironment,
    )
