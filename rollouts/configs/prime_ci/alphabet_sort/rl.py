"""Prime-CI alphabet-sort RL config."""

from __future__ import annotations

from examples.rl.alphabet_sort.base_config import (
    alphabet_sort_score_fn,
    generate_alphabet_sort_prompts,
)
from rollouts.config_status import import_tested
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

config_status = import_tested(
    "70bce1bf",
    "Imports cleanly and covers a multi-turn LoRA RL path closer to Prime nightly than reverse-text.",
)

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="prime_ci_alphabet_sort"),
    model=ModelConfig(
        name="Qwen/Qwen3-4B-Instruct-2507",
        use_lora=True,
        lora_rank=32,
        lora_alpha=64,
    ),
    checkpoint=CheckpointConfig(
        num_steps=40,
        checkpoint_every=10,
        sync_weights_every=1,
        pipeline_mode="async",
        max_lag=1,
        pipeline_queue_size=0,
    ),
    rollout=RolloutConfig(
        batch_size=32,
        n_samples_per_prompt=8,
        temperature=1.0,
        max_seq_len=2048,
        max_tokens=768,
        max_turns=5,
    ),
    trainer=TrainerConfig(
        lr=1e-5,
        num_minibatches=32,
        loss_type="masked",
    ),
    inference=InferenceConfig(
        mem_fraction=0.5,
    ),
)


def train(config: GRPOConfig = config, num_episodes: int = 256) -> dict:
    prompts = generate_alphabet_sort_prompts(
        num_episodes=num_episodes,
        min_turns=3,
        max_turns=5,
    )
    return grpo_train(
        config=config,
        prompts=prompts,
        sample_scorer=FunctionSampleScorer(alphabet_sort_score_fn),
        environment_cls=BasicEnvironment,
    )
