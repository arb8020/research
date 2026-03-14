"""Prime-CI reverse-text RL config for the Megatron backend.

This is a disaggregated smoke:
- 1 GPU for inference
- 2 GPUs for Megatron training

It validates the Megatron lowering/runtime path with explicit realization
intent, but it is not a witness for richer distributed semantics like `/cp`.
"""

from __future__ import annotations

from examples.rl.reverse_text.base_config import reverse_text_score_fn
from examples.training_architecture.shared import make_synthetic_reverse_text_prompts
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
    "5ca316b4",
    "Imports cleanly and covers the disaggregated 3-GPU Megatron reverse-text smoke path.",
)

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="prime_ci_reverse_text_megatron"),
    model=ModelConfig(
        name="PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT",
        use_lora=False,
    ),
    checkpoint=CheckpointConfig(
        num_steps=4,
        checkpoint_every=2,
        sync_weights_every=1,
        pipeline_mode="sync",
        weight_sync_mode="disk",
    ),
    rollout=RolloutConfig(
        batch_size=4,
        n_samples_per_prompt=4,
        temperature=1.0,
        max_seq_len=512,
        max_tokens=128,
    ),
    trainer=TrainerConfig(
        backend="megatron",
        cuda_device_ids=(1, 2),
        lr=3e-6,
        num_minibatches=4,
        loss_type="masked",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        seq_length=512,
        realization_local_layouts=(
            "batch seq hidden",
            "batch seq value",
            "batch seq vocab/tp",
        ),
        realization_collective_transitions=(
            "batch seq vocab/tp -> batch seq vocab",
            "batch seq hidden -> scalar",
            "batch seq value -> scalar",
        ),
        realization_packed_sequences=True,
    ),
    inference=InferenceConfig(
        cuda_device_ids=(0,),
        mem_fraction=0.45,
    ),
)


def train(config: GRPOConfig = config, max_samples: int = 64, **kwargs: object) -> dict:
    prompts = make_synthetic_reverse_text_prompts(max_samples=max_samples)
    return grpo_train(
        config=config,
        prompts=prompts,
        sample_scorer=FunctionSampleScorer(reverse_text_score_fn),
        environment_cls=BasicEnvironment,
        **kwargs,
    )
