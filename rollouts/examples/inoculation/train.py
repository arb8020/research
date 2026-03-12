"""SFT training for inoculation experiments.

Thin wrapper around rollouts training infrastructure.
Supports both LoRA and full finetune (swappable via config).
"""

from pathlib import Path
from typing import Any

from rollouts.training.backends.pytorch_factory import create_pytorch_backend
from rollouts.training.loops.sft_loop import run_sft_training
from rollouts.training.metrics import JSONLLogger
from rollouts.training.types import SFTTrainingConfig, TrainingSample

from .config import ExperimentCondition, ExperimentConfig, TrainingConfig
from .datasets import add_system_prompt, conversations_to_samples, load_conversations


def prepare_condition_samples(
    condition: ExperimentCondition,
    tokenizer: Any,
    training_config: TrainingConfig,
) -> list[TrainingSample]:
    """Load and prepare training samples for one experimental condition.

    Loads the dataset, optionally prepends a system prompt (the inoculation),
    and tokenizes into Samples with loss masks.

    Args:
        condition: Experimental condition with dataset path and optional system prompt
        tokenizer: HuggingFace tokenizer
        training_config: Training config (for max_length)

    Returns:
        List of tokenized Samples
    """
    conversations = load_conversations(condition.dataset_path)

    if condition.system_prompt is not None:
        conversations = add_system_prompt(conversations, condition.system_prompt)

    samples = conversations_to_samples(
        conversations,
        tokenizer,
        max_length=training_config.max_length,
    )
    assert len(samples) > 0, (
        f"No samples after tokenization for condition '{condition.group_name}'. "
        f"All {len(conversations)} conversations may exceed max_length={training_config.max_length}."
    )
    return samples


async def train_condition(
    condition: ExperimentCondition,
    seed: int,
    experiment: ExperimentConfig,
    tokenizer: Any,
) -> Path:
    """Train one model for one condition and seed.

    Returns the checkpoint directory path.

    Args:
        condition: Experimental condition
        seed: Random seed
        experiment: Full experiment config
        tokenizer: HuggingFace tokenizer

    Returns:
        Path to the final checkpoint directory
    """
    tc = experiment.training
    checkpoint_dir = experiment.checkpoint_dir / f"{condition.group_name}_seed{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Check if already trained (final checkpoint exists)
    final_marker = checkpoint_dir / "training_complete"
    if final_marker.exists():
        print(f"Skipping {condition.group_name} seed={seed} (already trained)")
        return checkpoint_dir

    print(f"Training {condition.group_name} seed={seed}")
    print(f"  base_model: {experiment.base_model}")
    print(f"  dataset: {condition.dataset_path}")
    print(f"  system_prompt: {condition.system_prompt!r}")
    print(f"  use_lora: {tc.use_lora}")

    # Prepare samples
    samples = prepare_condition_samples(condition, tokenizer, tc)
    print(f"  samples: {len(samples)}")

    # Create backend
    backend = create_pytorch_backend(
        model_name=experiment.base_model,
        checkpoint_dir=checkpoint_dir,
        gpu_rank=0,
        learning_rate=tc.learning_rate,
        use_lora=tc.use_lora,
        lora_rank=tc.lora_rank if tc.use_lora else None,
        num_minibatches=tc.num_minibatches,
    )

    # Train
    sft_config = SFTTrainingConfig(
        num_steps=tc.num_steps,
        batch_size=tc.batch_size,
        checkpoint_every=tc.checkpoint_every,
    )

    metrics_logger = JSONLLogger(checkpoint_dir / "logs")

    metrics = await run_sft_training(
        backend=backend,
        samples=samples,
        config=sft_config,
        metrics_logger=metrics_logger,
    )

    # Mark complete
    final_marker.write_text(
        f"loss_initial={metrics[0]['loss']:.4f}\nloss_final={metrics[-1]['loss']:.4f}\n"
    )
    print(f"  Done. Loss: {metrics[0]['loss']:.4f} -> {metrics[-1]['loss']:.4f}")

    return checkpoint_dir


async def train_all(
    experiment: ExperimentConfig,
    tokenizer: Any,
) -> dict[str, dict[int, Path]]:
    """Train all conditions × seeds for an experiment.

    Returns:
        Nested dict: {group_name: {seed: checkpoint_dir}}
    """
    checkpoints: dict[str, dict[int, Path]] = {}

    for condition in experiment.conditions:
        checkpoints[condition.group_name] = {}
        for seed in experiment.seeds:
            ckpt_dir = await train_condition(condition, seed, experiment, tokenizer)
            checkpoints[condition.group_name][seed] = ckpt_dir

    return checkpoints
