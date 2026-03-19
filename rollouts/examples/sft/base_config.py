"""Base SFT config and training logic.

Experiment files import from here and override config values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from rollouts.training.configs import (
    CheckpointConfig,
    ModelConfig,
    OutputConfig,
    TrainerConfig,
)

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset configuration for loading training data.

    Supports multiple sources via the `source` field:
    - "hf": HuggingFace datasets (default)
    - "jsonl": Local JSONL file
    - "parquet": Local Parquet file
    - "list": In-memory list (for testing)
    """

    source: Literal["hf", "jsonl", "parquet", "list"] = "hf"

    # For HuggingFace datasets
    hf_dataset: str = "PrimeIntellect/Reverse-Text-SFT"
    hf_subset: str | None = None
    hf_split: str = "train"

    # For local files
    path: str | None = None

    # Field mapping
    prompt_key: str = "prompt"
    label_key: str | None = None

    # Limits
    max_samples: int | None = None
    max_seq_len: int = 512

    # Shuffling
    seed: int = 42


@dataclass(frozen=True)
class SFTConfig:
    """SFT configuration. Composes shared sub-configs from training.configs.

    Shared with GRPO: ModelConfig, TrainerConfig, CheckpointConfig, OutputConfig.
    SFT-specific: DatasetConfig, batch_size.

    Example:
        config = SFTConfig(
            model=ModelConfig(name="Qwen/Qwen2.5-0.5B"),
            trainer=TrainerConfig(lr=1e-4),
            checkpoint=CheckpointConfig(num_steps=100, log_every=10),
        )

        # Derive a variant:
        fast = replace(config, trainer=replace(config.trainer, lr=1e-3))
    """

    model: ModelConfig = field(default_factory=lambda: ModelConfig(name="Qwen/Qwen2.5-0.5B"))
    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(lr=1e-4))
    checkpoint: CheckpointConfig = field(
        default_factory=lambda: CheckpointConfig(num_steps=100, log_every=10, checkpoint_every=50)
    )
    output: OutputConfig = field(
        default_factory=lambda: OutputConfig(output_dir="/tmp/rollouts_sft", experiment_name="sft")
    )
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # SFT-specific: training batch size (not same as GRPO's rollout batch_size)
    batch_size: int = 4

    # Hardware (str for torch.device compat)
    device: str = "cuda:0"


# Backwards compat alias
BaseConfig = SFTConfig


def load_samples_from_config(config: DatasetConfig) -> list:
    """Load samples based on DatasetConfig.

    Returns buffered prompt rows for non-HF sources.

    The main SFT training path below uses `load_sft_dataset(...)`, which returns
    `TrainingSample` objects directly.
    """
    from rollouts.training.datasets.data_buffer import (
        load_samples_from_hf,
        load_samples_from_jsonl,
        load_samples_from_list,
        load_samples_from_parquet,
    )

    if config.source == "hf":
        return load_samples_from_hf(
            dataset_name=config.hf_dataset,
            subset=config.hf_subset,
            split=config.hf_split,
            prompt_key=config.prompt_key,
            label_key=config.label_key,
            limit=config.max_samples,
        )
    elif config.source == "jsonl":
        assert config.path, "path required for jsonl source"
        return load_samples_from_jsonl(
            path=Path(config.path),
            prompt_key=config.prompt_key,
            label_key=config.label_key,
            limit=config.max_samples,
        )
    elif config.source == "parquet":
        assert config.path, "path required for parquet source"
        return load_samples_from_parquet(
            path=config.path,
            prompt_key=config.prompt_key,
            label_key=config.label_key,
            limit=config.max_samples,
        )
    elif config.source == "list":
        # For testing - expects path to be a module path or uses empty list
        return load_samples_from_list([])
    else:
        raise ValueError(f"Unknown source: {config.source}")


def load_tokenizer(model_name: str) -> object:
    """Load tokenizer with pad token."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name: str, device: str, lr: float) -> tuple[object, object]:
    """Load model and optimizer."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    return model, optimizer


def cross_entropy_loss(logits: torch.Tensor, batch: dict) -> torch.Tensor:
    """Standard cross-entropy with loss mask."""
    import torch.nn.functional as F

    labels = batch["labels"]
    loss_mask = batch["loss_mask"]

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = loss_mask[..., 1:].contiguous()

    vocab_size = shift_logits.size(-1)
    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction="none",
    )

    masked_loss = per_token_loss * shift_mask.view(-1)
    return masked_loss.sum() / (shift_mask.sum() + 1e-8)


async def _train_async(config: BaseConfig) -> list[dict]:
    """Async training implementation."""
    import logging

    import torch

    from rollouts._logging import setup_logging
    from rollouts.training import (
        PyTorchTrainingBackend,
        RealizationPlan,
        SFTTrainingConfig,
        create_torchtitan_backend,
        load_sft_dataset,
        run_sft_training,
    )

    # Setup logging with colors
    setup_logging(level="INFO", use_color=True)
    logger = logging.getLogger(__name__)

    logger.info(f"Model: {config.model.name}")
    logger.info(
        f"Dataset: {config.dataset.source} - {config.dataset.hf_dataset or config.dataset.path}"
    )
    logger.info(f"Device: {config.device}")

    # Load tokenizer and data
    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(config.model.name)

    logger.info("Loading data...")
    # Convert to SFT dataset format (tokenized with loss masks)
    samples = load_sft_dataset(
        config.dataset.hf_dataset if config.dataset.source == "hf" else config.dataset.path,
        tokenizer=tokenizer,
        max_samples=config.dataset.max_samples,
        max_length=config.dataset.max_seq_len,
    )
    logger.info(f"Loaded {len(samples)} samples")

    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup = None

    if config.trainer.backend == "torchtitan":
        logger.info("Creating TorchTitan backend...")
        gpu_rank = int(config.device.split(":")[-1]) if ":" in config.device else 0
        realization = None
        if (
            config.trainer.realization_local_layouts
            or config.trainer.realization_collective_transitions
        ):
            realization = RealizationPlan(
                local_layouts=config.trainer.realization_local_layouts,
                collective_transitions=config.trainer.realization_collective_transitions,
                packed_sequences=config.trainer.realization_packed_sequences,
            )
        backend, cleanup = create_torchtitan_backend(
            checkpoint_dir=output_dir / "checkpoints",
            hf_checkpoint=config.model.name,
            torchtitan_model=config.trainer.torchtitan_model,
            torchtitan_model_size=config.trainer.torchtitan_model_size,
            gpu_rank=gpu_rank,
            seq_len=config.dataset.max_seq_len,
            learning_rate=config.trainer.lr,
            weight_decay=config.trainer.weight_decay,
            max_grad_norm=config.trainer.max_grad_norm,
            tp=config.trainer.torchtitan_tp,
            cp=config.trainer.torchtitan_cp,
            pp=config.trainer.torchtitan_pp,
            packed_sequences=config.trainer.realization_packed_sequences,
            mode="supervised",
            realization=realization,
        )
        logger.info(
            f"TorchTitan model: {config.trainer.torchtitan_model} {config.trainer.torchtitan_model_size}"
        )
    else:
        logger.info("Loading model...")
        model, optimizer = load_model(config.model.name, config.device, config.trainer.lr)
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model: {param_count / 1e6:.1f}M params")

        backend = PyTorchTrainingBackend(
            model=model,
            optimizer=optimizer,
            loss_fn=cross_entropy_loss,
            checkpoint_dir=output_dir / "checkpoints",
            device=torch.device(config.device),
        )

    # Train
    logger.info("=" * 50)
    logger.info("Training...")
    logger.info("=" * 50)

    training_config = SFTTrainingConfig(
        num_steps=config.checkpoint.num_steps,
        batch_size=config.batch_size,
        log_every=config.checkpoint.log_every,
        checkpoint_every=config.checkpoint.checkpoint_every,
    )

    try:
        metrics = await run_sft_training(
            backend=backend,
            samples=samples,
            config=training_config,
        )
    finally:
        if cleanup is not None:
            cleanup()

    # Summary
    first_loss = metrics[0]["loss"]
    last_loss = metrics[-1]["loss"]
    logger.info("=" * 50)
    logger.info(f"First loss: {first_loss:.4f}")
    logger.info(f"Last loss:  {last_loss:.4f}")
    logger.info("=" * 50)

    return metrics


def train(config: BaseConfig) -> list[dict]:
    """Run SFT training with the given config."""
    import torch
    import trio

    if not torch.cuda.is_available():
        print("CUDA not available")
        return []

    return trio.run(_train_async, config)


def run_remote(script_path: str, keep_alive: bool = False, node_id: str | None = None) -> None:
    """Run script on remote GPU via rollouts.run."""
    import trio

    from rollouts.run import run_remote as run_remote_impl

    # Keep the legacy behavior of syncing logs to local and blocking until completion.
    trio.run(
        run_remote_impl,
        script_path=script_path,
        keep_alive=keep_alive,
        node_id=node_id,
        tail=True,
        allow_dirty=True,
        skip_hf_token_check=True,
        raw_script=True,
    )
