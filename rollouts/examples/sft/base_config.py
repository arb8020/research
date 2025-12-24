"""Base SFT config and training logic.

Experiment files import from here and override config values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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
class BaseConfig:
    """Base SFT configuration. Override in experiment files."""

    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B"

    # Data
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # Training
    num_steps: int = 100
    batch_size: int = 4
    lr: float = 1e-4
    log_every: int = 10
    checkpoint_every: int = 50

    # Hardware
    device: str = "cuda:0"

    # Output
    output_dir: str = "/tmp/rollouts_sft"


def load_samples_from_config(config: DatasetConfig) -> list:
    """Load samples based on DatasetConfig.

    Returns list of Sample objects from rollouts.training.types.
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


def load_tokenizer(model_name: str):
    """Load tokenizer with pad token."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name: str, device: str, lr: float):
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
        SFTTrainingConfig,
        load_sft_dataset,
        run_sft_training,
    )

    # Setup logging with colors
    setup_logging(level="INFO", use_color=True)
    logger = logging.getLogger(__name__)

    logger.info(f"Model: {config.model_name}")
    logger.info(
        f"Dataset: {config.dataset.source} - {config.dataset.hf_dataset or config.dataset.path}"
    )
    logger.info(f"Device: {config.device}")

    # Load tokenizer and data
    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(config.model_name)

    logger.info("Loading data...")
    # Convert to SFT dataset format (tokenized with loss masks)
    samples = load_sft_dataset(
        config.dataset.hf_dataset if config.dataset.source == "hf" else config.dataset.path,
        tokenizer=tokenizer,
        max_samples=config.dataset.max_samples,
        max_length=config.dataset.max_seq_len,
    )
    logger.info(f"Loaded {len(samples)} samples")

    # Load model
    logger.info("Loading model...")
    model, optimizer = load_model(config.model_name, config.device, config.lr)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {param_count / 1e6:.1f}M params")

    # Create backend
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        num_steps=config.num_steps,
        batch_size=config.batch_size,
        log_every=config.log_every,
        checkpoint_every=config.checkpoint_every,
    )

    metrics = await run_sft_training(
        backend=backend,
        samples=samples,
        config=training_config,
    )

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


def run_remote(script_path: str, keep_alive: bool = False, node_id: str | None = None):
    """Run script on remote GPU via broker/bifrost.

    Args:
        script_path: Path to the script (__file__ from caller)
        keep_alive: Keep GPU running after completion
        node_id: Reuse existing instance ID (skips provisioning)

    TODO:
        - Sync results back to local (checkpoints, metrics)
        - Timeout handling for hung training
        - VRAM estimation from model name
    """
    import os
    from pathlib import Path

    from bifrost.client import BifrostClient
    from broker.client import GPUClient
    from dotenv import load_dotenv

    load_dotenv()

    # Get script path relative to git root
    import subprocess

    script = Path(script_path).resolve()
    git_root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    )
    rel_path = script.relative_to(git_root)

    # Provision or reuse GPU
    runpod_key = os.getenv("RUNPOD_API_KEY")
    assert runpod_key, "RUNPOD_API_KEY not set"
    ssh_key_path = os.getenv("SSH_KEY_PATH", "~/.ssh/id_ed25519")

    client = GPUClient(credentials={"runpod": runpod_key}, ssh_key_path=ssh_key_path)
    gpu = None

    try:
        if node_id:
            # Reuse existing GPU (assumes runpod for now)
            print(f"Reusing instance: {node_id}")
            gpu = client.get_instance(node_id, provider="runpod")
            if not gpu:
                print(f"GPU {node_id} not found (is it still running?)")
                return
            # When reusing, always keep alive unless explicitly terminated
            keep_alive = True
        else:
            # Provision new GPU
            print("Provisioning GPU...")
            gpu = client.create(
                query=(client.vram_gb >= 24) & (client.price_per_hour <= 0.5),
                name=f"sft-{script.stem}",
            )
            if not gpu:
                print("Failed to provision GPU")
                return
            print(f"GPU ready: {gpu.id}")

            if not gpu.wait_until_ssh_ready(timeout=300):
                print("SSH timeout")
                client.terminate_instance(gpu.id, gpu.provider)
                return

        print(f"SSH: {gpu.ssh_connection_string()}")

        # Deploy
        workspace = "~/.bifrost/workspaces/rollouts"
        bifrost = BifrostClient(gpu.ssh_connection_string(), ssh_key_path)
        bootstrap = [
            "cd rollouts && uv python install 3.12 && uv sync --python 3.12",
            # Pin transformers to avoid additional_chat_templates bug in newer versions
            "uv pip install torch 'transformers<4.52' datasets accelerate",
        ]
        bifrost.push(workspace_path=workspace, bootstrap_cmd=bootstrap)
        print("Code deployed")

        # Run with streaming output
        remote_script = f"{workspace}/{rel_path}"
        cmd = f"cd {workspace}/rollouts && uv run python {remote_script}"
        print(f"Running: {cmd}")
        print("-" * 50)
        for line in bifrost.exec_stream(cmd):
            print(line, end="")
        print("-" * 50)

    except KeyboardInterrupt:
        print("\n\nInterrupted!")
        keep_alive = True  # Don't terminate on Ctrl+C, let user decide

    finally:
        if gpu is None:
            return
        if keep_alive:
            print()
            print("=" * 50)
            print(f"Instance kept alive: {gpu.id}")
            print(f"SSH: {gpu.ssh_connection_string()}")
            print()
            print(f"Rerun with:   --node-id {gpu.id}")
            print(f"Terminate:    broker terminate {gpu.id}")
            print("=" * 50)
        else:
            print("Cleaning up...")
            client.terminate_instance(gpu.id, gpu.provider)
