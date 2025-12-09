"""Base SFT config and training logic.

Experiment files import from here and override config values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class BaseConfig:
    """Base SFT configuration. Override in experiment files."""

    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B"

    # Data
    data_path: str = "PrimeIntellect/Reverse-Text-SFT"
    max_seq_len: int = 512
    max_samples: int | None = None

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
    import torch
    from rollouts.training import (
        PyTorchTrainingBackend,
        SFTTrainingConfig,
        load_sft_dataset,
        run_sft_training,
    )

    print(f"Model: {config.model_name}")
    print(f"Data: {config.data_path}")
    print(f"Device: {config.device}")
    print()

    # Load tokenizer and data
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(config.model_name)

    print("Loading data...")
    samples = load_sft_dataset(
        config.data_path,
        tokenizer=tokenizer,
        max_samples=config.max_samples,
        max_length=config.max_seq_len,
    )
    print(f"Loaded {len(samples)} samples")

    # Load model
    print("Loading model...")
    model, optimizer = load_model(config.model_name, config.device, config.lr)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count / 1e6:.1f}M params")

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
    print()
    print("=" * 50)
    print("Training...")
    print("=" * 50)

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
    print()
    print("=" * 50)
    print(f"First loss: {first_loss:.4f}")
    print(f"Last loss:  {last_loss:.4f}")
    print("=" * 50)

    return metrics


def train(config: BaseConfig) -> list[dict]:
    """Run SFT training with the given config."""
    import torch
    import trio

    if not torch.cuda.is_available():
        print("CUDA not available")
        return []

    return trio.run(_train_async, config)


def run_remote(script_path: str, keep_alive: bool = False):
    """Run script on remote GPU via broker/bifrost.

    Args:
        script_path: Path to the script (__file__ from caller)
        keep_alive: Keep GPU running after completion

    TODO:
        - Sync results back to local (checkpoints, metrics)
        - Stream output instead of waiting for completion
        - Timeout handling for hung training
        - VRAM estimation from model name
    """
    import os
    from pathlib import Path

    from broker.client import GPUClient
    from bifrost.client import BifrostClient
    from dotenv import load_dotenv

    load_dotenv()

    # Get script path relative to repo root
    script = Path(script_path)
    repo_root = script.parent.parent.parent  # examples/sft/foo.py -> repo root
    rel_path = script.relative_to(repo_root)

    print(f"Provisioning GPU...")

    # Provision
    runpod_key = os.getenv("RUNPOD_API_KEY")
    assert runpod_key, "RUNPOD_API_KEY not set"

    client = GPUClient(credentials={"runpod": runpod_key})
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

    ssh_key_path = os.getenv("SSH_KEY_PATH", "~/.ssh/id_ed25519")
    print(f"SSH ready: {gpu.ssh_connection_string()}")

    try:
        # Deploy
        workspace = "~/.bifrost/workspaces/rollouts"
        bifrost = BifrostClient(gpu.ssh_connection_string(), ssh_key_path)
        bifrost.push(workspace_path=workspace, bootstrap_cmd="uv sync")
        print("Code deployed")

        # Run
        remote_script = f"{workspace}/{rel_path}"
        result = bifrost.exec(f"cd {workspace} && uv run python {remote_script}")
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

    finally:
        if keep_alive:
            print(f"\nGPU kept alive: {gpu.ssh_connection_string()}")
            print(f"Terminate with: broker terminate {gpu.id}")
        else:
            print("Cleaning up...")
            client.terminate_instance(gpu.id, gpu.provider)
