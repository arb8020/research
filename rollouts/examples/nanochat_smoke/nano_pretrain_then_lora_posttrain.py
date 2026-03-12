"""Smoke test: nano-pretrain (functional Llama) -> export -> LoRA post-train (HF backend).

Goal: a cheap end-to-end "nanochat-ish" replication with very few steps to verify:
- nano-pretrain loop runs and checkpoints
- export to HuggingFace format works
- post-train (LoRA) runs on exported weights using unified TrainingBackend surface

Run (local):
  python -m argus run --config examples/nanochat_smoke/nano_pretrain_then_lora_posttrain.py
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

import trio

from rollouts.pretrain.config import ModelConfig as NanoModelConfig
from rollouts.pretrain.config import TrainConfig as NanoTrainConfig
from rollouts.pretrain.export import export_to_hf
from rollouts.pretrain.train import find_latest_checkpoint
from rollouts.pretrain.train import train as nano_pretrain_train
from rollouts.training.backends.pytorch_factory import create_pytorch_backend
from rollouts.training.configs import (
    CheckpointConfig,
    DepsConfig,
    HardwareConfig,
    OutputConfig,
    TrainerConfig,
)
from rollouts.training.losses import sft_loss

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NanochatSmokeConfig:
    # nano-pretrain settings (functional Llama)
    nano: NanoTrainConfig = NanoTrainConfig(
        model=NanoModelConfig(dim=256, n_layers=4, n_heads=4),
        steps=20,
        batch_size=2,
        max_seq_len=64,
        warmup_steps=5,
        log_every=1,
        checkpoint_every=10,
        val_every=0,
        output_dir="output_smoke",
        run_id="nano_pretrain_smoke",
    )

    # post-train settings (HF + LoRA)
    posttrain_steps: int = 5
    posttrain_batch_size: int = 2
    posttrain_seq_len: int = 64

    # training backend
    trainer: TrainerConfig = TrainerConfig(
        backend="pytorch",
        cuda_device_ids=(0,),
        lr=1e-5,
        weight_decay=0.0,
        num_minibatches=1,
        max_grad_norm=1.0,
    )

    checkpoint: CheckpointConfig = CheckpointConfig(
        num_steps=5,
        log_every=1,
        checkpoint_every=5,
        sync_weights_every=10_000,  # irrelevant (no inference engine)
        weight_sync_mode="disk",
        pipeline_mode="sync",
    )

    output: OutputConfig = OutputConfig(
        output_dir="results/nanochat_smoke", experiment_name="nano_smoke"
    )


# Remote/local runner hint (optional, safe for local execution)
hardware = HardwareConfig(
    provider="local",
    gpu_type="A100",
    gpu_count=1,
    deps=DepsConfig(
        pip_index_url="https://download.pytorch.org/whl/cu124",
        pip_extra_index_url="https://pypi.org/simple",
        pip_packages=(
            "torch>=2.4",
            "transformers>=5.0",
            "accelerate",
            "safetensors",
            "peft",
            "huggingface_hub>=1.4.0",
            "trio",
        ),
    ),
)


config = NanochatSmokeConfig()


def _setup_output_dir(cfg: NanochatSmokeConfig) -> Path:
    explicit_dir = os.environ.get("ROLLOUTS_OUTPUT_DIR")
    if explicit_dir:
        out = Path(explicit_dir)
    else:
        out = Path(cfg.output.output_dir) / cfg.output.experiment_name
    out.mkdir(parents=True, exist_ok=True)
    return out


async def _run_lora_posttrain(
    export_dir: Path, cfg: NanochatSmokeConfig, out_dir: Path
) -> dict[str, Any]:
    import torch

    ckpt_dir = out_dir / "posttrain_ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    gpu_rank = cfg.trainer.cuda_device_ids[0]
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    backend = create_pytorch_backend(
        model_name=str(export_dir),
        checkpoint_dir=ckpt_dir,
        device_type=device_type,
        dtype="bfloat16",
        gpu_rank=gpu_rank,
        learning_rate=cfg.trainer.lr,
        weight_decay=cfg.trainer.weight_decay,
        # Use LoRA to mimic "posttrain" shape without needing a dataset.
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        num_minibatches=cfg.trainer.num_minibatches,
        max_grad_norm=cfg.trainer.max_grad_norm,
        # We'll pass sft_loss per-step via forward_backward(loss_fn=...).
        loss_fn=None,
    )

    # Derive vocab size from loaded model
    vocab_size = int(getattr(getattr(backend.model, "config", None), "vocab_size", 50304))

    metrics_history: list[dict[str, float]] = []
    for step in range(cfg.posttrain_steps):
        input_ids = torch.randint(
            0,
            vocab_size,
            (cfg.posttrain_batch_size, cfg.posttrain_seq_len),
            dtype=torch.long,
        )
        labels = input_ids.clone()
        loss_mask = torch.ones_like(labels, dtype=torch.float32)

        batch = {"input_ids": input_ids, "labels": labels, "loss_mask": loss_mask}

        fwd = await backend.forward_backward(batch, loss_fn=sft_loss).result()
        opt = await backend.optim_step().result()
        metrics_history.append({**fwd, **opt, "step": float(step + 1)})

    return {"metrics_history": metrics_history}


def train(config: NanochatSmokeConfig, **_: Any) -> dict[str, Any]:
    """Entry point for `rollouts.run` (local execution path)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    out_dir = _setup_output_dir(config)
    nano_out = out_dir / "nano_pretrain"
    hf_out = out_dir / "hf_export"

    # Run nano-pretrain (functional Llama) into a run-scoped output dir
    nano_cfg = dc_replace(config.nano, output_dir=str(nano_out), run_id="nano_pretrain")
    nano_pretrain_train(nano_cfg, use_real_data=False, resume=False)

    ckpt_path = find_latest_checkpoint(Path(nano_cfg.output_dir))
    if ckpt_path is None:
        raise RuntimeError(f"nano-pretrain produced no checkpoint under {nano_cfg.output_dir}")

    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = ckpt["weights"]
    export_to_hf(weights, nano_cfg.model, hf_out)
    logger.info(f"Exported nano-pretrain checkpoint to HF dir: {hf_out}")

    # Post-train: run a few LoRA steps against the exported HF model
    post = trio.run(_run_lora_posttrain, hf_out, config, out_dir)
    return {
        "nano_checkpoint": str(ckpt_path),
        "hf_export_dir": str(hf_out),
        **post,
    }
