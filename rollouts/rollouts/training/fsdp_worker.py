"""FSDP worker entry point (multi-node launcher target).

This module is launched once per trainer GPU by `rollouts.training.multi_node`.
It is intentionally self-contained: it reads a JSON config, initializes
torch.distributed via env://, constructs an FSDP backend, and runs a small
training loop.

Important:
- This worker is infrastructure plumbing, not a full GRPO orchestration stack.
- It does NOT generate real rollouts. It uses synthetic batches so that:
  - forward/backward works end-to-end under FSDP
  - checkpoints save
  - weight sync to SGLang via NCCL can be exercised

Config contract:
- Expects a serialized `GRPOConfig` written by the launcher.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import trio

logger = logging.getLogger(__name__)


def init_distributed() -> tuple[int, int, int]:
    """Initialize torch.distributed from env vars.

    Expected env vars (set by launcher):
        MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK, LOCAL_RANK
    """
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def _setup_logging(rank: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [%(levelname)s] [rank={rank}] %(name)s: %(message)s",
    )

    # Silence noisy HTTP loggers unless debugging.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _load_config(path: str) -> Any:  # GRPOConfig
    from rollouts.training.grpo import GRPOConfig

    with open(path) as f:
        data = json.load(f)
    return GRPOConfig.from_dict(data)


def _select_loss_fn(config: Any) -> Any:
    from rollouts.training.losses import grpo_loss, grpo_loss_clipped, grpo_loss_masked

    if config.trainer.loss_type == "vanilla":
        return grpo_loss
    if config.trainer.loss_type == "clipped":
        return grpo_loss_clipped
    if config.trainer.loss_type == "masked":
        return grpo_loss_masked
    raise ValueError(
        f"Unknown trainer.loss_type={config.trainer.loss_type!r}. Use 'vanilla', 'clipped', or 'masked'."
    )


def _load_model_sequential(
    model_name: str,
    torch_dtype: Any,
    rank: int,
    world_size: int,
) -> Any:
    """Load HF model sequentially across ranks to avoid HF cache races."""
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM

    model = None
    for i in range(world_size):
        if i == rank:
            logger.info(f"Loading model on rank {rank}: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        dist.barrier()

    assert model is not None, "Model load failed"
    model.train()
    return model


def _create_fsdp_backend(
    config: Any,  # GRPOConfig
    output_dir: Path,
    local_rank: int,
    rank: int,
    world_size: int,
) -> Any:
    import torch

    from rollouts.training.backends.fsdp import FSDPConfig, FSDPTrainingBackend
    from rollouts.training.backends.pytorch_factory import parse_dtype

    loss_fn = _select_loss_fn(config)
    torch_dtype = parse_dtype(config.model.dtype)

    model = _load_model_sequential(
        model_name=config.model.name,
        torch_dtype=torch_dtype,
        rank=rank,
        world_size=world_size,
    )

    def make_optimizer(fsdp_model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            fsdp_model.parameters(),
            lr=config.trainer.lr,
            weight_decay=config.trainer.weight_decay,
        )

    fsdp_cfg = FSDPConfig(
        sharding_strategy="FULL_SHARD",
        mixed_precision=(torch_dtype in (torch.bfloat16, torch.float16)),
        cpu_offload=False,
        gradient_checkpointing=False,
        clip_grad=config.trainer.max_grad_norm,
    )

    # IMPORTANT: local_rank, not global rank. In multi-node mode the launcher
    # sets CUDA_VISIBLE_DEVICES to a single GPU per process.
    device = torch.device(f"cuda:{local_rank}")

    return FSDPTrainingBackend(
        model=model,
        optimizer_fn=make_optimizer,
        loss_fn=loss_fn,
        checkpoint_dir=output_dir,
        config=fsdp_cfg,
        device=device,
    )


async def _synthetic_batches(config: Any) -> AsyncIterator[dict[str, Any]]:
    """Yield synthetic GRPO-shaped batches for infra testing."""
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=True)
    vocab_size = int(getattr(tokenizer, "vocab_size", 32000))

    batch_size = config.rollout.batch_size * config.rollout.n_samples_per_prompt
    seq_len = min(int(config.rollout.max_seq_len), 128)

    for _step in range(int(config.checkpoint.num_steps)):
        # Keep tensors on CPU; backend moves to device.
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
        labels = input_ids.clone()
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
        advantages = torch.randn(batch_size, dtype=torch.float32)

        batch: dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "advantages": advantages,
        }

        if config.trainer.loss_type in ("clipped", "masked"):
            # Sequence-level old logprobs (dummy).
            batch["old_logprobs"] = torch.zeros(batch_size, dtype=torch.float32)

        yield batch


async def _run_worker(args: argparse.Namespace) -> dict[str, Any]:
    import torch.distributed as dist

    rank, world_size, local_rank = init_distributed()
    _setup_logging(rank)

    config = _load_config(args.config)

    output_dir = Path(config.output.output_dir) / config.output.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = _create_fsdp_backend(
        config=config,
        output_dir=output_dir,
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
    )

    # Optional NCCL weight sync to inference (rank 0 joins the stateless PG).
    inference_endpoints_raw = os.environ.get("INFERENCE_ENDPOINTS", "")
    inference_endpoints = [e for e in inference_endpoints_raw.split(",") if e]

    if config.checkpoint.weight_sync_mode == "nccl" and inference_endpoints:
        if rank == 0:
            logger.info(f"Initializing NCCL weight sync to {len(inference_endpoints)} engine(s)")
        await backend.init_nccl_weight_sync(
            inference_endpoints=inference_endpoints,
            master_port=int(config.checkpoint.nccl_master_port),
        )

    from rollouts.training.train import train as train_loop
    from rollouts.training.weight_sync import BackendNCCLWeightSyncer

    weight_syncer = None
    if config.checkpoint.weight_sync_mode == "nccl" and inference_endpoints:
        # Must exist on all ranks (backend.sync_weights_nccl is collective under FSDP).
        weight_syncer = BackendNCCLWeightSyncer(backend=backend, log=logger if rank == 0 else None)
    elif config.checkpoint.weight_sync_mode == "disk":
        if rank == 0:
            logger.warning(
                "weight_sync_mode='disk' is not supported in fsdp_worker (multi-node inference is remote). "
                "Use weight_sync_mode='nccl' for multi-node runs."
            )

    async def process_batch(_step: int, batch: dict[str, Any], _backend: Any) -> dict[str, Any]:
        fb = await backend.forward_backward(batch).result()
        opt = await backend.optim_step().result()
        return {**fb, **opt}

    async def save_checkpoint(step: int, step_metrics: dict[str, Any]) -> Path:
        numeric = {k: float(v) for k, v in step_metrics.items() if isinstance(v, (int, float))}
        ckpt_dir = await backend.save_checkpoint(step, numeric)
        if rank == 0:
            logger.info(f"Saved checkpoint: {ckpt_dir}")
        return ckpt_dir

    metrics_logger = None
    if rank == 0:
        from rollouts.training.metrics import JSONLLogger

        metrics_logger = JSONLLogger(output_dir)

    result = await train_loop(
        config=config.checkpoint,
        backend=backend,
        batch_iterator=_synthetic_batches(config),
        process_batch=process_batch,
        weight_syncer=weight_syncer,
        save_checkpoint=save_checkpoint,
        metrics_logger=metrics_logger,
        logger=logger,
    )

    # Extra best-effort cleanup (train_loop also closes weight_syncer).
    if config.checkpoint.weight_sync_mode == "nccl":
        try:
            await backend.cleanup_nccl_weight_sync()
        except Exception:
            pass

    dist.barrier()
    dist.destroy_process_group()

    return {"metrics_history": result.metrics_history}


def main() -> None:
    parser = argparse.ArgumentParser(description="FSDP worker (multi-node launcher target)")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument("--is-rank-0", type=int, default=0, help="Ignored (rank derived from env)")
    args = parser.parse_args()

    trio.run(_run_worker, args)


if __name__ == "__main__":
    main()
