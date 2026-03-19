"""Offline dense pretraining on the shared training substrate."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import trio

from .configs import CheckpointConfig, ModelConfig, OutputConfig, TrainerConfig
from .contract_witnesses import pretrain_batch_to_datum, pretrain_contract_loss
from .metrics import JSONLLogger
from .runtime_factory import build_megatron_lowering, create_training_backend_runtime
from .train import TrainResult, train

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PretrainSourceConfig:
    """One tokenized pretraining source for the deterministic loader."""

    id: str
    paths: tuple[str, ...]
    weight: float = 1.0

    def __post_init__(self) -> None:
        assert self.id.strip(), "PretrainSourceConfig.id cannot be empty"
        assert self.paths, "PretrainSourceConfig.paths cannot be empty"
        assert self.weight > 0, "PretrainSourceConfig.weight must be positive"

    def to_loader_source(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "paths": list(self.paths),
            "weight": self.weight,
        }


@dataclass(frozen=True)
class PretrainDataConfig:
    """Dense token-stream data settings for offline pretraining."""

    source_path: str | None = None
    sources: tuple[PretrainSourceConfig, ...] = ()
    seq_len: int = 1024
    batch_size: int = 8

    def __post_init__(self) -> None:
        assert self.seq_len > 0, "seq_len must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        if self.source_path is None and not self.sources:
            raise ValueError("PretrainDataConfig requires source_path or sources")
        if self.source_path is not None and self.sources:
            raise ValueError("Use either source_path or sources, not both")

    def loader_sources(self) -> str | list[dict[str, Any]]:
        if self.source_path is not None:
            return self.source_path
        return [source.to_loader_source() for source in self.sources]


@dataclass(frozen=True)
class PretrainConfig:
    """Configuration for offline dense pretraining on the shared stack."""

    data: PretrainDataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(loss_type="vanilla"))
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    output: OutputConfig = field(
        default_factory=lambda: OutputConfig(
            output_dir="results/pretrain", experiment_name="pretrain"
        )
    )

    def __post_init__(self) -> None:
        if self.trainer.loss_type != "vanilla":
            raise ValueError("PretrainConfig currently supports only trainer.loss_type='vanilla'.")

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(asdict(self), handle, indent=2)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _output_dir(config: PretrainConfig) -> Path:
    return Path(config.output.output_dir) / config.output.experiment_name


async def _pretrain_batch_iterator(config: PretrainConfig) -> AsyncIterator[Any]:
    from ..pretrain.dataloader import build_loader

    loader = build_loader(
        sources=config.data.loader_sources(),
        seq_len=config.data.seq_len,
        batch_size=config.data.batch_size,
        rank=0,
        world_size=1,
        device="cpu",
    )

    while True:
        input_ids, labels = loader.next()
        yield pretrain_batch_to_datum(input_ids, labels)
        await trio.lowlevel.checkpoint()


async def _save_backend_checkpoint(backend: Any, step: int, metrics: dict[str, Any]) -> Path:
    save_checkpoint = getattr(backend, "save_checkpoint", None)
    if save_checkpoint is None:
        raise ValueError(f"Backend {type(backend).__name__} does not support save_checkpoint")

    result = save_checkpoint(step, metrics)
    if hasattr(result, "result") and callable(result.result):
        return await result.result()
    return await result


def _spawn_megatron_workers(config: PretrainConfig) -> list[Any]:
    from .backends.megatron.remote_backend import (
        MegatronRemoteConfig,
        spawn_megatron_workers,
    )

    lowering = build_megatron_lowering(config.trainer, training_mode="supervised")
    megatron_config = MegatronRemoteConfig(
        model_name=config.model.name,
        dtype=config.model.dtype,
        checkpoint_path=config.model.checkpoint_path,
        lowering=lowering,
        sequence_parallel=config.trainer.sequence_parallel,
        lr=config.trainer.lr,
        weight_decay=config.trainer.weight_decay,
        max_grad_norm=config.trainer.max_grad_norm,
        loss_type=config.trainer.loss_type,
        mask_ratio_low=config.trainer.mask_ratio_low,
        mask_ratio_high=config.trainer.mask_ratio_high,
        micro_batch_size=config.trainer.micro_batch_size or 1,
        global_batch_size=config.data.batch_size,
        seq_length=config.data.seq_len,
        master_port=config.checkpoint.nccl_master_port,
        inference_endpoints=[],
        cuda_device_ids=config.trainer.cuda_device_ids,
    )

    return spawn_megatron_workers(
        num_gpus=len(config.trainer.cuda_device_ids),
        config=megatron_config,
    )


async def _run_pretrain_async(config: PretrainConfig) -> TrainResult:
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.json")

    metrics_logger = JSONLLogger(output_dir)
    backend = None
    cleanup = None
    megatron_workers: list[Any] | None = None

    if config.trainer.backend == "megatron":
        logger.info(
            "Spawning %d Megatron workers for offline pretraining...",
            len(config.trainer.cuda_device_ids),
        )
        megatron_workers = _spawn_megatron_workers(config)

    try:
        backend, cleanup = create_training_backend_runtime(
            model=config.model,
            trainer=config.trainer,
            checkpoint=config.checkpoint,
            output_dir=output_dir,
            seq_len=config.data.seq_len,
            global_batch_size=config.data.batch_size,
            loss_fn=pretrain_contract_loss,
            training_mode="supervised",
            megatron_workers=megatron_workers,
        )

        async def process_batch(
            step: int,
            batch: Any,
            training_backend: Any,
        ) -> dict[str, Any]:
            del step
            fwd_result = await training_backend.forward_backward(
                batch,
                loss_fn=pretrain_contract_loss,
            ).result()
            opt_metrics = await training_backend.optim_step().result()
            return {
                **fwd_result.losses,
                **fwd_result.other_metrics,
                **opt_metrics,
            }

        async def save_checkpoint(step: int, metrics: dict[str, Any]) -> Path:
            assert backend is not None
            return await _save_backend_checkpoint(backend, step, metrics)

        return await train(
            config=config.checkpoint,
            backend=backend,
            batch_iterator=_pretrain_batch_iterator(config),
            process_batch=process_batch,
            weight_syncer=None,
            save_checkpoint=save_checkpoint,
            metrics_logger=metrics_logger,
            logger=logger,
        )
    finally:
        if cleanup is not None:
            cleanup()


def run_pretrain(config: PretrainConfig) -> TrainResult:
    """Run offline pretraining on the shared training stack."""
    return trio.run(_run_pretrain_async, config)
