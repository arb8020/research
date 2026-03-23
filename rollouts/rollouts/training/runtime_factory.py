"""Training-only backend/runtime construction.

This module owns the shared backend setup story for training consumers that are
not tied to RL orchestration. GRPO uses it today; offline pretraining should be
the next consumer.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from .configs import CheckpointConfig, ModelConfig, TrainerConfig
from .lowering import (
    MegatronLowering,
    MegatronProvisioning,
    RealizationPlan,
    dense_rl_realization,
    dense_supervised_realization,
)

logger = logging.getLogger(__name__)

TrainingMode = Literal["rl", "supervised"]


def trainer_realization_from_config(trainer: TrainerConfig) -> RealizationPlan | None:
    """Return explicit realization intent from trainer config, if any."""
    if not trainer.realization_local_layouts and not trainer.realization_collective_transitions:
        return None
    return RealizationPlan(
        local_layouts=trainer.realization_local_layouts,
        collective_transitions=trainer.realization_collective_transitions,
        packed_sequences=trainer.realization_packed_sequences,
    )


def build_megatron_lowering(
    trainer: TrainerConfig,
    *,
    training_mode: TrainingMode = "rl",
) -> MegatronLowering:
    """Build Megatron lowering from shared trainer config."""
    realization = trainer_realization_from_config(trainer)
    if realization is None:
        if training_mode == "rl":
            realization = dense_rl_realization(
                tp=trainer.tensor_parallel_size,
                cp=trainer.context_parallel_size,
                pp=trainer.pipeline_parallel_size,
                packed_sequences=trainer.realization_packed_sequences,
            )
        elif training_mode == "supervised":
            realization = dense_supervised_realization(
                tp=trainer.tensor_parallel_size,
                cp=trainer.context_parallel_size,
                pp=trainer.pipeline_parallel_size,
                packed_sequences=trainer.realization_packed_sequences,
            )
        else:
            raise ValueError(f"Unknown training_mode for Megatron lowering: {training_mode!r}")

    return MegatronLowering.from_realization(
        provisioning=MegatronProvisioning(
            tp=trainer.tensor_parallel_size,
            pp=trainer.pipeline_parallel_size,
            ep=trainer.expert_parallel_size,
            packed_sequences=trainer.realization_packed_sequences,
        ),
        realization=realization,
    )


def create_training_backend_runtime(
    *,
    model: ModelConfig,
    trainer: TrainerConfig,
    checkpoint: CheckpointConfig,
    output_dir: Path,
    seq_len: int,
    global_batch_size: int,
    loss_fn: Callable[..., Any] | None = None,
    training_mode: TrainingMode = "rl",
    megatron_workers: list[Any] | None = None,
    megatron_inference_endpoints: Sequence[str] = (),
    emit_phase: Callable[[str], None] | None = None,
) -> tuple[Any, Callable[[], None] | None]:
    """Construct a training backend plus optional cleanup.

    This is deliberately training-only. It does not build tokenizers, endpoints,
    or inference-side client surfaces.
    """
    from .backends.pytorch_factory import create_pytorch_backend, parse_dtype

    cleanup: Callable[[], None] | None = None
    backend_name = trainer.backend

    def _emit(event: str) -> None:
        if emit_phase is not None:
            emit_phase(event)

    if backend_name == "pytorch":
        if loss_fn is None:
            raise ValueError("pytorch backend requires an explicit loss_fn")
        gpu_rank = trainer.cuda_device_ids[0]
        backend = create_pytorch_backend(
            model_name=model.name,
            checkpoint_dir=output_dir,
            device_type="cuda",
            dtype=model.dtype,
            gpu_rank=gpu_rank,
            learning_rate=trainer.lr,
            weight_decay=trainer.weight_decay,
            loss_fn=loss_fn,
            num_minibatches=trainer.num_minibatches,
            max_grad_norm=trainer.max_grad_norm,
            use_lora=model.use_lora,
            lora_rank=model.lora_rank,
            lora_alpha=model.lora_alpha,
        )
    elif backend_name == "nmoe":
        from .backends.nmoe_backend import raise_nmoe_backend_unavailable

        raise_nmoe_backend_unavailable(context="create_training_backend_runtime(backend='nmoe')")
    elif backend_name in ("fsdp", "fsdp2"):
        if loss_fn is None:
            raise ValueError(f"{backend_name} backend requires an explicit loss_fn")
        if backend_name == "fsdp2":
            logger.warning(
                "trainer.backend='fsdp2' selected; using FSDPTrainingBackend (fully_shard) "
                "bring-up path for now."
            )

        import os
        import socket

        import torch
        import torch.distributed as dist
        from transformers import AutoModelForCausalLM

        from .backends.fsdp import FSDPConfig, FSDPTrainingBackend

        trainer_gpu = trainer.cuda_device_ids[0]
        torch.cuda.set_device(trainer_gpu)

        if not dist.is_initialized():

            def find_free_port(start_port: int, max_attempts: int = 100) -> int:
                for port in range(start_port, start_port + max_attempts):
                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                            sock.bind(("", port))
                            return port
                    except OSError:
                        continue
                raise RuntimeError(
                    f"No free port found in range {start_port}-{start_port + max_attempts}"
                )

            master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
            master_port = find_free_port(checkpoint.nccl_master_port + 50)

            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_addr}:{master_port}",
                rank=0,
                world_size=1,
            )

            def _cleanup_dist() -> None:
                if dist.is_initialized():
                    dist.destroy_process_group()

            cleanup = _cleanup_dist

        torch_dtype = parse_dtype(model.dtype)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model.name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        def make_optimizer(fsdp_model: torch.nn.Module) -> torch.optim.Optimizer:
            return torch.optim.AdamW(
                fsdp_model.parameters(),
                lr=trainer.lr,
                weight_decay=trainer.weight_decay,
            )

        fsdp_config = FSDPConfig(
            sharding_strategy="FULL_SHARD",
            mixed_precision=(torch_dtype in (torch.bfloat16, torch.float16)),
            gradient_checkpointing=False,
            clip_grad=trainer.max_grad_norm,
        )

        backend = FSDPTrainingBackend(
            model=hf_model,
            optimizer_fn=make_optimizer,
            loss_fn=loss_fn,
            checkpoint_dir=output_dir,
            config=fsdp_config,
            device=torch.device(f"cuda:{trainer_gpu}"),
        )
    elif backend_name == "megatron":
        from .backends.megatron.remote_backend import (
            MegatronRemoteBackend,
            MegatronRemoteConfig,
        )

        if megatron_workers is None:
            raise ValueError(
                "megatron backend requires pre-spawned workers. "
                "Workers must be forked before CUDA initialization. "
                "Pass megatron_workers to create_training_backend_runtime."
            )

        _emit("training_preflight_backend_runtime_create_start")
        lowering = build_megatron_lowering(trainer, training_mode=training_mode)
        megatron_overrides = trainer.megatron_overrides
        sequence_parallel = trainer.sequence_parallel
        if megatron_overrides is not None and megatron_overrides.sequence_parallel is not None:
            sequence_parallel = megatron_overrides.sequence_parallel
        megatron_config = MegatronRemoteConfig(
            model_name=model.name,
            dtype=model.dtype,
            checkpoint_path=model.checkpoint_path,
            lowering=lowering,
            sequence_parallel=sequence_parallel,
            megatron_overrides=megatron_overrides,
            lr=trainer.lr,
            weight_decay=trainer.weight_decay,
            max_grad_norm=trainer.max_grad_norm,
            loss_type=trainer.loss_type,
            mask_ratio_low=trainer.mask_ratio_low,
            mask_ratio_high=trainer.mask_ratio_high,
            micro_batch_size=trainer.micro_batch_size or 1,
            global_batch_size=global_batch_size,
            seq_length=seq_len,
            save_optimizer_state=checkpoint.save_optimizer_state,
            master_port=checkpoint.nccl_master_port,
            inference_endpoints=list(megatron_inference_endpoints),
            cuda_device_ids=trainer.cuda_device_ids,
        )

        backend = MegatronRemoteBackend(
            workers=megatron_workers,
            config=megatron_config,
            checkpoint_dir=output_dir,
            phase_callback=emit_phase,
        )
        _emit("training_preflight_backend_runtime_create_ok")
        _emit("training_preflight_backend_initialize_start")
        backend.initialize()
        _emit("training_preflight_backend_initialize_ok")

        def _cleanup_megatron() -> None:
            backend.shutdown()

        cleanup = _cleanup_megatron
    elif backend_name == "torchtitan":
        from .backends import create_torchtitan_backend

        realization = trainer_realization_from_config(trainer)
        backend, cleanup = create_torchtitan_backend(
            checkpoint_dir=output_dir,
            hf_checkpoint=model.name,
            torchtitan_model=trainer.torchtitan_model,
            torchtitan_model_size=trainer.torchtitan_model_size,
            gpu_rank=trainer.cuda_device_ids[0],
            seq_len=seq_len,
            micro_batch_size=trainer.micro_batch_size,
            num_minibatches=trainer.num_minibatches,
            learning_rate=trainer.lr,
            weight_decay=trainer.weight_decay,
            max_grad_norm=trainer.max_grad_norm,
            tp=trainer.torchtitan_tp,
            cp=trainer.torchtitan_cp,
            pp=trainer.torchtitan_pp,
            packed_sequences=trainer.realization_packed_sequences,
            activation_checkpointing=trainer.activation_checkpointing,
            mode=training_mode,
            realization=realization,
        )
    else:
        raise ValueError(
            f"Unknown trainer backend: {backend_name!r}. "
            "Use 'pytorch', 'fsdp', 'fsdp2', 'nmoe', 'megatron', or 'torchtitan'."
        )

    return backend, cleanup
