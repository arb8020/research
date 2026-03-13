"""TorchTitan backend for distributed training.

Wraps torchtitan's training infrastructure for use with our GRPO loop.
Provides access to torchtitan's 4D parallelism (TP + CP + PP + FSDP),
expert parallelism, and other distributed training features.

Semantic note:
    This backend does not execute our seqax-inspired realization semantics as
    an explicit collective program. `lowering.realization` is currently used
    for validation and lowering only. Actual execution is still TorchTitan's
    backend-native distributed model: `ParallelDims`, `DeviceMesh`, DTensor,
    FSDP, and model-specific parallelization hooks.

Limitations:
    - NCCL weight sync not supported. Use weight_sync_mode="disk" in config.
    - Currently only FSDP parallelism is implemented. TP/CP/PP are stubbed.

Usage:
    from rollouts.training.backends.torchtitan_backend import TorchTitanBackend
    from rollouts.training.models import glm  # Registers GLM with torchtitan

    backend = TorchTitanBackend(
        model_name="glm",
        model_size="4.7-flash",
        checkpoint_dir=Path("checkpoints"),
        loss_fn=grpo_loss,
    )

    # Training loop
    for batch in batches:
        fwd_result = await backend.forward_backward(batch).result()
        opt_result = await backend.optim_step().result()
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..contracts import (
    ForwardProducts,
    LossFnLike,
    StepResult,
    TrainableParameterPolicy,
    TrainingDatum,
)
from ..lowering import TorchTitanLowering
from ..types import ImmediateTrainFuture, TrainFuture

logger = logging.getLogger(__name__)


@dataclass
class TorchTitanConfig:
    """Configuration for TorchTitan backend."""

    # Parallelism
    tp_degree: int = 1
    cp_degree: int = 1
    pp_degree: int = 1
    fsdp_enabled: bool = True

    # Training
    seq_len: int = 4096
    batch_size: int = 1
    mixed_precision_param: str = "bfloat16"
    mixed_precision_reduce: str = "float32"

    # Optimization
    lr: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Activation checkpointing
    activation_checkpoint_mode: str = "none"

    # Compile
    compile_enabled: bool = False


@dataclass
class TorchTitanBackend:
    """TorchTitan backend for distributed training."""

    model_name: str
    model_size: str
    checkpoint_dir: Path
    loss_fn: LossFnLike | Callable[..., Any]

    config: TorchTitanConfig = field(default_factory=TorchTitanConfig)
    lowering: TorchTitanLowering = field(default_factory=TorchTitanLowering)
    hf_checkpoint: str | None = None

    # Set in __post_init__
    rank: int = field(default=0, init=False)
    world_size: int = field(default=1, init=False)
    step: int = 0
    weight_version: int = 0

    # Internal state (initialized in __post_init__)
    _model: Any = field(default=None, init=False, repr=False)
    _model_args: Any = field(default=None, init=False, repr=False)
    _optimizer: Any = field(default=None, init=False, repr=False)
    _train_spec: Any = field(default=None, init=False, repr=False)
    _parallel_dims: Any = field(default=None, init=False, repr=False)
    _device: Any = field(default=None, init=False, repr=False)
    _active_trainable_policy: TrainableParameterPolicy | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize TorchTitan backend."""
        import torch
        import torch.distributed as dist
        from torchtitan.protocols.train_spec import get_train_spec

        # Validate loss_fn
        assert callable(self.loss_fn), f"loss_fn must be callable, got {type(self.loss_fn)}"

        # Ensure distributed is initialized
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed not initialized. "
                "Call dist.init_process_group() before creating TorchTitanBackend."
            )

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._device = torch.device(f"cuda:{self.rank}")

        # Get train spec
        logger.info(f"[Rank {self.rank}] Loading train spec for {self.model_name}")
        self._train_spec = get_train_spec(self.model_name)
        assert self._train_spec is not None, f"Train spec not found for {self.model_name}"

        # Get model args for the specified size
        if self.model_size not in self._train_spec.model_args:
            available = list(self._train_spec.model_args.keys())
            raise ValueError(
                f"Unknown model size '{self.model_size}' for {self.model_name}. "
                f"Available: {available}"
            )

        self._model_args = self._train_spec.model_args[self.model_size]
        logger.info(f"[Rank {self.rank}] Model args: {self._model_args}")

        # Build model
        self._build_model()
        assert self._model is not None, "Model build failed"

        # Load HF checkpoint if provided
        if self.hf_checkpoint:
            self._load_hf_checkpoint(self.hf_checkpoint)

        # Apply parallelization (TP, FSDP, etc.)
        self._apply_parallelization()

        # Build optimizer
        self._build_optimizer()
        assert self._optimizer is not None, "Optimizer build failed"

        logger.info(f"[Rank {self.rank}] TorchTitan backend initialized")

    def _build_model(self) -> None:
        """Build model from train spec."""
        import torch

        logger.info(f"[Rank {self.rank}] Building {self.model_name} model")

        # Create model on meta device first, then materialize empty tensors on
        # the target device before initializing weights. Meta-initialized
        # modules cannot be moved with `.to(...)`; they must be materialized
        # with `to_empty(...)` first.
        with torch.device("meta"):
            self._model = self._train_spec.model_cls(self._model_args)

        # Materialize parameters/buffers on the target device, then initialize.
        self._model = self._model.to_empty(device=self._device)
        self._model.init_weights(buffer_device=self._device)

    def _load_hf_checkpoint(self, checkpoint_path: str) -> None:
        """Load weights from HuggingFace checkpoint."""
        import os

        from huggingface_hub import snapshot_download
        from safetensors import safe_open

        logger.info(f"[Rank {self.rank}] Loading HF checkpoint: {checkpoint_path}")

        # Download if needed (HF repo ID contains /)
        if "/" in checkpoint_path and not os.path.exists(checkpoint_path):
            local_path = snapshot_download(checkpoint_path)
        else:
            local_path = checkpoint_path

        # Load state dict from safetensors
        hf_state_dict: dict[str, Any] = {}

        for filename in os.listdir(local_path):
            if filename.endswith(".safetensors"):
                filepath = os.path.join(local_path, filename)
                with safe_open(filepath, framework="pt") as f:
                    for key in f.keys():
                        hf_state_dict[key] = f.get_tensor(key)

        # Convert using state dict adapter
        if self._train_spec.state_dict_adapter is not None:
            adapter = self._train_spec.state_dict_adapter(self._model_args, local_path)
            state_dict = adapter.from_hf(hf_state_dict)
        else:
            state_dict = hf_state_dict

        # Load into model
        assert self._model is not None
        model_params = len(list(self._model.state_dict().keys()))
        self._model.load_state_dict(state_dict, strict=False)
        logger.info(f"[Rank {self.rank}] Loaded {len(state_dict)} tensors from checkpoint")

        # Warn if suspiciously few parameters loaded (might be wrong model)
        if len(state_dict) < model_params * 0.5:
            logger.warning(
                f"[Rank {self.rank}] Only loaded {len(state_dict)}/{model_params} parameters. "
                "This may indicate a checkpoint format mismatch."
            )

    def _apply_parallelization(self) -> None:
        """Apply TorchTitan-native parallelization.

        Important:
        The current lowering path derives a backend provisioning summary from
        our `RealizationPlan`, but this method does not interpret explicit
        collectives like `materialize(...)` or `all_gather(...)` from our
        semantics. It applies TorchTitan's own parallelization mechanisms.
        """
        from torchtitan.distributed import ParallelDims

        logger.info(f"[Rank {self.rank}] Applying parallelization")
        parallel = self.lowering.parallel

        # Build parallel dims
        self._parallel_dims = ParallelDims(
            tp=parallel.tp,
            cp=parallel.cp,
            pp=parallel.pp,
            world_size=self.world_size,
            enable_loss_parallel=parallel.enable_loss_parallel,
        )

        # Build a minimal job config for parallelize_fn
        job_config = self._build_job_config()

        # Apply parallelization
        assert self._model is not None
        self._model = self._train_spec.parallelize_fn(
            self._model,
            self._parallel_dims,
            job_config,
        )

    def _build_job_config(self) -> Any:
        """Build torchtitan JobConfig from our config.

        Creates a minimal object that mimics torchtitan's JobConfig structure.
        """
        parallel = self.lowering.parallel

        # Use SimpleNamespace-style object that allows arbitrary attributes
        class _Cfg:
            def __init__(self, **kwargs: Any) -> None:
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return _Cfg(
            training=_Cfg(
                seq_len=self.config.seq_len,
                mixed_precision_param=self.config.mixed_precision_param,
                mixed_precision_reduce=self.config.mixed_precision_reduce,
                enable_cpu_offload=False,
            ),
            parallelism=_Cfg(
                context_parallel_degree=parallel.cp,
                enable_async_tensor_parallel=False,
                disable_loss_parallel=not parallel.enable_loss_parallel,
                fsdp_reshard_after_forward="default",
            ),
            compile=_Cfg(
                enable=self.config.compile_enabled,
                components=["model"] if self.config.compile_enabled else [],
            ),
            activation_checkpoint=_Cfg(
                mode=self.config.activation_checkpoint_mode,
            ),
            job=_Cfg(
                dump_folder=str(self.checkpoint_dir),
            ),
            model=_Cfg(
                converters=[],
            ),
            quantize=_Cfg(
                linear=_Cfg(
                    float8=_Cfg(
                        recipe_name="",
                    ),
                ),
            ),
            debug=_Cfg(
                moe_force_load_balance=False,
            ),
        )

    def _build_optimizer(self) -> None:
        """Build optimizer."""
        import torch

        logger.info(f"[Rank {self.rank}] Building optimizer")

        assert self._model is not None
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    def _move_value_to_device(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(self._device)
        if isinstance(value, dict):
            return {k: self._move_value_to_device(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._move_value_to_device(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._move_value_to_device(v) for v in value)
        return value

    def _slice_value(self, value: Any, start_idx: int, end_idx: int) -> Any:
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            return value[start_idx:end_idx]
        if isinstance(value, dict):
            return {k: self._slice_value(v, start_idx, end_idx) for k, v in value.items()}
        if isinstance(value, list):
            return value[start_idx:end_idx]
        if isinstance(value, tuple):
            return value[start_idx:end_idx]
        return value

    def _slice_datum(self, datum: TrainingDatum, start_idx: int, end_idx: int) -> TrainingDatum:
        return TrainingDatum(
            model_input=type(datum.model_input)(
                tokens=self._slice_value(datum.model_input.tokens, start_idx, end_idx),
                positions=self._slice_value(datum.model_input.positions, start_idx, end_idx),
                attention_mask=self._slice_value(
                    datum.model_input.attention_mask, start_idx, end_idx
                ),
                metadata=datum.model_input.metadata,
            ),
            objective_inputs={
                k: self._slice_value(v, start_idx, end_idx)
                for k, v in datum.objective_inputs.items()
            },
            precision_policy=datum.precision_policy,
            trainable_parameter_policy=datum.trainable_parameter_policy,
            metadata=datum.metadata,
        )

    def _forward_products_from_output(self, output: Any) -> ForwardProducts:
        logits = output.logits if hasattr(output, "logits") else output
        values = getattr(output, "values", None)
        hidden_states = getattr(output, "hidden_states", None)
        aux: dict[str, Any] = {}
        if hasattr(output, "router_aux_loss"):
            aux["router_aux_loss"] = output.router_aux_loss
        if hasattr(output, "aux") and isinstance(output.aux, dict):
            aux.update(output.aux)
        return ForwardProducts(
            logits=logits,
            values=values,
            hidden_states=hidden_states,
            aux=aux,
        )

    def _apply_trainable_parameter_policy(self, policy: TrainableParameterPolicy) -> None:
        assert self._model is not None, "Model not initialized"
        for name, param in self._model.named_parameters():
            if param.grad is None:
                continue
            if not policy.allows_param(name):
                param.grad = None

    def _forward_backward_contract_impl(
        self,
        datum: TrainingDatum,
        *,
        loss_fn: LossFnLike,
        operation: str,
    ) -> TrainFuture[StepResult]:
        assert self._model is not None, "Model not initialized"

        self._model.train()
        self._optimizer.zero_grad()

        active_policy = datum.trainable_parameter_policy or TrainableParameterPolicy.full_weight()
        self._active_trainable_policy = active_policy

        datum = TrainingDatum(
            model_input=type(datum.model_input)(
                tokens=self._move_value_to_device(datum.model_input.tokens),
                positions=self._move_value_to_device(datum.model_input.positions),
                attention_mask=self._move_value_to_device(datum.model_input.attention_mask),
                metadata=datum.model_input.metadata,
            ),
            objective_inputs={
                k: self._move_value_to_device(v) for k, v in datum.objective_inputs.items()
            },
            precision_policy=datum.precision_policy,
            trainable_parameter_policy=active_policy,
            metadata=datum.metadata,
        )

        batch_size = datum.model_input.tokens.shape[0]
        num_minibatches = 1
        micro_batch_size = batch_size

        total_primary_loss = 0.0
        accumulated_losses: dict[str, float] = {}
        accumulated_other_metrics: dict[str, float] = {}
        accumulated_events: list[dict[str, Any]] = []

        for i in range(num_minibatches):
            start_idx = i * micro_batch_size
            end_idx = start_idx + micro_batch_size
            micro_datum = self._slice_datum(datum, start_idx, end_idx)

            model_kwargs: dict[str, Any] = {}
            if micro_datum.model_input.positions is not None:
                model_kwargs["position_ids"] = micro_datum.model_input.positions
            if micro_datum.model_input.attention_mask is not None:
                model_kwargs["attention_mask"] = micro_datum.model_input.attention_mask

            output = self._model(micro_datum.model_input.tokens, **model_kwargs)
            products = self._forward_products_from_output(output)
            step_result = loss_fn(products, micro_datum)

            scaled_loss = step_result.backprop_loss / num_minibatches
            scaled_loss.backward()
            self._apply_trainable_parameter_policy(active_policy)
            total_primary_loss += float(step_result.backprop_loss.detach().item())

            for k, v in step_result.losses.items():
                accumulated_losses[k] = accumulated_losses.get(k, 0.0) + float(v)
            for k, v in step_result.other_metrics.items():
                accumulated_other_metrics[k] = accumulated_other_metrics.get(k, 0.0) + float(v)
            accumulated_events.extend(step_result.events)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._model.parameters(),
            self.config.max_grad_norm,
        )
        grad_norm_val = (
            float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        )

        if "total" not in accumulated_losses:
            accumulated_losses["total"] = total_primary_loss / num_minibatches
        accumulated_other_metrics["grad_norm"] = grad_norm_val
        accumulated_other_metrics["num_minibatches"] = float(num_minibatches)
        accumulated_other_metrics["micro_batch_size"] = float(micro_batch_size)

        return ImmediateTrainFuture(
            StepResult(
                backprop_loss=torch.tensor(accumulated_losses["total"], device=self._device),
                losses=accumulated_losses,
                other_metrics=accumulated_other_metrics,
                events=tuple(accumulated_events),
            ),
            operation=operation,
        )

    @property
    def model(self) -> Any:
        """Get the model (for weight sync)."""
        return self._model

    def forward_backward(
        self,
        batch: TrainingDatum | dict[str, Any],
        *,
        loss_fn: LossFnLike | Callable[..., Any] | None = None,
        loss_fn_config: dict[str, float] | None = None,
    ) -> TrainFuture[StepResult | dict[str, float]]:
        """Compute loss and gradients.

        Contract-native `TrainingDatum` is the primary surface. Legacy dict
        batches are still accepted as an edge compatibility path.
        """
        assert self._model is not None, "Model not initialized"
        if loss_fn_config is not None:
            raise ValueError(
                "loss_fn_config is not supported for TorchTitanTrainingBackend.forward_backward yet. "
                "Pass a closure via loss_fn that captures any config instead."
            )

        if isinstance(batch, TrainingDatum):
            active_loss_fn = loss_fn or self.loss_fn
            assert callable(active_loss_fn), "loss_fn must be callable"
            return self._forward_backward_contract_impl(
                batch,
                loss_fn=active_loss_fn,
                operation="forward_backward",
            )

        self._active_trainable_policy = TrainableParameterPolicy.full_weight()
        self._model.train()
        input_ids = batch["input_ids"]
        logits = self._model(input_ids)
        active_loss_fn = loss_fn or self.loss_fn
        loss = active_loss_fn(logits, batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._model.parameters(),
            self.config.max_grad_norm,
        )
        grad_norm_val = (
            float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        )
        metrics: dict[str, float] = {
            "loss": float(loss.item()),
            "grad_norm": grad_norm_val,
        }
        return ImmediateTrainFuture(metrics)

    def optim_step(self) -> TrainFuture[dict[str, float]]:
        """Apply gradients and update weights."""
        assert self._optimizer is not None, "Optimizer not initialized"

        if self._active_trainable_policy is not None:
            self._apply_trainable_parameter_policy(self._active_trainable_policy)

        self._optimizer.step()
        self._optimizer.zero_grad()

        self.step += 1
        self._active_trainable_policy = None

        metrics: dict[str, float] = {
            "lr": float(self._optimizer.param_groups[0]["lr"]),
            "step": float(self.step),
        }

        return ImmediateTrainFuture(metrics)

    def get_weights(self) -> TrainFuture[dict[str, Any]]:
        """Get model weights for syncing."""
        assert self._model is not None, "Model not initialized"
        state_dict = self._model.state_dict()
        return ImmediateTrainFuture(state_dict)

    def load_weights(self, weights: dict[str, Any]) -> TrainFuture[None]:
        """Load model weights."""
        assert self._model is not None, "Model not initialized"
        self._model.load_state_dict(weights)
        return ImmediateTrainFuture(None)

    async def save_checkpoint(self, step: int, metrics: dict[str, float]) -> Path:
        """Save checkpoint."""
        import json

        import torch

        assert self._model is not None
        assert self._optimizer is not None

        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = checkpoint_path / "model.pt"
        torch.save(self._model.state_dict(), model_path)

        # Save optimizer
        optimizer_path = checkpoint_path / "optimizer.pt"
        torch.save(self._optimizer.state_dict(), optimizer_path)

        # Save metadata
        metadata = {
            "step": step,
            "metrics": metrics,
            "model_name": self.model_name,
            "model_size": self.model_size,
        }
        with open(checkpoint_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        self.weight_version += 1
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        return checkpoint_path

    async def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load checkpoint."""
        import json

        import torch

        assert self._model is not None
        assert self._optimizer is not None

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Load model
        model_path = checkpoint_path / "model.pt"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=self._device)
            self._model.load_state_dict(state_dict)

        # Load optimizer
        optimizer_path = checkpoint_path / "optimizer.pt"
        if optimizer_path.exists():
            state_dict = torch.load(optimizer_path, map_location=self._device)
            self._optimizer.load_state_dict(state_dict)

        # Load metadata
        metadata_path = checkpoint_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            self.step = metadata.get("step", 0)

        logger.info(f"Loaded checkpoint from step {self.step}")
