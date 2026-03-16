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

import json
import logging
import math
import os
import shutil
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
from ..weight_sync_protocol import (
    InitWeightUpdateGroupRequest,
    InitWeightUpdateGroupResponse,
    ReceiveWeightUpdateRequest,
    WeightUpdatePayload,
    WeightWireTensor,
)

logger = logging.getLogger(__name__)


def _parse_torch_dtype(name: str) -> torch.dtype:
    normalized = name.replace("torch.", "").lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype name for TorchTitan sync: {name!r}")


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
    micro_batch_size: int | None = None
    num_minibatches: int = 1
    mixed_precision_param: str = "bfloat16"
    mixed_precision_reduce: str = "float32"

    # Optimization
    lr: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Activation checkpointing
    activation_checkpoint_mode: str = "none"
    selective_ac_option: str = "op"
    per_op_sac_force_recompute_mm_shapes_by_fqns: tuple[str, ...] = ()
    early_stop: bool = False
    memory_budget: float = 0.5
    visualize_memory_budget_pareto: bool = False
    preserve_rng_state: bool = True
    determinism_check: str = "default"
    debug: bool = False

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
    _hf_assets_path: str | None = field(default=None, init=False, repr=False)
    _nccl_weight_sender: Any = field(default=None, init=False, repr=False)
    _nccl_inference_endpoints: list[str] = field(default_factory=list, init=False, repr=False)

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
        current_cuda_device = torch.cuda.current_device()
        self._device = torch.device(f"cuda:{current_cuda_device}")

        # Get train spec
        logger.info(
            "[Rank %s] Loading train spec for %s on device %s",
            self.rank,
            self.model_name,
            self._device,
        )
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
        self._hf_assets_path = local_path

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
        provisioning = self.lowering.provisioning

        # Build parallel dims
        self._parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=-1,
            ep=provisioning.ep,
            etp=1,
            tp=provisioning.tp,
            cp=provisioning.cp,
            pp=provisioning.pp,
            world_size=self.world_size,
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
        provisioning = self.lowering.provisioning

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
                context_parallel_degree=provisioning.cp,
                enable_async_tensor_parallel=False,
                disable_loss_parallel=not provisioning.enable_loss_parallel,
                fsdp_reshard_after_forward="default",
            ),
            compile=_Cfg(
                enable=self.config.compile_enabled,
                components=["model"] if self.config.compile_enabled else [],
            ),
            activation_checkpoint=_Cfg(
                mode=self.config.activation_checkpoint_mode,
                selective_ac_option=self.config.selective_ac_option,
                per_op_sac_force_recompute_mm_shapes_by_fqns=list(
                    self.config.per_op_sac_force_recompute_mm_shapes_by_fqns
                ),
                early_stop=self.config.early_stop,
                memory_budget=self.config.memory_budget,
                visualize_memory_budget_pareto=self.config.visualize_memory_budget_pareto,
                preserve_rng_state=self.config.preserve_rng_state,
                determinism_check=self.config.determinism_check,
                debug=self.config.debug,
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
        if self.config.micro_batch_size is not None:
            micro_batch_size = min(self.config.micro_batch_size, batch_size)
            num_minibatches = max(1, math.ceil(batch_size / micro_batch_size))
        else:
            configured_num_minibatches = max(1, self.config.num_minibatches)
            micro_batch_size = max(1, math.ceil(batch_size / configured_num_minibatches))
            num_minibatches = max(1, math.ceil(batch_size / micro_batch_size))

        total_primary_loss = 0.0
        accumulated_losses: dict[str, float] = {}
        accumulated_other_metrics: dict[str, float] = {}
        accumulated_events: list[dict[str, Any]] = []

        for i in range(num_minibatches):
            start_idx = i * micro_batch_size
            if start_idx >= batch_size:
                break
            end_idx = min(start_idx + micro_batch_size, batch_size)
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

    async def save_weights_for_sampler(self, path: Path | str) -> Path:
        """Export inference-ready HuggingFace weights for sampler sync.

        Current TorchTitan realization only supports filesystem sync for
        single-rank adapter-based models. Build the complete HF checkpoint
        explicitly at the backend boundary instead of leaking partial TorchTitan
        state into weight_sync.py.
        """
        import torch.distributed as dist
        import trio
        from huggingface_hub import snapshot_download
        from safetensors.torch import save_file

        assert self._model is not None, "Model not initialized"
        assert self._train_spec is not None, "Train spec not initialized"
        assert self.hf_checkpoint, "TorchTitan sampler export requires hf_checkpoint"
        assert self._train_spec.state_dict_adapter is not None, (
            "TorchTitan sampler export requires a state_dict_adapter"
        )

        output_path = Path(path)
        temp_path = output_path.parent / f"{output_path.name}_tmp_{self.weight_version}"
        rank = dist.get_rank() if dist.is_initialized() else 0

        source_assets_path = self.hf_checkpoint
        if "/" in self.hf_checkpoint and not os.path.exists(self.hf_checkpoint):
            source_assets_path = await trio.to_thread.run_sync(
                lambda: snapshot_download(self.hf_checkpoint)
            )

        if rank == 0:
            if temp_path.exists():
                shutil.rmtree(temp_path)
            temp_path.mkdir(parents=True, exist_ok=True)

            adapter = self._train_spec.state_dict_adapter(self._model_args, source_assets_path)
            native_state_dict = self._model.state_dict()
            hf_state_dict = adapter.to_hf(native_state_dict)

            cpu_state_dict: dict[str, torch.Tensor] = {}
            for key, value in hf_state_dict.items():
                if hasattr(value, "full_tensor"):
                    value = value.full_tensor()
                if isinstance(value, torch.Tensor):
                    cpu_state_dict[key] = value.detach().cpu().contiguous()

            await trio.to_thread.run_sync(
                lambda: save_file(cpu_state_dict, str(temp_path / "model.safetensors"))
            )

            source_path = Path(source_assets_path)
            for asset in source_path.iterdir():
                if not asset.is_file():
                    continue
                if asset.name.endswith((".safetensors", ".bin", ".pt", ".pth")):
                    continue
                if asset.name == "model.safetensors.index.json":
                    continue
                shutil.copy2(asset, temp_path / asset.name)

            config_path = temp_path / "config.json"
            assert config_path.exists(), f"HF asset copy must produce config.json at {config_path}"

            if output_path.exists():
                shutil.rmtree(output_path)
            temp_path.rename(output_path)

        if dist.is_initialized():
            dist.barrier()

        self.weight_version += 1
        return output_path

    def _build_inference_weight_update_payload(self) -> WeightUpdatePayload:
        import torch

        assert self._model is not None, "Model not initialized"
        assert self._train_spec is not None, "Train spec not initialized"
        assert self._train_spec.state_dict_adapter is not None, (
            "TorchTitan NCCL sync requires a state_dict_adapter"
        )

        source_assets_path = self._hf_assets_path or self.hf_checkpoint
        assert source_assets_path, "TorchTitan NCCL sync requires hf_checkpoint assets"

        adapter = self._train_spec.state_dict_adapter(self._model_args, source_assets_path)
        native_state_dict = self._model.state_dict()
        target_dtype = _parse_torch_dtype(self.config.mixed_precision_param)

        tensors: list[WeightWireTensor] = []
        for key, value in native_state_dict.items():
            if hasattr(value, "full_tensor"):
                value = value.full_tensor()
            if isinstance(value, torch.Tensor):
                mapped = adapter.to_hf({key: value})
                if not mapped:
                    continue
                if len(mapped) != 1:
                    raise RuntimeError(
                        "TorchTitan trainer-parameter sync requires a 1:1 native->HF mapping; "
                        f"native key {key!r} mapped to {list(mapped.keys())!r}"
                    )
                load_name, mapped_value = next(iter(mapped.items()))
                if hasattr(mapped_value, "full_tensor"):
                    mapped_value = mapped_value.full_tensor()
                if not isinstance(mapped_value, torch.Tensor):
                    raise RuntimeError(
                        "TorchTitan trainer-parameter sync requires tensor-valued adapter outputs; "
                        f"native key {key!r} produced {type(mapped_value)!r}"
                    )
                if tuple(mapped_value.shape) != tuple(value.shape):
                    raise RuntimeError(
                        "TorchTitan trainer-parameter sync requires shape-preserving adapter mapping; "
                        f"native key {key!r} shape {tuple(value.shape)!r} mapped to "
                        f"{load_name!r} shape {tuple(mapped_value.shape)!r}"
                    )
                prepared = value.detach().to(device=self._device, dtype=target_dtype).contiguous()
                tensors.append(
                    WeightWireTensor(
                        wire_name=key,
                        load_name=load_name,
                        shape=tuple(prepared.shape),
                        dtype=str(prepared.dtype).replace("torch.", ""),
                        tensor=prepared,
                        payload_kind="trainer_parameter",
                        metadata={
                            "source": "torchtitan_native_state_dict",
                            "adapter_load_name": load_name,
                        },
                    )
                )

        assert tensors, "TorchTitan NCCL sync produced no trainer-parameter tensors"
        return WeightUpdatePayload(
            tensors=tuple(tensors),
            payload_kind="trainer_parameter",
            version=self.weight_version + 1,
            metadata={
                "source_contract": "trainer_parameter",
                "adapter": "torchtitan_state_dict_adapter.to_hf",
            },
        )

    async def init_nccl_weight_sync(
        self,
        inference_endpoints: list[str],
        master_addr: str | None = None,
        master_port: int = 29500,
    ) -> None:
        import os
        import socket

        import httpx
        import trio

        from ...inference.weight_sync import WeightSyncSender

        if self._nccl_weight_sender is not None:
            return
        if not inference_endpoints:
            logger.info("TorchTitan NCCL sync skipped (no inference endpoints)")
            return

        if master_addr is None:
            master_addr = "127.0.0.1"

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", master_port))
            master_port = int(sock.getsockname()[1])

        world_size = 1 + len(inference_endpoints)
        # TODO: Move this one-time NCCL group initialization under the managed
        # WeightUpdateChannel.initialize() lifecycle so GRPO/channel semantics own
        # it once per run, QED-style, instead of treating init like a retryable RPC.
        group_name = f"weight_sync_{master_port}"
        sender = WeightSyncSender(
            master_addr=master_addr,
            master_port=master_port,
            inference_world_size=len(inference_endpoints),
            group_name=group_name,
            device=self._device,
        )

        async def register_inference_endpoint(endpoint: str, rank: int) -> None:
            timeout = httpx.Timeout(connect=5.0, read=300.0, write=300.0, pool=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                init_request = InitWeightUpdateGroupRequest(
                    master_address=master_addr,
                    master_port=master_port,
                    rank_offset=rank,
                    world_size=world_size,
                    group_name=group_name,
                )
                logger.info(
                    "[Rank %s] init_weights_update_group request start endpoint=%s rank=%s mode=blocking_once",
                    self.rank,
                    endpoint,
                    rank,
                )
                done = trio.Event()

                async def poll_weight_sync_trace() -> None:
                    last_signature: tuple[str, ...] = ()
                    while not done.is_set():
                        await trio.sleep(5)
                        if done.is_set():
                            break
                        try:
                            trace_response = await client.get(
                                f"{endpoint}/weight_sync_trace", params={"limit": 20}
                            )
                            trace_response.raise_for_status()
                            trace_entries = trace_response.json().get("entries", [])
                            tail_events = tuple(
                                str(entry.get("event", "<missing>"))
                                for entry in trace_entries[-10:]
                            )
                            if tail_events and tail_events != last_signature:
                                last_signature = tail_events
                                logger.info(
                                    "[Rank %s] init_weights_update_group trace_poll endpoint=%s rank=%s events=%s tail=%s",
                                    self.rank,
                                    endpoint,
                                    rank,
                                    list(tail_events),
                                    trace_entries[-3:],
                                )
                        except Exception as trace_exc:
                            logger.info(
                                "[Rank %s] init_weights_update_group trace_poll_failed endpoint=%s rank=%s error_type=%s error=%r",
                                self.rank,
                                endpoint,
                                rank,
                                type(trace_exc).__name__,
                                trace_exc,
                            )

                try:
                    async with trio.open_nursery() as nursery:
                        nursery.start_soon(poll_weight_sync_trace)
                        try:
                            response = await client.post(
                                f"{endpoint}/init_weights_update_group",
                                json=init_request.to_dict(),
                            )
                            logger.info(
                                "[Rank %s] init_weights_update_group response endpoint=%s rank=%s mode=blocking_once status=%s",
                                self.rank,
                                endpoint,
                                rank,
                                response.status_code,
                            )
                            response.raise_for_status()
                            InitWeightUpdateGroupResponse.from_dict(response.json())
                        finally:
                            done.set()
                            nursery.cancel_scope.cancel()
                except Exception as exc:
                    trace_summary = "trace_unavailable"
                    try:
                        trace_response = await client.get(
                            f"{endpoint}/weight_sync_trace", params={"limit": 20}
                        )
                        trace_response.raise_for_status()
                        trace_entries = trace_response.json().get("entries", [])
                        tail_events = [
                            entry.get("event", "<missing>") for entry in trace_entries[-10:]
                        ]
                        trace_summary = json.dumps(
                            {
                                "entry_count": len(trace_entries),
                                "tail_events": tail_events,
                                "tail_entries": trace_entries[-5:],
                            },
                            sort_keys=True,
                        )
                    except Exception as trace_exc:
                        trace_summary = (
                            f"trace_fetch_failed={type(trace_exc).__name__}: {trace_exc!r}"
                        )
                    logger.exception(
                        "[Rank %s] init_weights_update_group failed endpoint=%s rank=%s mode=blocking_once error_type=%s error=%r trace=%s",
                        self.rank,
                        endpoint,
                        rank,
                        type(exc).__name__,
                        exc,
                        trace_summary,
                    )
                    raise RuntimeError(
                        "init_weights_update_group failed "
                        f"endpoint={endpoint} rank={rank} error={type(exc).__name__}: {exc!r} trace={trace_summary}"
                    ) from exc

        def trainer_join() -> None:
            os.environ.setdefault("NCCL_SHM_DISABLE", "1")
            os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
            sender.init_group()

        logger.info(
            "[Rank %s] init_weights_update_group sender_init_start group=%s master=%s:%s world_size=%s",
            self.rank,
            group_name,
            master_addr,
            master_port,
            world_size,
        )
        await trio.to_thread.run_sync(trainer_join, abandon_on_cancel=True)
        logger.info(
            "[Rank %s] init_weights_update_group sender_init_ok group=%s master=%s:%s world_size=%s",
            self.rank,
            group_name,
            master_addr,
            master_port,
            world_size,
        )

        with trio.fail_after(330):
            async with trio.open_nursery() as nursery:
                for i, endpoint in enumerate(inference_endpoints):
                    nursery.start_soon(register_inference_endpoint, endpoint, i + 1)

        self._nccl_weight_sender = sender
        self._nccl_inference_endpoints = list(inference_endpoints)
        logger.info(
            "[Rank %s] TorchTitan NCCL weight sync initialized for %d endpoint(s)",
            self.rank,
            len(inference_endpoints),
        )

    async def sync_weights_nccl(self) -> None:
        import httpx
        import torch
        import trio

        sender = self._nccl_weight_sender
        assert sender is not None, "Call init_nccl_weight_sync() first"
        if not self._nccl_inference_endpoints:
            return

        payload = self._build_inference_weight_update_payload()
        param_info = [
            {
                "name": item.wire_name,
                "load_name": item.load_name,
                "shape": list(item.shape),
                "dtype": item.dtype,
            }
            for item in payload.tensors
        ]
        total_bytes = sum(
            int(item.tensor.numel() * item.tensor.element_size()) for item in payload.tensors
        )
        first_tensors = [
            {
                "wire_name": item.wire_name,
                "load_name": item.load_name,
                "shape": list(item.shape),
                "dtype": item.dtype,
                "device": str(item.tensor.device),
                "numel": int(item.tensor.numel()),
                "is_contiguous": bool(item.tensor.is_contiguous()),
                "stride": list(item.tensor.stride()),
                "payload_kind": item.payload_kind,
            }
            for item in list(payload.tensors[:3])
        ]
        responses: list[dict[str, Any]] = []

        logger.info(
            "[Rank %s] torchtitan_nccl_sync_start payload_kind=%s tensors=%s total_bytes=%s endpoints=%s first_tensors=%s",
            self.rank,
            payload.payload_kind,
            len(param_info),
            total_bytes,
            self._nccl_inference_endpoints,
            first_tensors,
        )

        async with httpx.AsyncClient(timeout=300.0) as client:
            async with trio.open_nursery() as nursery:

                async def request_receive(
                    endpoint: str,
                    *,
                    task_status: trio.TaskStatus[None] = trio.TASK_STATUS_IGNORED,
                ) -> None:
                    receive_request = ReceiveWeightUpdateRequest(
                        names=tuple(item["name"] for item in param_info),
                        load_names=tuple(item["load_name"] for item in param_info),
                        shapes=tuple(tuple(item["shape"]) for item in param_info),
                        dtypes=tuple(item["dtype"] for item in param_info),
                    )
                    task_status.started()
                    response = await client.post(
                        f"{endpoint}/receive_weight_update",
                        json=receive_request.to_dict(),
                    )
                    response.raise_for_status()
                    responses.append(response.json())

                for endpoint in self._nccl_inference_endpoints:
                    await nursery.start(request_receive, endpoint)

                await trio.to_thread.run_sync(sender.broadcast_payload, payload)

        logger.info(
            "[Rank %s] torchtitan_nccl_sync_receive_acks responses=%s",
            self.rank,
            responses,
        )

        self.weight_version += 1
        torch.cuda.empty_cache()
        logger.info(
            "[Rank %s] TorchTitan NCCL synced %d tensors to %d endpoint(s)",
            self.rank,
            len(param_info),
            len(self._nccl_inference_endpoints),
        )

    async def cleanup_nccl_weight_sync(self) -> None:
        import httpx

        sender = self._nccl_weight_sender
        if sender is None:
            return

        async with httpx.AsyncClient(timeout=60.0) as client:
            for endpoint in self._nccl_inference_endpoints:
                try:
                    response = await client.post(f"{endpoint}/destroy_weights_update_group")
                    response.raise_for_status()
                except Exception:
                    pass

        sender.cleanup()
        self._nccl_weight_sender = None
        self._nccl_inference_endpoints = []

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
