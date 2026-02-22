"""TorchTitan backend for distributed training.

Wraps torchtitan's training infrastructure for use with our GRPO loop.
Provides access to torchtitan's 4D parallelism (TP + CP + PP + FSDP),
expert parallelism, and other distributed training features.

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
    loss_fn: Callable[..., Any]

    config: TorchTitanConfig = field(default_factory=TorchTitanConfig)
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

        # Create model on meta device first (for large models)
        with torch.device("meta"):
            self._model = self._train_spec.model_cls(self._model_args)

        # Initialize weights
        self._model.init_weights(buffer_device=self._device)

        # Move to device
        self._model = self._model.to(self._device)

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
        """Apply torchtitan parallelization (TP, FSDP, etc.)."""
        from torchtitan.distributed import ParallelDims

        logger.info(f"[Rank {self.rank}] Applying parallelization")

        # Build parallel dims
        self._parallel_dims = ParallelDims(
            tp=self.config.tp_degree,
            cp=self.config.cp_degree,
            pp=self.config.pp_degree,
            world_size=self.world_size,
            enable_loss_parallel=True,
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
                context_parallel_degree=self.config.cp_degree,
                enable_async_tensor_parallel=False,
                disable_loss_parallel=False,
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

    @property
    def model(self) -> Any:
        """Get the model (for weight sync)."""
        return self._model

    def forward_backward(self, batch: dict[str, Any]) -> TrainFuture[dict[str, float]]:
        """Compute loss and gradients."""
        import torch

        assert self._model is not None, "Model not initialized"

        self._model.train()

        input_ids = batch["input_ids"]

        # Forward pass
        logits = self._model(input_ids)

        # Compute loss using provided loss_fn
        loss = self.loss_fn(logits, batch)

        # Backward pass
        loss.backward()

        # Compute grad norm
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._model.parameters(),
            self.config.max_grad_norm,
        )

        # Handle grad_norm type (could be tensor or float)
        if isinstance(grad_norm, torch.Tensor):
            grad_norm_val = float(grad_norm.item())
        else:
            grad_norm_val = float(grad_norm)

        metrics: dict[str, float] = {
            "loss": float(loss.item()),
            "grad_norm": grad_norm_val,
        }

        return ImmediateTrainFuture(metrics)

    def optim_step(self) -> TrainFuture[dict[str, float]]:
        """Apply gradients and update weights."""
        assert self._optimizer is not None, "Optimizer not initialized"

        self._optimizer.step()
        self._optimizer.zero_grad()

        self.step += 1

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
