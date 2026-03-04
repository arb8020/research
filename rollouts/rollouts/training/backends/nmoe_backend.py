"""nmoe-style training backend adapter (self-contained).

This backend is meant to support "nmoe-style" training runs without requiring the
external `nmoe` repo to be installed as a Python package.

Scope (pragmatic):
- Loads a HuggingFace model (trust_remote_code=True).
- Uses a hybrid optimizer:
  - AdamW for embeddings/norms/biases/router params.
  - Muon-like optimizer for large 2D weight matrices (optional).

References for parameter grouping and Muon intent:
- /tmp/nmoe/nmoe/opt.py (build_optimizer)

Caveat:
- The real nmoe implementation uses custom CUDA kernels for Muon orthogonalization
  and additional MoE/ZeRO/RDEP machinery. Here we implement a portable approximation
  (QR-based orthogonalization) so configs "work and run" out of the box.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ...training.backends.pytorch import PyTorchTrainingBackend
from ...training.backends.pytorch_factory import (
    compute_device_map_single_gpu,
    parse_dtype,
    wrap_model_with_lora,
)
from ...training.types import TrainerConfig, TrainFuture

logger = logging.getLogger(__name__)


def _is_adam_only_param(name: str) -> bool:
    # Mirrors nmoe's opt.py is_adam_only heuristic.
    return (
        name.endswith(".bias")
        or "norm" in name
        or "embedding" in name
        or "lm_head" in name
        or "bungee" in name
    )


def _orthogonalize_like_muon(update: torch.Tensor) -> torch.Tensor:
    """Approximate Muon "polar express" step using QR.

    nmoe uses a custom CUDA kernel for Newton–Schulz orthogonalization.
    For portability, we orthogonalize via QR in float32.

    This returns an update with orthonormal columns (or rows if M < N),
    preserving the original shape.
    """
    assert update.dim() == 2, f"Expected 2D tensor, got {update.dim()}D"

    m, n = update.shape
    update_fp32 = update.to(dtype=torch.float32)
    if m >= n:
        q, _r = torch.linalg.qr(update_fp32, mode="reduced")
        return q.to(dtype=update.dtype)
    else:
        q, _r = torch.linalg.qr(update_fp32.T, mode="reduced")
        return q.T.to(dtype=update.dtype)


class Muon(torch.optim.Optimizer):
    """Portable Muon-like optimizer for 2D weight matrices.

    Implements the high-level recipe from nmoe:
    - Nesterov momentum base
    - Orthogonalization step (approx via QR)
    - Per-matrix scaling: update_rms * sqrt(max(M, N))
    - Decoupled weight decay (AdamW-style)
    """

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        *,
        lr: float,
        momentum: float,
        weight_decay: float,
        update_rms: float = 0.2,
        orthogonalize: bool = True,
    ) -> None:
        defaults = dict(
            lr=float(lr),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
            update_rms=float(update_rms),
            orthogonalize=bool(orthogonalize),
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> None:  # type: ignore[override]
        if closure is not None:
            raise RuntimeError("Muon does not support closures")

        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            weight_decay = float(group["weight_decay"])
            update_rms = float(group.get("update_rms", 0.2))
            orthogonalize = bool(group.get("orthogonalize", True))

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach()

                if grad.dim() != 2:
                    raise RuntimeError(f"Muon only supports 2D params, got grad.dim()={grad.dim()}")

                # Momentum buffer in fp32 for stability.
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad, dtype=torch.float32)
                buf: torch.Tensor = state["momentum_buffer"]

                grad_fp32 = grad.to(dtype=torch.float32)
                # mom_buf.lerp_(grad, 1 - momentum)  ->  buf = buf*momentum + grad*(1-momentum)
                buf.lerp_(grad_fp32, 1.0 - momentum)
                # update = grad.lerp(buf, momentum)  ->  grad*(1-momentum) + buf*momentum
                update_fp32 = grad_fp32.lerp(buf, momentum)

                update = update_fp32.to(dtype=grad.dtype)
                if orthogonalize:
                    update = _orthogonalize_like_muon(update)

                m, n = update.shape
                update.mul_(update_rms)
                update.mul_(math.sqrt(float(max(m, n))))

                if weight_decay > 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                if update.dtype != p.dtype:
                    update = update.to(dtype=p.dtype)
                p.add_(update, alpha=-lr)


@dataclass
class HybridOptimizer:
    """Duck-typed optimizer wrapper for stepping multiple optimizers."""

    optimizers: dict[str, torch.optim.Optimizer]
    param_groups: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        assert self.optimizers, "optimizers cannot be empty"
        # Expose a combined view (used by debug info / LR reporting).
        param_groups: list[dict[str, Any]] = []
        for opt in self.optimizers.values():
            param_groups.extend(list(opt.param_groups))
        self.param_groups = param_groups

    def zero_grad(self, set_to_none: bool = False) -> None:
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in self.optimizers.values():
            opt.step()

    def state_dict(self) -> dict[str, Any]:
        return {name: opt.state_dict() for name, opt in self.optimizers.items()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for name, opt in self.optimizers.items():
            if name in state_dict:
                opt.load_state_dict(state_dict[name])


@dataclass(frozen=True)
class NmoeConfig:
    """nmoe-style optimizer configuration.

    This is intentionally small and focused on optimizer behavior used in this repo.
    """

    dtype: str = "bfloat16"
    # AdamW groups
    lr_dense: float = 3e-4
    lr_router: float = 3e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    # Muon groups
    # NOTE: Default to False for portability/perf. The portable QR-based
    # orthogonalization is much slower than nmoe's CUDA kernel.
    use_muon: bool = False
    lr_muon: float = 3.4e-4
    muon_momentum: float = 0.95
    muon_update_rms: float = 0.2
    muon_orthogonalize: bool = True


def _build_nmoe_optimizers(
    model: torch.nn.Module,
    cfg: NmoeConfig,
) -> HybridOptimizer:
    """Build a hybrid optimizer matching nmoe's parameter grouping approach."""
    muon_params: list[torch.nn.Parameter] = []
    dense_decay: list[torch.nn.Parameter] = []
    dense_no_decay: list[torch.nn.Parameter] = []
    router_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Router: separate group, no decay.
        if "router" in name:
            router_params.append(param)
            continue

        is_2d_weight = param.dim() == 2 and param.numel() > 1024
        if cfg.use_muon and is_2d_weight and not _is_adam_only_param(name):
            muon_params.append(param)
            continue

        if _is_adam_only_param(name) or param.dim() < 2:
            dense_no_decay.append(param)
        else:
            dense_decay.append(param)

    adam_groups: list[dict[str, Any]] = []
    if dense_decay:
        adam_groups.append({
            "name": "dense_decay",
            "params": dense_decay,
            "lr": cfg.lr_dense,
            "weight_decay": cfg.weight_decay,
        })
    if dense_no_decay:
        adam_groups.append({
            "name": "dense_no_decay",
            "params": dense_no_decay,
            "lr": cfg.lr_dense,
            "weight_decay": 0.0,
        })
    if router_params:
        adam_groups.append({
            "name": "router",
            "params": router_params,
            "lr": cfg.lr_router,
            "weight_decay": 0.0,
        })

    if not adam_groups and not muon_params:
        raise RuntimeError("No trainable parameters found for NmoeTrainingBackend")

    optimizers: dict[str, torch.optim.Optimizer] = {}
    if adam_groups:
        optimizers["adamw"] = torch.optim.AdamW(
            adam_groups,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            eps=cfg.adam_eps,
        )
    if muon_params:
        optimizers["muon"] = Muon(
            muon_params,
            lr=cfg.lr_muon,
            momentum=cfg.muon_momentum,
            weight_decay=cfg.weight_decay,
            update_rms=cfg.muon_update_rms,
            orthogonalize=cfg.muon_orthogonalize,
        )

    return HybridOptimizer(optimizers=optimizers)


@dataclass
class NmoeTrainingBackend:
    """nmoe-style backend implemented via a wrapped PyTorchTrainingBackend."""

    model_name: str
    checkpoint_dir: Path
    loss_fn: Callable[[torch.Tensor, dict[str, Any]], Any]
    config: NmoeConfig = field(default_factory=NmoeConfig)
    device_type: str = "cuda"
    gpu_rank: int = 0
    num_minibatches: int | None = None
    max_grad_norm: float | None = 1.0
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32

    _inner: PyTorchTrainingBackend = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from transformers import AutoModelForCausalLM

        torch_dtype = parse_dtype(self.config.dtype)
        device_map = compute_device_map_single_gpu(self.device_type, self.gpu_rank)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        is_lora = False
        if self.use_lora:
            model = wrap_model_with_lora(
                model, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha
            )
            is_lora = True

        model.train()

        optimizer = _build_nmoe_optimizers(model, self.config)

        device = (
            torch.device(f"{self.device_type}:{self.gpu_rank}")
            if self.device_type == "cuda"
            else torch.device(self.device_type)
        )

        trainer_config = TrainerConfig(
            num_minibatches=self.num_minibatches,
            max_grad_norm=self.max_grad_norm,
        )

        self._inner = PyTorchTrainingBackend(
            model=model,
            optimizer=optimizer,  # type: ignore[arg-type]  # duck-typed wrapper
            loss_fn=self.loss_fn,
            checkpoint_dir=self.checkpoint_dir,
            device=device,
            trainer_config=trainer_config,
            is_lora=is_lora,
        )

        logger.info(
            "Initialized NmoeTrainingBackend",
            extra={
                "model": self.model_name,
                "gpu_rank": self.gpu_rank,
                "use_muon": self.config.use_muon,
                "use_lora": self.use_lora,
            },
        )

    # Expose common attributes used by the rest of the stack.
    @property
    def model(self) -> torch.nn.Module:
        return self._inner.model

    @property
    def optimizer(self) -> Any:
        return self._inner.optimizer

    @property
    def weight_version(self) -> int:
        return self._inner.weight_version

    # TrainingBackend protocol methods
    def forward_backward(
        self,
        batch: dict[str, Any],
        *,
        loss_fn: Callable[..., Any] | None = None,
        loss_fn_config: dict[str, float] | None = None,
    ) -> TrainFuture[dict[str, float]]:
        return self._inner.forward_backward(batch, loss_fn=loss_fn, loss_fn_config=loss_fn_config)

    def optim_step(self) -> TrainFuture[dict[str, float]]:
        return self._inner.optim_step()

    def get_weights(self) -> TrainFuture[dict[str, Any]]:
        return self._inner.get_weights()

    def load_weights(self, weights: dict[str, Any]) -> TrainFuture[None]:
        return self._inner.load_weights(weights)

    # Common extra methods (used by GRPO orchestration / weight sync)
    async def save_checkpoint(self, step: int, metrics: dict[str, float]) -> Path:
        return await self._inner.save_checkpoint(step, metrics)

    async def load_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        return await self._inner.load_checkpoint(checkpoint_path)

    async def save_weights_for_sampler(self, path: Path | str) -> Path:
        return await self._inner.save_weights_for_sampler(path)

    async def init_nccl_weight_sync(
        self,
        inference_endpoints: list[str],
        master_addr: str | None = None,
        master_port: int = 29500,
    ) -> None:
        await self._inner.init_nccl_weight_sync(
            inference_endpoints=inference_endpoints,
            master_addr=master_addr,
            master_port=master_port,
        )

    async def sync_weights_nccl(self) -> None:
        await self._inner.sync_weights_nccl()

    async def cleanup_nccl_weight_sync(self) -> None:
        await self._inner.cleanup_nccl_weight_sync()

    def __getattr__(self, name: str) -> Any:
        # Delegate any additional helpers (e.g., save_hf_checkpoint) to the inner backend.
        return getattr(self._inner, name)
