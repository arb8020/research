"""nmoe training backend adapter.

Wraps nmoe's Zero2 + Muon/AdamW training for MoE models.

nmoe is optimized for B200 GPUs with:
- Zero2 for dense params (gradient sharding)
- RDEP for expert params (IPC-based expert parallelism)
- Muon optimizer (Newton-Schulz orthogonalization) for 2D matrices
- Blockscaled FP4/FP8 for memory efficiency

Reference: https://github.com/Noumena-Network/nmoe

Usage:
    backend = NmoeTrainingBackend(
        model=model,
        config=nmoe_config,
        checkpoint_dir=Path("/checkpoints"),
    )
    metrics = await backend.forward_backward(batch).result()
    step_metrics = await backend.optim_step().result()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...training.types import ImmediateTrainFuture, TrainFuture

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class NmoeConfig:
    """Configuration for nmoe training backend.

    Mirrors nmoe.config.Config fields needed for training.

    Attributes:
        dtype: Model dtype ("bf16", "fp8", "nvfp4")
        lr_dense: Learning rate for dense params (AdamW)
        lr_router: Learning rate for router params
        lr_expert: Learning rate for expert params
        lr_muon: Learning rate for Muon optimizer (2D matrices)
        weight_decay: Weight decay coefficient
        adam_beta1: AdamW beta1
        adam_beta2: AdamW beta2 for dense params
        adam_beta2_expert: AdamW beta2 for experts (higher for FP8/NVFP4 noise)
        adam_eps: AdamW epsilon
        muon_momentum: Muon momentum coefficient
        muon_update_rms: Muon update RMS scaling
        aux_loss_alpha: MoE auxiliary loss coefficient (load balancing)
        seed: Random seed
    """

    dtype: str = "bf16"
    lr_dense: float = 3e-4
    lr_router: float = 3e-4
    lr_expert: float = 3e-4
    lr_muon: float = 3.4e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_beta2_expert: float = 0.99
    adam_eps: float = 1e-8
    muon_momentum: float = 0.95
    muon_update_rms: float = 0.2
    aux_loss_alpha: float = 0.0
    seed: int = 42


@dataclass
class NmoeTrainingBackend:
    """nmoe training backend for MoE models.

    Implements TrainingBackend protocol using nmoe's Zero2 + Muon/AdamW.

    This is a thin adapter that wraps nmoe's training primitives.
    The actual model and optimizers are created externally and passed in.

    Attributes:
        model: PyTorch model (MoE architecture expected)
        config: NmoeConfig with optimizer hyperparameters
        checkpoint_dir: Directory for checkpoints

    Example:
        >>> from nmoe.model import Transformer
        >>> model = Transformer(cfg).cuda()
        >>> backend = NmoeTrainingBackend(
        ...     model=model,
        ...     config=NmoeConfig(dtype="bf16"),
        ...     checkpoint_dir=Path("/checkpoints"),
        ... )
        >>> metrics = await backend.forward_backward(batch).result()
    """

    model: Any  # nn.Module (MoE)
    config: NmoeConfig
    checkpoint_dir: Path

    # Internal state
    _dense_optimizer: Any = field(default=None, init=False)
    _muon_optimizer: Any = field(default=None, init=False)
    _expert_optimizer: Any = field(default=None, init=False)
    _zero2_state: dict = field(default_factory=dict, init=False)
    _step: int = field(default=0, init=False)
    _initialized: bool = field(default=False, init=False)

    def _lazy_init(self) -> None:
        """Initialize optimizers on first use.

        Deferred to allow model to be moved to GPU first.
        """
        if self._initialized:
            return

        try:
            from nmoe.opt import build_optimizer
        except ImportError as e:
            raise ImportError(
                "nmoe is required for NmoeTrainingBackend. "
                "Install from: https://github.com/Noumena-Network/nmoe"
            ) from e

        # Build optimizers using nmoe's factory
        # This creates Muon for 2D matrices, AdamW for embeddings/norms/biases
        optimizers = build_optimizer(self.model, self.config)
        self._dense_optimizer = optimizers.get("dense")
        self._muon_optimizer = optimizers.get("muon")
        self._expert_optimizer = optimizers.get("expert")

        self._initialized = True
        logger.info("NmoeTrainingBackend initialized")

    def forward_backward(self, batch: dict[str, Any]) -> TrainFuture[dict[str, float]]:
        """Compute loss and gradients using nmoe's chunked cross-entropy.

        Args:
            batch: {
                "input_ids": Tensor [batch, seq_len],
                "labels": Tensor [batch, seq_len],
                "advantages": Optional Tensor [batch] for RL,
            }

        Returns:
            Future resolving to {"loss": float, "aux_loss": float, ...}
        """
        self._lazy_init()

        import torch

        try:
            from quack.linear_cross_entropy import chunked_linear_cross_entropy
        except ImportError as e:
            raise ImportError(
                "quack is required for nmoe's chunked cross-entropy. "
                "See nmoe installation instructions."
            ) from e

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # Forward pass
        hidden = self.model(input_ids, return_hidden=True)

        # Compute loss using quack's memory-efficient CE
        logits_gain = float(getattr(self.model, "fp4_logits_gain", 1.0))
        x = (hidden * logits_gain).reshape(-1, hidden.shape[-1])
        t = labels.reshape(-1)

        ce_loss = chunked_linear_cross_entropy(
            x,
            self.model.lm_head.weight,
            t,
            chunk_size=8192,
            ignore_index=-100,
            reduction="mean",
            tuned=False,
        )

        # MoE auxiliary loss (load balancing)
        aux_loss = torch.tensor(0.0, device=ce_loss.device)
        if self.config.aux_loss_alpha > 0.0:
            try:
                from nmoe.moe import MoE

                moe_layers = [
                    blk.ffn
                    for blk in self.model.blocks
                    if isinstance(getattr(blk, "ffn", None), MoE)
                ]
                aux_losses = [m.last_aux_loss for m in moe_layers if m.last_aux_loss is not None]
                if aux_losses:
                    aux_loss = torch.stack(aux_losses).mean()
            except ImportError:
                pass

        loss = ce_loss + self.config.aux_loss_alpha * aux_loss

        # RL advantage weighting if present
        advantages = batch.get("advantages")
        if advantages is not None:
            # Per-sequence weighting for policy gradient
            # This is a simplified version - full GRPO uses sequence-level logprobs
            loss = loss * advantages.mean()

        # Backward pass
        self.model.zero_grad(set_to_none=True)
        loss.backward()

        metrics = {
            "loss": float(ce_loss.detach()),
            "aux_loss": float(aux_loss.detach()),
            "total_loss": float(loss.detach()),
        }

        return ImmediateTrainFuture(metrics)

    def optim_step(self) -> TrainFuture[dict[str, float]]:
        """Apply gradients using nmoe's Zero2 + Muon/AdamW.

        Returns:
            Future resolving to {"lr_dense": float, "lr_muon": float, "step": int, ...}
        """
        self._lazy_init()

        try:
            from nmoe.opt import step as nmoe_step
            from nmoe.zero2 import step_dense_adamw
        except ImportError as e:
            raise ImportError("nmoe is required for NmoeTrainingBackend.") from e

        import torch.distributed as dist

        world = dist.get_world_size() if dist.is_initialized() else 1

        # nmoe step: handles Zero2 reduce-scatter/all-gather + optimizer updates
        nmoe_step(
            self.model,
            self._expert_optimizer,
            self._muon_optimizer,
            dense_groups=None,  # Will use default grouping
            zero2_state=self._zero2_state,
            cfg=self.config,
            world=world,
        )

        self._step += 1

        metrics = {
            "step": self._step,
            "lr_dense": self.config.lr_dense,
            "lr_muon": self.config.lr_muon,
            "lr_expert": self.config.lr_expert,
        }

        return ImmediateTrainFuture(metrics)

    def get_weights(self) -> TrainFuture[dict[str, Any]]:
        """Get model weights for syncing to inference.

        Returns:
            Future resolving to state_dict
        """
        state_dict = self.model.state_dict()
        return ImmediateTrainFuture(state_dict)

    def load_weights(self, weights: dict[str, Any]) -> TrainFuture[None]:
        """Load model weights.

        Args:
            weights: state_dict to load
        """
        self.model.load_state_dict(weights)
        return ImmediateTrainFuture(None)

    def save_checkpoint(self, step: int) -> TrainFuture[Path]:
        """Save checkpoint to disk.

        Args:
            step: Current training step

        Returns:
            Future resolving to checkpoint path
        """
        try:
            from nmoe.checkpoint import save_checkpoint
        except ImportError:
            # Fallback to simple torch.save
            import torch

            ckpt_path = self.checkpoint_dir / f"step_{step}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), ckpt_path / "model.pt")
            return ImmediateTrainFuture(ckpt_path)

        ckpt_path = self.checkpoint_dir / f"step_{step}"
        save_checkpoint(self.model, ckpt_path, step=step)
        return ImmediateTrainFuture(ckpt_path)
