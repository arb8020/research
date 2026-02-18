"""Muon + AdamW hybrid optimizer for functional weights.

Muon: Momentum Orthogonalized by Newton-Schulz (Polar Express variant)
- Used for 2D weight matrices (attention projections, MLP weights)
- Much more efficient than AdamW for transformer training

AdamW: Standard AdamW
- Used for embeddings, norms, biases, 1D params

Based on:
- nanochat: https://github.com/karpathy/nanochat
- nmoe: Moonlight recipe (arXiv:2502.16982)
- Polar Express: https://arxiv.org/pdf/2505.16932
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

# Polar Express coefficients (5 Newton-Schulz iterations)
# From https://arxiv.org/pdf/2505.16932
POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def polar_express(g: Tensor, ns_steps: int = 5) -> Tensor:
    """Polar Express orthogonalization via Newton-Schulz iteration.

    Approximates the polar decomposition X = U @ S @ V.T -> returns U @ V.T
    (the nearest orthogonal matrix to the input).

    Args:
        g: Input gradient tensor of shape (M, N)
        ns_steps: Number of Newton-Schulz iterations (default 5)

    Returns:
        Orthogonalized tensor of same shape
    """
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)

    if g.size(-2) > g.size(-1):  # Tall matrix
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:  # Wide matrix
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X

    return X


class Muon(torch.optim.Optimizer):
    """Muon optimizer for 2D weight matrices.

    Moonlight recipe (arXiv:2502.16982):
    - SGD-Nesterov momentum base
    - Polar Express orthogonalization (Newton-Schulz, 5 steps)
    - Per-matrix update scaling: update_rms * sqrt(max(M, N))
    - Standard decoupled weight decay (AdamW-style)

    Only use for 2D weight tensors (attention projections, MLP weights).
    Do NOT use for embeddings, norms, biases, or 1D params.
    """

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        update_rms: float = 1.0,
        ns_steps: int = 5,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            update_rms=update_rms,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: None = None) -> None:
        if closure is not None:
            raise RuntimeError("Muon does not support closure")

        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            wd = float(group["weight_decay"])
            update_rms = float(group["update_rms"])
            ns_steps = int(group["ns_steps"])

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.dim() != 2:
                    raise RuntimeError(f"Muon only supports 2D tensors, got {grad.dim()}D")

                state = self.state[p]
                M, N = grad.shape

                # Initialize momentum buffer
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                mom_buf = state["momentum_buffer"]

                # 1) Nesterov momentum
                mom_buf.lerp_(grad, 1 - momentum)
                update = grad.lerp(mom_buf, momentum).contiguous()

                # 2) Polar Express orthogonalization
                update = polar_express(update, ns_steps)

                # 3) Moonlight scaling: update_rms * sqrt(max(M, N))
                if update_rms != 1.0:
                    update = update * update_rms
                update = update * math.sqrt(float(max(M, N)))

                # 4) Decoupled weight decay (AdamW-style)
                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)

                # 5) Apply update
                if p.dtype != update.dtype:
                    update = update.to(p.dtype)
                p.sub_(update, alpha=lr)


def classify_params(
    weights: dict[str, Tensor],
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """Classify functional weights into Muon vs AdamW groups.

    Args:
        weights: Dict of parameter name -> tensor

    Returns:
        (muon_params, adamw_decay_params, adamw_no_decay_params)
    """
    muon_params: list[Tensor] = []
    adamw_decay: list[Tensor] = []
    adamw_no_decay: list[Tensor] = []

    for name, param in weights.items():
        if not param.requires_grad:
            continue

        # Check if this is a 2D weight eligible for Muon
        is_2d_weight = param.dim() == 2 and param.numel() > 1024

        # These always go to AdamW (even if 2D)
        is_adamw_only = "embed" in name or "lm_head" in name or "norm" in name or "bias" in name

        if is_2d_weight and not is_adamw_only:
            muon_params.append(param)
        elif is_adamw_only or param.dim() < 2:
            # No weight decay for embeddings, norms, biases
            adamw_no_decay.append(param)
        else:
            adamw_decay.append(param)

    return muon_params, adamw_decay, adamw_no_decay


def build_optimizers(
    weights: dict[str, Tensor],
    lr_muon: float = 0.02,
    lr_adamw: float = 3e-4,
    momentum: float = 0.95,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    use_muon: bool = True,
) -> tuple[Muon | None, torch.optim.AdamW]:
    """Build Muon + AdamW optimizers for functional weights.

    Args:
        weights: Dict of parameter name -> tensor
        lr_muon: Learning rate for Muon (2D matrices)
        lr_adamw: Learning rate for AdamW (embeddings, norms, etc.)
        momentum: Muon momentum (default 0.95)
        weight_decay: Weight decay for both optimizers
        betas: AdamW betas
        eps: AdamW epsilon
        use_muon: If False, use AdamW for everything

    Returns:
        (muon_optimizer, adamw_optimizer)
        muon_optimizer is None if use_muon=False or no 2D params
    """
    muon_params, adamw_decay, adamw_no_decay = classify_params(weights)

    # Build AdamW with param groups
    adamw_groups = []
    if adamw_decay:
        adamw_groups.append({
            "params": adamw_decay,
            "weight_decay": weight_decay,
        })
    if adamw_no_decay:
        adamw_groups.append({
            "params": adamw_no_decay,
            "weight_decay": 0.0,
        })

    # If not using Muon, put 2D params in AdamW too
    if not use_muon and muon_params:
        adamw_groups.append({
            "params": muon_params,
            "weight_decay": weight_decay,
        })
        muon_params = []

    adamw_optimizer = torch.optim.AdamW(
        adamw_groups,
        lr=lr_adamw,
        betas=betas,
        eps=eps,
    )

    # Build Muon if we have 2D params
    muon_optimizer = None
    if muon_params:
        muon_optimizer = Muon(
            muon_params,
            lr=lr_muon,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    return muon_optimizer, adamw_optimizer
