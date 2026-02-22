"""Neural network layers for inference.

Simple, explicit implementations following the codebase style:
- No global state
- Explicit dimensions in constructors
- Forward takes tensors, returns tensors
"""

from .activation import silu_and_mul
from .linear import ColumnParallelLinear, Linear, QKVParallelLinear, RowParallelLinear
from .moe import MoeExperts, MoeGate, MoeLayer, MoeMLP, is_fused_moe_available
from .norm import RMSNorm
from .rotary import RotaryEmbedding, apply_rotary_pos_emb

__all__ = [
    "silu_and_mul",
    "Linear",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "QKVParallelLinear",
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    # MoE layers
    "MoeGate",
    "MoeExperts",
    "MoeLayer",
    "MoeMLP",
    "is_fused_moe_available",
]
