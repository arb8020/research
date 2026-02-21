"""Linear layers for inference with tensor parallelism support.

Column-parallel: shard output dim, no sync needed
Row-parallel: shard input dim, all_reduce after forward

For single-GPU (tp.world_size=1), these behave like standard nn.Linear.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from ..tp import TPConfig


class Linear(nn.Module):
    """Standard linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.linear(x, self.weight, self.bias)


class ColumnParallelLinear(nn.Module):
    """Linear layer with column-parallel weight (output dim sharded).

    For tensor parallelism: weight is sharded along output dimension.
    No communication needed - each rank computes its subset of outputs.

    Used for: gate_proj, up_proj, q_proj, k_proj, v_proj
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Shard output dimension across TP ranks
        if tp is not None and tp.is_distributed:
            from ..tp import get_local_size

            self.tp = tp
            self.local_out_features = get_local_size(out_features, tp.world_size)
        else:
            self.tp = None
            self.local_out_features = out_features

        self.linear = Linear(in_features, self.local_out_features, bias, dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class RowParallelLinear(nn.Module):
    """Linear layer with row-parallel weight (input dim sharded).

    For tensor parallelism: weight is sharded along input dimension.
    Requires all_reduce after forward to sum partial outputs.

    Used for: down_proj, o_proj
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Shard input dimension across TP ranks
        if tp is not None and tp.is_distributed:
            from ..tp import get_local_size

            self.tp = tp
            self.local_in_features = get_local_size(in_features, tp.world_size)
        else:
            self.tp = None
            self.local_in_features = in_features

        self.linear = Linear(self.local_in_features, out_features, bias, dtype)

    def forward(self, x: Tensor) -> Tensor:
        y = self.linear(x)
        if self.tp is not None and self.tp.is_distributed:
            y = self.tp.all_reduce(y)
        return y


class QKVParallelLinear(nn.Module):
    """Merged Q, K, V projection with tensor parallelism.

    Projects input to concatenated [Q, K, V] outputs.
    Handles different head counts for GQA (grouped query attention).
    For TP: each rank gets subset of Q and KV heads.

    Output shape: [batch, seq, local_q_size + 2 * local_kv_size]
    """

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Shard heads across TP ranks
        if tp is not None and tp.is_distributed:
            from ..tp import get_local_size

            self.tp = tp
            self.local_num_q_heads = get_local_size(num_q_heads, tp.world_size)
            self.local_num_kv_heads = get_local_size(num_kv_heads, tp.world_size)
        else:
            self.tp = None
            self.local_num_q_heads = num_q_heads
            self.local_num_kv_heads = num_kv_heads

        self.q_size = self.local_num_q_heads * head_dim
        self.kv_size = self.local_num_kv_heads * head_dim
        total_size = self.q_size + 2 * self.kv_size

        self.linear = Linear(hidden_size, total_size, bias, dtype)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (q, k, v) tensors."""
        qkv = self.linear(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return q, k, v


class GateUpParallelLinear(nn.Module):
    """Merged gate and up projection for MLP with tensor parallelism.

    Projects input to concatenated [gate, up] outputs.
    For TP: intermediate_size is sharded across ranks.

    Output shape: [batch, seq, 2 * local_intermediate_size]
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Shard intermediate dimension across TP ranks
        if tp is not None and tp.is_distributed:
            from ..tp import get_local_size

            self.tp = tp
            self.local_intermediate_size = get_local_size(intermediate_size, tp.world_size)
        else:
            self.tp = None
            self.local_intermediate_size = intermediate_size

        self.linear = Linear(hidden_size, 2 * self.local_intermediate_size, bias, dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)
