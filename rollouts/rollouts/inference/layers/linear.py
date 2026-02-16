"""Linear layers for inference.

These are simple wrappers around nn.Linear with explicit shapes.
Tensor parallelism support is deferred - currently single-GPU only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


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
    """Linear layer with column-parallel weight.

    For tensor parallelism: weight is sharded along output dimension.
    Currently just wraps Linear (no TP yet).

    Used for: gate_proj, up_proj, q_proj, k_proj, v_proj
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        # TODO: divide out_features by tp_size
        self.linear = Linear(in_features, out_features, bias, dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class RowParallelLinear(nn.Module):
    """Linear layer with row-parallel weight.

    For tensor parallelism: weight is sharded along input dimension.
    Currently just wraps Linear (no TP yet).

    Used for: down_proj, o_proj
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        # TODO: divide in_features by tp_size
        self.linear = Linear(in_features, out_features, bias, dtype)

    def forward(self, x: Tensor) -> Tensor:
        # TODO: all-reduce output for TP
        return self.linear(x)


class QKVParallelLinear(nn.Module):
    """Merged Q, K, V projection.

    Projects input to concatenated [Q, K, V] outputs.
    Handles different head counts for GQA (grouped query attention).

    Output shape: [batch, seq, q_size + 2 * kv_size]
    where q_size = num_q_heads * head_dim
          kv_size = num_kv_heads * head_dim
    """

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        q_size = num_q_heads * head_dim
        kv_size = num_kv_heads * head_dim
        total_size = q_size + 2 * kv_size

        self.linear = Linear(hidden_size, total_size, bias, dtype)

        # Store split sizes for forward
        self.q_size = q_size
        self.kv_size = kv_size

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (q, k, v) tensors."""
        qkv = self.linear(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return q, k, v


class GateUpParallelLinear(nn.Module):
    """Merged gate and up projection for MLP.

    Projects input to concatenated [gate, up] outputs.
    Output shape: [batch, seq, 2 * intermediate_size]
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.linear = Linear(hidden_size, 2 * intermediate_size, bias, dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)
