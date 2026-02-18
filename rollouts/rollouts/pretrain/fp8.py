"""Minimal FP8 training for functional models — tensorwise dynamic scaling.

Adapted from nanochat's fp8.py for functional weight dicts instead of nn.Module.
Provides fp8_linear() as a drop-in replacement for F.linear when training on H100+.

How FP8 training works
======================
A standard Linear layer does one matmul in forward and two in backward:
  forward:      output     = input      @ weight.T
  backward:     grad_input = grad_output @ weight
                grad_weight= grad_output.T @ input

FP8 training wraps each matmul with dynamic quantization:
  1. Compute scale = FP8_MAX / max(|tensor|) for each operand
  2. Quantize: fp8_tensor = clamp(tensor * scale, -FP8_MAX, FP8_MAX).to(fp8)
  3. Matmul via torch._scaled_mm (cuBLAS FP8 kernel, ~2x faster than bf16)
  4. Dequantize: _scaled_mm handles this internally using the inverse scales

FP8 dtype choice
================
  - float8_e4m3fn: Higher precision (3-bit mantissa), range [-448, 448]
    Used for input and weight.
  - float8_e5m2: Wider range (5-bit exponent), range [-57344, 57344]
    Used for gradients which can be large.

Usage
=====
Replace F.linear calls with fp8_linear when use_fp8=True:

    if use_fp8:
        output = fp8_linear(input, weight)
    else:
        output = F.linear(input, weight)

Requirements: H100+ GPU with native FP8 tensor cores. Dimensions must be
divisible by 16.
"""

import torch

# Avoid division by zero when computing scale from an all-zeros tensor
EPS = 1e-12


def is_fp8_available() -> bool:
    """Check if FP8 training is available (H100+ with CUDA)."""
    if not torch.cuda.is_available():
        return False
    # FP8 requires sm_90 (Hopper) or newer
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 9


@torch.no_grad()
def _to_fp8(x: torch.Tensor, fp8_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize a tensor to FP8 using tensorwise scaling.

    Returns (fp8_data, inverse_scale) for use with torch._scaled_mm.
    """
    fp8_max = torch.finfo(fp8_dtype).max
    amax = x.float().abs().max()
    # Use float64 for division to ensure consistent numerics between compile/eager
    scale = fp8_max / amax.double().clamp(min=EPS)
    scale = scale.float()
    x_scaled = x.float() * scale
    x_clamped = x_scaled.clamp(-fp8_max, fp8_max)
    x_fp8 = x_clamped.to(fp8_dtype)
    inv_scale = scale.reciprocal()
    return x_fp8, inv_scale


def _to_col_major(x: torch.Tensor) -> torch.Tensor:
    """Rearrange a 2D tensor's memory to column-major layout.

    torch._scaled_mm requires its second operand in column-major layout.
    """
    return x.t().contiguous().t()


@torch._dynamo.allow_in_graph
class _Float8Matmul(torch.autograd.Function):
    """Custom autograd for the three FP8 GEMMs of a Linear layer."""

    @staticmethod
    def forward(ctx, input_2d: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input_2d, weight)

        # Quantize both operands to e4m3 (higher precision format)
        input_fp8, input_inv = _to_fp8(input_2d, torch.float8_e4m3fn)
        weight_fp8, weight_inv = _to_fp8(weight, torch.float8_e4m3fn)

        # output = input @ weight.T
        output = torch._scaled_mm(
            input_fp8,
            weight_fp8.t(),
            scale_a=input_inv,
            scale_b=weight_inv,
            out_dtype=input_2d.dtype,
            use_fast_accum=True,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_2d, weight = ctx.saved_tensors

        # GEMM 1: grad_input = grad_output @ weight
        go_fp8, go_inv = _to_fp8(grad_output, torch.float8_e5m2)
        w_fp8, w_inv = _to_fp8(weight, torch.float8_e4m3fn)
        w_col = _to_col_major(w_fp8)
        grad_input = torch._scaled_mm(
            go_fp8,
            w_col,
            scale_a=go_inv,
            scale_b=w_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )

        # GEMM 2: grad_weight = grad_output.T @ input
        go_fp8_2, go_inv_2 = _to_fp8(grad_output, torch.float8_e5m2)
        in_fp8, in_inv = _to_fp8(input_2d, torch.float8_e4m3fn)
        go_T = go_fp8_2.t().contiguous()
        in_col = _to_col_major(in_fp8)
        grad_weight = torch._scaled_mm(
            go_T,
            in_col,
            scale_a=go_inv_2,
            scale_b=in_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )

        return grad_input, grad_weight


def fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP8 linear layer - drop-in replacement for F.linear.

    Args:
        input: Input tensor of shape (..., in_features)
        weight: Weight tensor of shape (out_features, in_features)
        bias: Optional bias tensor of shape (out_features,)

    Returns:
        Output tensor of shape (..., out_features)

    Note: Dimensions must be divisible by 16 for H100 FP8 tensor cores.
    """
    # Handle autocast (match F.linear behavior)
    if torch.is_autocast_enabled():
        input = input.to(torch.get_autocast_gpu_dtype())

    # _scaled_mm only works on 2D tensors, so flatten batch dimensions
    orig_shape = input.shape
    input_2d = input.reshape(-1, orig_shape[-1])

    output = _Float8Matmul.apply(input_2d, weight)

    # Unflatten
    output = output.reshape(*orig_shape[:-1], output.shape[-1])

    if bias is not None:
        output = output + bias.to(output.dtype)

    return output


def can_use_fp8(in_features: int, out_features: int) -> bool:
    """Check if a linear layer can use FP8 (dims divisible by 16)."""
    return in_features % 16 == 0 and out_features % 16 == 0
