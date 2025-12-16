import torch
import triton
import triton.language as tl


# Triton kernel for _adaptive_avg_pool2d_backward.default
@triton.jit
def _adaptive_avg_pool2d_backward__default_triton_kernel(
    grad_output_ptr,
    grad_input_ptr,
    grad_output_H,
    grad_output_W,
    grad_input_H,
    grad_input_W,
    stride_H,
    stride_W,
    kernel_H,
    kernel_W,
    n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Each program handles one channel of one batch element
    # pid indexes over batch * channels
    batch = pid // n_channels
    channel = pid % n_channels

    # Pointers to grad_output and grad_input for this batch and channel
    grad_output_ptr = (
        grad_output_ptr
        + batch * n_channels * grad_output_H * grad_output_W
        + channel * grad_output_H * grad_output_W
    )
    grad_input_ptr = (
        grad_input_ptr
        + batch * n_channels * grad_input_H * grad_input_W
        + channel * grad_input_H * grad_input_W
    )

    # Iterate over grad_input pixels in blocks
    offs = tl.arange(0, BLOCK_SIZE)
    for h in range(0, grad_input_H, BLOCK_SIZE):
        for w in range(0, grad_input_W, BLOCK_SIZE):
            h_idx = h + offs
            w_idx = w + offs

            # Mask for valid indices
            mask_h = h_idx < grad_input_H
            mask_w = w_idx < grad_input_W

            # We want to accumulate grad_output values that correspond to this grad_input pixel
            # For adaptive avg pool backward, each grad_output pixel distributes grad_output / (kernel_H * kernel_W) to grad_input pixels in its receptive field

            # For each grad_input pixel, find which grad_output pixels it belongs to
            # The mapping is:
            # grad_output pixel (oh, ow) corresponds to grad_input pixels in:
            # h_start = oh * stride_H
            # w_start = ow * stride_W
            # h_start <= h_idx < h_start + kernel_H
            # w_start <= w_idx < w_start + kernel_W

            # So for each grad_input pixel (h_idx, w_idx), find all (oh, ow) such that:
            # oh * stride_H <= h_idx < oh * stride_H + kernel_H
            # ow * stride_W <= w_idx < ow * stride_W + kernel_W

            # This means:
            # oh in [ceil((h_idx - kernel_H + 1)/stride_H), floor(h_idx/stride_H)]
            # ow in [ceil((w_idx - kernel_W + 1)/stride_W), floor(w_idx/stride_W)]

            # Compute ranges of oh and ow for each h_idx, w_idx
            # Because BLOCK_SIZE is 1D, we will iterate over h_idx and w_idx separately

            # To vectorize, we process one dimension at a time

            # Here, we process BLOCK_SIZE elements in h and w separately, so we do a nested loop over BLOCK_SIZE for h and w

            # But to keep kernel simple and efficient, process one dimension at a time:
            # We'll process grad_input pixels in a 1D flattened manner: h_idx * grad_input_W + w_idx

            # Instead, we can process grad_input pixels in a 1D loop over BLOCK_SIZE * BLOCK_SIZE

            # Let's do that:

            # Flattened indices for grad_input pixels in this block
            # We will process BLOCK_SIZE * BLOCK_SIZE pixels per iteration

            # But Triton kernel has only one dimension of program_id, so we can do a 2D loop inside kernel

            # Let's rewrite kernel to process BLOCK_SIZE x BLOCK_SIZE pixels per program

            # To simplify, let's do a 2D grid with program_id(0) over batch*channels,
            # program_id(1) over h blocks,
            # program_id(2) over w blocks

            # So we need to rewrite kernel signature and grid accordingly

            # Let's do that now.

            pass  # placeholder for now


@triton.jit
def _adaptive_avg_pool2d_backward__default_triton_kernel(
    grad_output_ptr,
    grad_input_ptr,
    grad_output_H,
    grad_output_W,
    grad_input_H,
    grad_input_W,
    stride_H,
    stride_W,
    kernel_H,
    kernel_W,
    n_channels,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    batch_channel = tl.program_id(0)
    h_block = tl.program_id(1)
    w_block = tl.program_id(2)

    batch = batch_channel // n_channels
    channel = batch_channel % n_channels

    # Compute start indices for this block
    h_start = h_block * BLOCK_SIZE_H
    w_start = w_block * BLOCK_SIZE_W

    offs_h = tl.arange(0, BLOCK_SIZE_H)
    offs_w = tl.arange(0, BLOCK_SIZE_W)

    h_idx = h_start + offs_h  # shape [BLOCK_SIZE_H]
    w_idx = w_start + offs_w  # shape [BLOCK_SIZE_W]

    # Create 2D meshgrid of h_idx and w_idx
    h_idx_2d = tl.broadcast_to(h_idx[:, None], (BLOCK_SIZE_H, BLOCK_SIZE_W))
    w_idx_2d = tl.broadcast_to(w_idx[None, :], (BLOCK_SIZE_H, BLOCK_SIZE_W))

    # Mask for valid indices
    mask_h = h_idx_2d < grad_input_H
    mask_w = w_idx_2d < grad_input_W
    mask = mask_h & mask_w

    # Initialize accumulator for grad_input pixels
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # For each grad_input pixel (h_idx_2d, w_idx_2d), find grad_output pixels (oh, ow) that cover it
    # oh in [ceil((h_idx - kernel_H + 1)/stride_H), floor(h_idx/stride_H)]
    # ow in [ceil((w_idx - kernel_W + 1)/stride_W), floor(w_idx/stride_W)]

    # Compute oh_min and oh_max for each h_idx
    oh_min = (h_idx_2d - kernel_H + 1 + stride_H - 1) // stride_H  # ceil division
    oh_max = h_idx_2d // stride_H

    ow_min = (w_idx_2d - kernel_W + 1 + stride_W - 1) // stride_W
    ow_max = w_idx_2d // stride_W

    # Clamp to valid range
    oh_min = tl.maximum(oh_min, 0)
    ow_min = tl.maximum(ow_min, 0)
    oh_max = tl.minimum(oh_max, grad_output_H - 1)
    ow_max = tl.minimum(ow_max, grad_output_W - 1)

    # For each grad_input pixel, accumulate grad_output pixels in the range [oh_min, oh_max], [ow_min, ow_max]
    # Because ranges can be empty, we skip if oh_min > oh_max or ow_min > ow_max

    # We'll iterate over oh and ow ranges and accumulate grad_output / (kernel_H * kernel_W)

    # To vectorize, we iterate over max possible range and mask invalid

    max_oh_range = kernel_H  # max oh range length
    max_ow_range = kernel_W  # max ow range length

    for i in range(max_oh_range):
        oh = oh_min + i
        valid_oh = (oh <= oh_max) & (oh < grad_output_H)
        for j in range(max_ow_range):
            ow = ow_min + j
            valid_ow = (ow <= ow_max) & (ow < grad_output_W)
            valid = valid_oh & valid_ow & mask
            if not tl.any(valid):
                continue

            # Load grad_output at (batch, channel, oh, ow)
            grad_output_offset = (
                batch * n_channels * grad_output_H * grad_output_W
                + channel * grad_output_H * grad_output_W
                + oh * grad_output_W
                + ow
            )
            grad_output_val = tl.load(grad_output_ptr + grad_output_offset, mask=valid, other=0.0)

            # grad_output_val is scalar per (oh, ow), but we have multiple grad_input pixels
            # We need to broadcast grad_output_val to grad_input pixels where valid is True

            # Actually, grad_output_val is scalar per (oh, ow), but we have multiple grad_input pixels (h_idx_2d, w_idx_2d)
            # We need to accumulate grad_output_val / (kernel_H * kernel_W) to all grad_input pixels covered by (oh, ow)

            # But we are iterating over grad_input pixels, so we accumulate grad_output_val / (kernel_H * kernel_W) to acc[h_idx_2d, w_idx_2d] where valid is True

            # So for each grad_input pixel, add grad_output_val / (kernel_H * kernel_W) if valid

            # grad_output_val is scalar, but we have multiple grad_input pixels, so we broadcast grad_output_val to acc shape

            # Actually, grad_output_val is scalar per (oh, ow), but we are iterating over grad_input pixels, so grad_output_val is scalar, acc is 2D

            # So we add grad_output_val / (kernel_H * kernel_W) to acc where valid

            acc += tl.where(valid, grad_output_val / (kernel_H * kernel_W), 0.0)

    # Write acc to grad_input_ptr
    grad_input_offset = (
        batch * n_channels * grad_input_H * grad_input_W
        + channel * grad_input_H * grad_input_W
        + h_idx_2d * grad_input_W
        + w_idx_2d
    )
    tl.store(grad_input_ptr + grad_input_offset, acc, mask=mask)


def _adaptive_avg_pool2d_backward__default_kernel_impl(*args, **kwargs):
    # Expected args: grad_output, input, output_size (H_out, W_out)
    # grad_output: Tensor of shape (N, C, H_out, W_out)
    # input: Tensor of shape (N, C, H_in, W_in)
    # output_size: tuple (H_out, W_out)

    # Parse inputs
    if len(args) >= 2:
        grad_output = args[0]
        input = args[1]
    else:
        grad_output = kwargs.get("grad_output", None)
        input = kwargs.get("input", None)
    if grad_output is None or input is None:
        raise ValueError("grad_output and input tensors must be provided as args or kwargs")

    # Determine output size from grad_output shape
    N, C, H_out, W_out = grad_output.shape
    N_in, C_in, H_in, W_in = input.shape
    if N != N_in or C != C_in:
        raise ValueError("Batch size and channel count of grad_output and input must match")

    # Compute kernel size and stride for adaptive avg pool
    # kernel_H = floor(H_in / H_out)
    # kernel_W = floor(W_in / W_out)
    # stride_H = floor(H_in / H_out)
    # stride_W = floor(W_in / W_out)
    # This matches PyTorch adaptive avg pool backward logic

    kernel_H = H_in // H_out
    kernel_W = W_in // W_out
    stride_H = kernel_H
    stride_W = kernel_W

    # Device management
    orig_device = grad_output.device
    orig_input_device = input.device

    if grad_output.device.type == "cpu":
        if torch.cuda.is_available():
            grad_output = grad_output.cuda()
        else:
            raise RuntimeError("CUDA is not available but grad_output is on CPU")
    if input.device.type == "cpu":
        if torch.cuda.is_available():
            input = input.cuda()
        else:
            raise RuntimeError("CUDA is not available but input is on CPU")

    # Allocate grad_input tensor on same device as input
    grad_input = torch.zeros_like(input, device=input.device, dtype=grad_output.dtype)

    # Launch Triton kernel
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16

    grid = (
        N * C,
        (H_in + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (W_in + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,
    )

    _adaptive_avg_pool2d_backward__default_triton_kernel[grid](
        grad_output.data_ptr(),
        grad_input.data_ptr(),
        H_out,
        W_out,
        H_in,
        W_in,
        stride_H,
        stride_W,
        kernel_H,
        kernel_W,
        C,
        BLOCK_SIZE_H,
        BLOCK_SIZE_W,
    )

    # Move grad_input back to original device if needed
    if orig_input_device.type == "cpu" and grad_input.device.type == "cuda":
        grad_input = grad_input.cpu()

    return grad_input
