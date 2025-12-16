import torch
import triton
import triton.language as tl


@triton.jit
def _adaptive_avg_pool2d__default_triton_kernel(
    input_ptr,
    output_ptr,
    input_H,
    input_W,
    output_H,
    output_W,
    stride_H,
    stride_W,
    kernel_H,
    kernel_W,
    N,
    C,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    # pid indexes over N * (C // BLOCK_C) * (output_H * output_W // BLOCK_HW)
    # Let's decompose pid into n, c_block, hw_block
    n = pid // ((C // BLOCK_C) * ((output_H * output_W + BLOCK_HW - 1) // BLOCK_HW))
    rem = pid % ((C // BLOCK_C) * ((output_H * output_W + BLOCK_HW - 1) // BLOCK_HW))
    c_block = rem // ((output_H * output_W + BLOCK_HW - 1) // BLOCK_HW)
    hw_block = rem % ((output_H * output_W + BLOCK_HW - 1) // BLOCK_HW)

    c_start = c_block * BLOCK_C
    hw_start = hw_block * BLOCK_HW

    offs_c = c_start + tl.arange(0, BLOCK_C)
    offs_hw = hw_start + tl.arange(0, BLOCK_HW)

    # Compute output h and w from offs_hw
    offs_h = offs_hw // output_W
    offs_w = offs_hw % output_W

    # Mask for valid output pixels and channels
    mask_hw = offs_hw < (output_H * output_W)
    mask_c = offs_c < C
    mask = mask_hw[:, None] & mask_c[None, :]

    # Compute input start and end indices for pooling window
    in_h_start = offs_h * stride_H
    in_w_start = offs_w * stride_W

    # We will sum over kernel_H * kernel_W elements
    # For each output pixel, sum over the corresponding input window

    # Initialize accumulator
    acc = tl.zeros((BLOCK_HW, BLOCK_C), dtype=tl.float32)

    # input layout: (N, C, H, W)
    # input_ptr points to input[0,0,0,0]
    # strides:
    stride_n = C * input_H * input_W
    stride_c = input_H * input_W
    stride_h = input_W
    stride_w = 1

    for kh in range(kernel_H):
        for kw in range(kernel_W):
            in_h = in_h_start + kh
            in_w = in_w_start + kw

            mask_in = (in_h < input_H) & (in_w < input_W) & mask_hw[:, None] & mask_c[None, :]

            # Compute input offsets
            offs_n = n
            offs_c_ = offs_c
            offs_h_ = in_h
            offs_w_ = in_w

            # Broadcast to shape (BLOCK_HW, BLOCK_C)
            offs_n = offs_n * stride_n
            offs_c_ = offs_c_ * stride_c
            offs_h_ = offs_h_ * stride_h
            offs_w_ = offs_w_ * stride_w

            # offs_c_ shape (BLOCK_C,), offs_h_, offs_w_ shape (BLOCK_HW,)
            # We want to compute input_ptr + offs_n + offs_c_ + offs_h_ + offs_w_ for each (hw, c)
            # So we broadcast:
            offs_c_b = tl.broadcast_to(offs_c_, (BLOCK_HW, BLOCK_C))
            offs_h_b = tl.broadcast_to(offs_h_.unsqueeze(1), (BLOCK_HW, BLOCK_C))
            offs_w_b = tl.broadcast_to(offs_w_.unsqueeze(1), (BLOCK_HW, BLOCK_C))

            input_offsets = offs_n + offs_c_b + offs_h_b + offs_w_b

            vals = tl.load(input_ptr + input_offsets, mask=mask_in, other=0.0)
            acc += vals

    # Compute average
    kernel_size = kernel_H * kernel_W
    acc = acc / kernel_size

    # Write output
    # output layout: (N, C, output_H, output_W)
    stride_n_out = C * output_H * output_W
    stride_c_out = output_H * output_W
    stride_h_out = output_W
    stride_w_out = 1

    offs_n_out = n * stride_n_out
    offs_c_out = offs_c * stride_c_out
    offs_h_out = offs_h * stride_h_out
    offs_w_out = offs_w * stride_w_out

    offs_c_out_b = tl.broadcast_to(offs_c, (BLOCK_HW, BLOCK_C))
    offs_h_out_b = tl.broadcast_to(offs_h.unsqueeze(1), (BLOCK_HW, BLOCK_C))
    offs_w_out_b = tl.broadcast_to(offs_w.unsqueeze(1), (BLOCK_HW, BLOCK_C))

    output_offsets = offs_n_out + offs_c_out_b + offs_h_out_b + offs_w_out_b

    tl.store(output_ptr + output_offsets, acc, mask=mask)


def _adaptive_avg_pool2d__default_kernel_impl(*args, **kwargs) -> torch.Tensor:
    # Extract input tensor from args or kwargs
    # _adaptive_avg_pool2d.default(input, output_size)
    # output_size can be int or tuple (H_out, W_out)
    # We assume input is first positional argument or in kwargs with key 'input'
    if len(args) > 0:
        input = args[0]
        other_args = args[1:]
    elif "input" in kwargs:
        input = kwargs["input"]
        other_args = ()
    else:
        raise ValueError(
            "Input tensor must be provided as first positional argument or as 'input' keyword argument"
        )

    if len(other_args) > 0:
        output_size = other_args[0]
    elif "output_size" in kwargs:
        output_size = kwargs["output_size"]
    else:
        raise ValueError(
            "output_size must be provided as second positional argument or as 'output_size' keyword argument"
        )

    # Validate input dims
    if input.dim() != 4:
        raise ValueError(f"Input must be 4D tensor, got {input.dim()}D")

    # Determine output size
    if isinstance(output_size, int):
        output_H = output_W = output_size
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 2:
        output_H, output_W = output_size
    else:
        raise ValueError(f"output_size must be int or tuple of two ints, got {output_size}")

    # Move input to CUDA if needed
    input_device = input.device
    if input_device.type == "cpu":
        if torch.cuda.is_available():
            input_cuda = input.cuda()
        else:
            raise RuntimeError(
                "CUDA is not available but input tensor is on CPU. Cannot run Triton kernel."
            )
    elif input_device.type == "cuda":
        input_cuda = input
    else:
        raise RuntimeError(f"Unsupported device type {input_device.type}")

    N, C, input_H, input_W = input_cuda.shape

    # Compute stride and kernel size for adaptive avg pool
    # stride = floor(input_size / output_size)
    # kernel_size = input_size - (output_size - 1) * stride
    stride_H = input_H // output_H
    stride_W = input_W // output_W
    kernel_H = input_H - (output_H - 1) * stride_H
    kernel_W = input_W - (output_W - 1) * stride_W

    # Allocate output tensor on CUDA
    output_cuda = torch.empty(
        (N, C, output_H, output_W), device=input_cuda.device, dtype=input_cuda.dtype
    )

    # Launch Triton kernel
    BLOCK_C = 32
    BLOCK_HW = 64

    grid_n = N
    grid_c = (C + BLOCK_C - 1) // BLOCK_C
    grid_hw = (output_H * output_W + BLOCK_HW - 1) // BLOCK_HW
    grid = (grid_n * grid_c * grid_hw,)

    _adaptive_avg_pool2d__default_triton_kernel[grid](
        input_cuda.data_ptr(),
        output_cuda.data_ptr(),
        input_H,
        input_W,
        output_H,
        output_W,
        stride_H,
        stride_W,
        kernel_H,
        kernel_W,
        N,
        C,
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
        num_stages=2,
    )

    # Move output back to original device if needed
    if input_device.type == "cpu":
        output = output_cuda.cpu()
    else:
        output = output_cuda

    return output
