import torch
import triton
import triton.language as tl


@triton.jit
def _log_softmax__default_triton_kernel(
    X_ptr, Y_ptr, M, N, stride_m_x, stride_n_x, stride_m_y, stride_n_y, BLOCK_SIZE: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_SIZE
    n_start = pid_n * BLOCK_SIZE

    offs_m = m_start + tl.arange(0, BLOCK_SIZE)
    offs_n = n_start + tl.arange(0, BLOCK_SIZE)

    # Mask for valid elements in M dimension
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Load block of X
    # We want to load a BLOCK_SIZE x BLOCK_SIZE tile from X
    # X shape: (M, N)
    # Compute offsets for load
    # Use broadcasting to get a (BLOCK_SIZE, BLOCK_SIZE) matrix of indices
    offs_m_2d = offs_m[:, None]  # (BLOCK_SIZE, 1)
    offs_n_2d = offs_n[None, :]  # (1, BLOCK_SIZE)

    # Compute pointers for load
    ptrs = X_ptr + offs_m_2d * stride_m_x + offs_n_2d * stride_n_x

    # Load with mask
    x = tl.load(ptrs, mask=(mask_m[:, None] & mask_n[None, :]), other=-float("inf"))

    # Compute max per row (m dimension)
    # max over n dimension (axis=1)
    max_per_row = tl.max(x, axis=1)

    # Broadcast max_per_row to subtract
    max_per_row_broadcast = max_per_row[:, None]

    # Compute exp(x - max)
    exp_x = tl.exp(x - max_per_row_broadcast)

    # Compute sum exp per row
    sum_exp = tl.sum(exp_x, axis=1)

    # Compute log sum exp
    log_sum_exp = tl.log(sum_exp)

    # Compute log_softmax = x - max - log_sum_exp
    log_softmax = x - max_per_row_broadcast - log_sum_exp[:, None]

    # Store result
    tl.store(
        Y_ptr + offs_m_2d * stride_m_y + offs_n_2d * stride_n_y,
        log_softmax,
        mask=(mask_m[:, None] & mask_n[None, :]),
    )


def _log_softmax__default_kernel_impl(*args, **kwargs):
    # Expect input tensor as first positional argument or in kwargs with key 'input'
    # Support signature: _log_softmax.default(input, dim=-1, dtype=None, ...)
    # We will only implement dim support here, dtype ignored for simplicity

    # Extract input tensor
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

    # Extract dim argument
    dim = kwargs.get("dim", -1)
    if dim < 0:
        dim = input.dim() + dim
    if dim < 0 or dim >= input.dim():
        raise ValueError(f"Invalid dim argument {dim} for input with {input.dim()} dims")

    # Move input to GPU if needed
    input_device = input.device
    if input_device.type == "cpu":
        if torch.cuda.is_available():
            input_cuda = input.cuda()
        else:
            raise RuntimeError(
                "CUDA is not available but input tensor is on CPU and kernel requires GPU"
            )
    elif input_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but input tensor is on CUDA device")
        input_cuda = input
    else:
        raise RuntimeError(f"Unsupported device type {input_device.type}")

    # Permute input so that dim is last dimension
    # This simplifies kernel implementation (2D: M x N)
    permute_dims = [d for d in range(input_cuda.dim()) if d != dim] + [dim]
    input_perm = input_cuda.permute(*permute_dims)
    shape = input_perm.shape
    M = 1
    for i in range(len(shape) - 1):
        M *= shape[i]
    N = shape[-1]

    input_2d = input_perm.reshape(M, N)

    # Prepare output tensor
    output_2d = torch.empty_like(input_2d)

    # Get strides in elements (not bytes)
    stride_m_x = input_2d.stride(0)
    stride_n_x = input_2d.stride(1)
    stride_m_y = output_2d.stride(0)
    stride_n_y = output_2d.stride(1)

    # Define block size
    BLOCK_SIZE = 128

    grid_m = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_n = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch kernel
    _log_softmax__default_triton_kernel[grid_m, grid_n](
        input_2d.data_ptr(),
        output_2d.data_ptr(),
        M,
        N,
        stride_m_x,
        stride_n_x,
        stride_m_y,
        stride_n_y,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )

    # Reshape output back to original permuted shape
    output_perm = output_2d.reshape(shape)

    # Inverse permute to original shape
    inv_permute_dims = [0] * len(permute_dims)
    for i, d in enumerate(permute_dims):
        inv_permute_dims[d] = i
    output = output_perm.permute(*inv_permute_dims)

    # Move output back to original device if needed
    if input_device.type == "cpu":
        output = output.cpu()

    return output
