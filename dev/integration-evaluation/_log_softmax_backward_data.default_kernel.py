import torch
import triton
import triton.language as tl


@triton.jit
def _log_softmax_backward_data__default_triton_kernel(
    grad_output_ptr, output_ptr, grad_input_ptr,
    stride_go_batch, stride_go_feature,
    stride_o_batch, stride_o_feature,
    stride_gi_batch, stride_gi_feature,
    n_features,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_idx = pid

    offs = tl.arange(0, BLOCK_SIZE)
    feature_idx = offs

    mask = feature_idx < n_features

    go_ptrs = grad_output_ptr + batch_idx * stride_go_batch + feature_idx * stride_go_feature
    o_ptrs = output_ptr + batch_idx * stride_o_batch + feature_idx * stride_o_feature
    gi_ptrs = grad_input_ptr + batch_idx * stride_gi_batch + feature_idx * stride_gi_feature

    grad_output = tl.load(go_ptrs, mask=mask, other=0.0)
    output = tl.load(o_ptrs, mask=mask, other=0.0)

    # Compute sum of grad_output over features
    sum_grad_output = tl.sum(grad_output, axis=0)

    # Compute grad_input = grad_output - exp(output) * sum_grad_output
    exp_output = tl.exp(output)
    grad_input = grad_output - exp_output * sum_grad_output

    tl.store(gi_ptrs, grad_input, mask=mask)


def _log_softmax_backward_data__default_kernel_impl(*args, **kwargs):
    # Expecting grad_output, output tensors as inputs, output grad_input tensor returned
    # Signature: grad_input = _log_softmax_backward_data__default_kernel_impl(grad_output, output)
    # or with kwargs

    # Extract grad_output and output from args or kwargs
    if len(args) >= 2:
        grad_output = args[0]
        output = args[1]
    else:
        grad_output = kwargs.get('grad_output', None)
        output = kwargs.get('output', None)
        if grad_output is None or output is None:
            raise ValueError("grad_output and output tensors must be provided as positional or keyword arguments")

    # Check devices and move to CUDA if needed
    orig_device = grad_output.device
    orig_device_output = output.device

    if grad_output.device.type == 'cpu':
        if torch.cuda.is_available():
            grad_output = grad_output.cuda()
        else:
            raise RuntimeError("CUDA is not available but grad_output is on CPU")
    if output.device.type == 'cpu':
        if torch.cuda.is_available():
            output = output.cuda()
        else:
            raise RuntimeError("CUDA is not available but output is on CPU")

    if grad_output.device.type != 'cuda' or output.device.type != 'cuda':
        raise RuntimeError("grad_output and output must be on CUDA device")

    # Check shapes
    if grad_output.shape != output.shape:
        raise ValueError(f"grad_output and output must have the same shape, got {grad_output.shape} and {output.shape}")

    batch_size, n_features = grad_output.shape

    # Allocate grad_input tensor on CUDA
    grad_input = torch.empty_like(grad_output, device=grad_output.device)

    # Compute strides
    stride_go_batch, stride_go_feature = grad_output.stride()
    stride_o_batch, stride_o_feature = output.stride()
    stride_gi_batch, stride_gi_feature = grad_input.stride()

    BLOCK_SIZE = 128
    grid = (batch_size,)

    _log_softmax_backward_data__default_triton_kernel[grid](
        grad_output.data_ptr(), output.data_ptr(), grad_input.data_ptr(),
        stride_go_batch, stride_go_feature,
        stride_o_batch, stride_o_feature,
        stride_gi_batch, stride_gi_feature,
        n_features,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Move grad_input back to original device if needed
    if orig_device.type == 'cpu':
        grad_input = grad_input.cpu()

    return grad_input