"""CuTe/CUTLASS implementation of NVFP4 block-scaled GEMV.

This is a template - replace with your actual CuTe CUDA extension.
"""
import torch
from kernel_utils.task import input_t, output_t
from kernel_utils.backends import BACKENDS


def cute_kernel(data: input_t) -> output_t:
    """CuTe/CUTLASS implementation of NVFP4 block-scaled GEMV.

    Args:
        data: Input tuple containing:
            - a: [m, k, l] float4_e2m1fn_x2
            - b: [1, k, l] float4_e2m1fn_x2
            - scale_a: [m, k, l] float8_e4m3fn (CPU)
            - scale_b: [1, k, l] float8_e4m3fn (CPU)
            - scale_a_permuted: GPU version (CuTe needs this format)
            - scale_b_permuted: GPU version (CuTe needs this format)
            - c: [m, 1, l] float16 (output buffer)

    Returns:
        Modified c tensor with GEMV result
    """
    a, b, scale_a, scale_b, scale_a_perm, scale_b_perm, c = data

    # TODO: Replace this with your CuTe kernel implementation
    # For now, fall back to reference for testing the infrastructure
    from nvfp4.reference_kernel import ref_kernel
    return ref_kernel(data)

    # Example of how to call your CUDA extension (uncomment when implemented):
    #
    # # Option 1: Load compiled CUDA extension
    # import nvfp4_cuda_extension
    # nvfp4_cuda_extension.gemv_forward(
    #     a, b, scale_a_perm, scale_b_perm, c
    # )
    # return c
    #
    # # Option 2: If using torch.ops
    # torch.ops.nvfp4.gemv(a, b, scale_a_perm, scale_b_perm, c)
    # return c


# Auto-register this backend when file is imported
BACKENDS.register(
    name="cute",
    kernel_fn=cute_kernel,
    description="CuTe/CUTLASS FP4 tensor core kernel (template - using reference for now)",
    language="cutlass",
)


if __name__ == "__main__":
    print("✅ CuTe backend registered")
    print(f"   Available backends: {BACKENDS.list()}")
