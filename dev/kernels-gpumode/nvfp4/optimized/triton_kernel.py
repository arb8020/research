"""Triton implementation of NVFP4 block-scaled GEMV.

This is a template - replace the implementation with your actual Triton kernel.
"""

from kernel_utils.backends import BACKENDS
from kernel_utils.task import input_t, output_t

# Uncomment when you implement the kernel
# import triton
# import triton.language as tl


# @triton.jit
# def nvfp4_gemv_triton_kernel(
#     # Pointers
#     a_ptr,
#     b_ptr,
#     c_ptr,
#     scale_a_ptr,
#     scale_b_ptr,
#     # Dimensions
#     M, K, L,
#     # Strides
#     stride_am, stride_ak, stride_al,
#     stride_bk, stride_bl,
#     stride_cm, stride_cl,
#     # Block sizes
#     BLOCK_M: tl.constexpr,
#     BLOCK_K: tl.constexpr,
# ):
#     """Triton kernel for NVFP4 GEMV.
#
#     TODO: Implement your Triton kernel here.
#     """
#     pass


def triton_kernel(data: input_t) -> output_t:
    """Triton implementation of NVFP4 block-scaled GEMV.

    Args:
        data: Input tuple containing:
            - a: [m, k, l] float4_e2m1fn_x2
            - b: [1, k, l] float4_e2m1fn_x2
            - scale_a: [m, k, l] float8_e4m3fn (CPU)
            - scale_b: [1, k, l] float8_e4m3fn (CPU)
            - scale_a_permuted: GPU version
            - scale_b_permuted: GPU version
            - c: [m, 1, l] float16 (output buffer)

    Returns:
        Modified c tensor with GEMV result
    """
    a, b, scale_a, scale_b, scale_a_perm, scale_b_perm, c = data

    # Get dimensions
    M, K, L = a.shape

    # TODO: Replace this with your Triton kernel implementation
    # For now, fall back to reference for testing the infrastructure
    from nvfp4.reference_kernel import ref_kernel

    return ref_kernel(data)

    # Example of how to launch Triton kernel (uncomment when implemented):
    # grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), L)
    # nvfp4_gemv_triton_kernel[grid](
    #     a, b, c,
    #     scale_a_perm, scale_b_perm,
    #     M, K, L,
    #     # Strides
    #     a.stride(0), a.stride(1), a.stride(2),
    #     b.stride(1), b.stride(2),
    #     c.stride(0), c.stride(2),
    #     # Block sizes
    #     BLOCK_M=64,
    #     BLOCK_K=64,
    # )
    # return c


# Auto-register this backend when file is imported
BACKENDS.register(
    name="triton",
    kernel_fn=triton_kernel,
    description="Triton FP4 GEMV kernel (template - using reference for now)",
    language="triton",
)


if __name__ == "__main__":
    print("✅ Triton backend registered")
    print(f"   Available backends: {BACKENDS.list()}")
