"""CuTe/CUTLASS implementation of NVFP4 block-scaled GEMV.

Based on gpu-mode/reference-kernels template_cute.py
Adapted to work with our backend system.

To use this:
1. Install CUTLASS: pip install cutlass
2. Uncomment the code below
3. Rename to cute_kernel.py
"""
from kernel_utils.backends import BACKENDS
from kernel_utils.task import input_t, output_t

# Uncomment when CUTLASS is installed:
"""
import cutlass
from cutlass.backend import cute
from cutlass.backend.blockscaled_utils import blockscaled_utils

# Configuration
mma_tiler_mnk = (128, 1, 64)
ab_dtype = cutlass.Float4E2M1FN
sf_dtype = cutlass.Float8E4M3FN
c_dtype = cutlass.Float16
sf_vec_size = 16
threads_per_cta = 128


@cutlass.kernel
def kernel(
    a_tensor: cute.TensorView,
    b_tensor: cute.TensorView,
    c_tensor: cute.TensorView,
    sfa_tensor: cute.TensorView,
    sfb_tensor: cute.TensorView,
    M: int,
    K: int,
    L: int,
):
    # Get thread coordinates
    tidx = cutlass.ThreadIdx.x
    cta_m = cutlass.BlockIdx.x * mma_tiler_mnk[0]
    cta_l = cutlass.BlockIdx.z

    # Bounds check
    if cta_m + tidx >= M:
        return

    # Get local tiles
    a_tile = cute.local_tile(a_tensor, (mma_tiler_mnk[0], K, 1), (cta_m, 0, cta_l))
    b_tile = cute.local_tile(b_tensor, (1, K, 1), (0, 0, cta_l))
    sfa_tile = cute.local_tile(sfa_tensor, ((32, 4), (K // sf_vec_size, 4), 1), ((0, 0), (0, 0), cta_l))
    sfb_tile = cute.local_tile(sfb_tensor, ((32, 4), (K // sf_vec_size, 4), 1), ((0, 0), (0, 0), cta_l))
    c_tile = cute.local_tile(c_tensor, (mma_tiler_mnk[0], 1, 1), (cta_m, 0, cta_l))

    # Accumulator
    result = cutlass.Float32(0.0)

    # Reduction loop over K dimension
    for k_idx in range(0, K, sf_vec_size):
        k_block = k_idx // sf_vec_size

        for k_offset in range(sf_vec_size):
            if k_idx + k_offset >= K:
                break

            # Load values
            a_val_nvfp4 = a_tile[tidx, k_idx + k_offset, 0]
            b_val_nvfp4 = b_tile[0, k_idx + k_offset, 0]

            # Load scales
            sfa_val_fp8 = sfa_tile[tidx % 32, tidx // 32, k_block, k_offset // 4, 0]
            sfb_val_fp8 = sfb_tile[0, 0, k_block, k_offset // 4, 0]

            # Convert to FP32 and accumulate
            a_val = a_val_nvfp4.to(cutlass.Float32)
            b_val = b_val_nvfp4.to(cutlass.Float32)
            sfa_val = sfa_val_fp8.to(cutlass.Float32)
            sfb_val = sfb_val_fp8.to(cutlass.Float32)

            result += a_val * sfa_val * b_val * sfb_val

    # Store result
    c_tile[tidx, 0, 0] = result.to(cutlass.Float16)


@cutlass.jit
def my_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sfa_ptr,
    sfb_ptr,
    M,
    K,
    L,
    stride_am,
    stride_ak,
    stride_al,
    stride_bk,
    stride_bl,
    stride_cm,
    stride_cl,
):
    # Create CuTe tensors
    a_layout = cute.Layout((M, K, L), (stride_am, stride_ak, stride_al))
    a_tensor = cute.TensorView(a_ptr, a_layout, ab_dtype)

    b_layout = cute.Layout((1, K, L), (0, stride_bk, stride_bl))
    b_tensor = cute.TensorView(b_ptr, b_layout, ab_dtype)

    c_layout = cute.Layout((M, 1, L), (stride_cm, 0, stride_cl))
    c_tensor = cute.TensorView(c_ptr, c_layout, c_dtype)

    # Scale factor layouts
    sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
        (M, K, L), sf_vec_size
    )
    sfa_tensor = cute.TensorView(sfa_ptr, sfa_layout, sf_dtype)

    sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
        (1, K, L), sf_vec_size
    )
    sfb_tensor = cute.TensorView(sfb_ptr, sfb_layout, sf_dtype)

    # Launch kernel
    grid_dims = (
        (M + mma_tiler_mnk[0] - 1) // mma_tiler_mnk[0],
        1,
        L
    )
    block_dims = (threads_per_cta, 1, 1)

    kernel(
        a_tensor, b_tensor, c_tensor, sfa_tensor, sfb_tensor,
        M, K, L,
        grid=grid_dims,
        block=block_dims
    )


# Kernel cache
_compiled_kernel_cache = None

def compile_kernel():
    global _compiled_kernel_cache
    if _compiled_kernel_cache is None:
        _compiled_kernel_cache = my_kernel
    return _compiled_kernel_cache


def custom_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    M, K, L = a.shape

    compiled = compile_kernel()

    compiled(
        a.data_ptr(),
        b.data_ptr(),
        c.data_ptr(),
        sfa.data_ptr(),
        sfb.data_ptr(),
        M, K, L,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(1), b.stride(2),
        c.stride(0), c.stride(2),
    )

    return c
"""


def cute_kernel(data: input_t) -> output_t:
    """CuTe/CUTLASS implementation of NVFP4 block-scaled GEMV.

    This is a template showing how to integrate CuTe kernels.
    See cute_kernel_template.py for the full implementation.

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

    # TODO: Uncomment when CUTLASS is installed
    # return custom_kernel(a, b, scale_a_perm, scale_b_perm, c)

    # For now, fall back to reference
    from nvfp4.reference_kernel import ref_kernel
    return ref_kernel(data)


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
    print("\n⚠️  Template only - uncomment CuTe code to use actual implementation")
