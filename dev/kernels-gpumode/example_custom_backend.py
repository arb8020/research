"""Example: How to register a custom backend.

This shows how you would register a Triton or CuTe kernel.
Copy this pattern when you write your optimized kernels.
"""

from kernel_utils.backends import BACKENDS
from kernel_utils.task import input_t, output_t


def my_triton_kernel(data: input_t) -> output_t:
    """Example Triton kernel (replace with your actual implementation).

    For now, this just calls the reference for demonstration.
    """
    # TODO: Replace with actual Triton kernel implementation
    from nvfp4_reference_kernel import ref_kernel

    return ref_kernel(data)


def my_cute_kernel(data: input_t) -> output_t:
    """Example CuTe kernel (replace with your actual implementation).

    For now, this just calls the reference for demonstration.
    """
    # TODO: Replace with actual CuTe/CUDA kernel implementation
    from nvfp4_reference_kernel import ref_kernel

    return ref_kernel(data)


# Register custom backends
# Just import this file to auto-register them
BACKENDS.register(
    name="triton",
    kernel_fn=my_triton_kernel,
    description="Triton-based FP4 GEMV kernel (example)",
    language="triton",
)

BACKENDS.register(
    name="cute",
    kernel_fn=my_cute_kernel,
    description="CuTe/CUTLASS FP4 tensor core kernel (example)",
    language="cutlass",
)

if __name__ == "__main__":
    print("✅ Registered custom backends:")
    for name in BACKENDS.list():
        backend = BACKENDS[name]
        print(f"  - {name}: {backend.description} ({backend.language})")
