"""KernelBench prompt templates.

System and user prompts for kernel optimization task.
"""

from __future__ import annotations

# System prompt instructs the model on the task format
SYSTEM_PROMPT = """\
You are a GPU kernel optimization expert. Your task is to write optimized {backend} kernels.

## Task Format
Given a PyTorch Model class, write an optimized ModelNew class that:
1. Has the same __init__ signature as Model
2. Has a forward() method with the same input/output signature
3. Uses custom {backend} kernels via torch.utils.cpp_extension.load_inline
4. Achieves speedup > 1.0x over the PyTorch baseline

## Requirements
- Your kernel MUST use __global__ kernel definitions (not PyTorch ops)
- Your kernel MUST produce outputs matching the reference within tolerance
- Include all necessary imports in your response

## Output Format
Put your complete implementation in <kernel> tags:

<kernel>
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Your custom kernel source
kernel_source = '''
__global__ void my_kernel(...) {{
    // Your CUDA/HIP kernel code
}}
'''

# Load the kernel
my_module = load_inline(
    name='my_kernel',
    cpp_sources=[''],
    cuda_sources=[kernel_source],
    functions=['my_kernel_wrapper'],
)

class ModelNew(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Same signature as Model

    def forward(self, ...):
        # Use your custom kernel
        return my_module.my_kernel_wrapper(...)
</kernel>
"""

# User prompt provides the specific problem
USER_PROMPT = """\
Optimize this PyTorch kernel for {backend}:

**Problem**: {name}
**Level**: {level}

```python
{ref_arch_src}
```

Write an optimized ModelNew class with custom {backend} kernels.
The reference file contains `get_inputs()` and `get_init_inputs()` for testing.

Your implementation must:
1. Pass correctness tests (outputs match within tolerance)
2. Achieve speedup > 1.0x over the PyTorch baseline
3. Use actual {backend} kernels with __global__ definitions
"""


def format_system_prompt(backend: str = "CUDA") -> str:
    """Format system prompt with backend."""
    return SYSTEM_PROMPT.format(backend=backend)


def format_user_prompt(
    name: str,
    level: str,
    ref_arch_src: str,
    backend: str = "CUDA",
) -> str:
    """Format user prompt with problem details."""
    return USER_PROMPT.format(
        name=name,
        level=level,
        ref_arch_src=ref_arch_src,
        backend=backend,
    )
