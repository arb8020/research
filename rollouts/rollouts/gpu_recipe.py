"""GPU environment recipes.

A GPURecipe specifies how to set up a working PyTorch environment on a GPU.
It captures the coupling between:
- Base Docker image (has drivers, Python)
- PyTorch version and wheel index
- Environment variables (CUDA_HOME, ROCM_HOME, etc.)

Usage:
    from rollouts.gpu_recipe import GPURecipe, CUDA_124, ROCM_61

    # Use a preset
    recipe = CUDA_124

    # Or define a custom recipe
    custom = GPURecipe(
        name="cuda-12.8-nightly",
        image="nvidia/cuda:12.9.0-devel-ubuntu22.04",
        torch_index_url="https://download.pytorch.org/whl/nightly/cu128",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GPURecipe:
    """Recipe for setting up a PyTorch environment on a GPU.

    All fields that must match for a working setup are grouped here.
    The recipe is declarative — providers (Modal, RunPod, SSH) interpret it.

    Attributes:
        name: Human-readable identifier (e.g., "cuda-12.4", "rocm-6.1")
        image: Docker image with drivers and Python (e.g., "nvidia/cuda:12.4.0-devel-ubuntu22.04")
        torch_index_url: PyTorch wheel index (e.g., "https://download.pytorch.org/whl/cu124")
        python_version: Python version to use (default: "3.12")
        system_packages: Apt packages to install (default: tmux, git, curl, build-essential)
        env: Environment variables to set (CUDA_HOME, etc.)
        gpu_arch: GPU architecture for JIT compilation (e.g., "sm_90" for H100, "gfx942" for MI300X)
    """

    name: str
    image: str
    torch_index_url: str
    python_version: str = "3.12"
    system_packages: tuple[str, ...] = ("tmux", "git", "curl", "build-essential", "libnuma1")
    env: dict[str, str] = field(default_factory=dict)
    gpu_arch: str | None = None

    def __post_init__(self) -> None:
        assert self.name, "name cannot be empty"
        assert self.image, "image cannot be empty"
        assert self.torch_index_url, "torch_index_url cannot be empty"
        assert self.python_version, "python_version cannot be empty"


# =============================================================================
# NVIDIA CUDA Recipes
# =============================================================================

# CUDA 12.4 — stable, works with H100, A100, A10G, L4, etc.
CUDA_124 = GPURecipe(
    name="cuda-12.4",
    image="nvidia/cuda:12.4.0-devel-ubuntu22.04",
    torch_index_url="https://download.pytorch.org/whl/cu124",
    env={
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
    },
    gpu_arch="sm_89",  # Ada Lovelace (L4, RTX 4090)
)

# CUDA 12.8 nightly — for Blackwell (B200, GB200)
CUDA_128_NIGHTLY = GPURecipe(
    name="cuda-12.8-nightly",
    image="nvidia/cuda:12.9.0-devel-ubuntu22.04",
    torch_index_url="https://download.pytorch.org/whl/nightly/cu128",
    env={
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
    },
    gpu_arch="sm_100",  # Blackwell
)


# =============================================================================
# AMD ROCm Recipes
# =============================================================================

# ROCm 6.1 — RunPod default, stable
ROCM_61 = GPURecipe(
    name="rocm-6.1",
    image="runpod/pytorch:2.4.0-py3.10-rocm6.1.0-ubuntu22.04",
    torch_index_url="https://download.pytorch.org/whl/rocm6.1",
    python_version="3.10",  # RunPod image uses 3.10
    env={
        "ROCM_HOME": "/opt/rocm",
        "HIP_PLATFORM": "amd",
        "PATH": "/opt/rocm/bin:$PATH",
        "LD_LIBRARY_PATH": "/opt/rocm/lib:$LD_LIBRARY_PATH",
    },
    gpu_arch="gfx942",  # MI300X
)

# ROCm 6.4 — DigitalOcean, newer
ROCM_64 = GPURecipe(
    name="rocm-6.4",
    image="rocm/pytorch:rocm6.4_ubuntu22.04_py3.10_pytorch_release_2.6.0",
    torch_index_url="https://download.pytorch.org/whl/rocm6.4",
    python_version="3.10",
    env={
        "ROCM_HOME": "/opt/rocm",
        "HIP_PLATFORM": "amd",
        "PATH": "/opt/rocm/bin:$PATH",
        "LD_LIBRARY_PATH": "/opt/rocm/lib:$LD_LIBRARY_PATH",
    },
    gpu_arch="gfx942",  # MI300X
)

# ROCm 7.2 — bleeding edge, hipBLASLt benchmarks
ROCM_72 = GPURecipe(
    name="rocm-7.2",
    image="rocm/dev-ubuntu-22.04:7.2-complete",
    torch_index_url="https://download.pytorch.org/whl/rocm6.4",  # No 7.2 wheels yet
    python_version="3.10",
    env={
        "ROCM_HOME": "/opt/rocm",
        "HIP_PLATFORM": "amd",
        "PATH": "/opt/rocm/bin:$PATH",
        "LD_LIBRARY_PATH": "/opt/rocm/lib:$LD_LIBRARY_PATH",
    },
    gpu_arch="gfx942",  # MI300X
)
