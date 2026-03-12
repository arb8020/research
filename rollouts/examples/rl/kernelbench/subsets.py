"""Canonical KernelBench problem subsets."""

from __future__ import annotations

import os
from typing import Any

from .dataset import load_kernelbench_prompts, load_kernelbench_v3_prompts

# Mirrors the easy multi-turn smoke subset used in Wafer's KernelBench evals.
SMOKE_PROBLEM_SUFFIXES = ("ReLU", "Softmax")

# Curated 41-problem KernelBench-v3 subset used in Wafer's Elliot configs.
ELLIOT_V3_PROBLEM_NAMES = (
    "Square_matrix_multiplication_",
    "Standard_matrix_multiplication_",
    "Batched_matrix_multiplication",
    "Matrix_vector_multiplication_",
    "Matmul_with_irregular_shapes_",
    "Tall_skinny_matrix_multiplication_",
    "Softmax",
    "GELU_",
    "RMSNorm_",
    "LayerNorm",
    "Max_Pooling_2D",
    "Sum_reduction_over_a_dimension",
    "conv_standard_2D__square_input__square_kernel",
    "conv_depthwise_2D_square_input_square_kernel",
    "CrossEntropyLoss",
    "Conv3d_Softmax_MaxPool_MaxPool",
    "Conv2d_InstanceNorm_Divide",
    "Matmul_Swish_Sum_GroupNorm",
    "Matmul_Scaling_ResidualAdd",
    "Conv2d_Subtract_Tanh_Subtract_AvgPool",
    "Conv2d_Activation_BatchNorm",
    "Matmul_MaxPool_Sum_Scale",
    "Matmul_Swish_Scaling",
    "Matmul_Dropout_Mean_Softmax",
    "Conv2d_BatchNorm_Scaling",
    "Conv2d_Tanh_Scaling_BiasAdd_Max",
    "Conv2d_GroupNorm_Scale_MaxPool_Clamp",
    "Matmul_Divide_GELU",
    "Matmul_AvgPool_GELU_Scale_Max",
    "Matmul_GELU_Softmax",
    "VisionAttention",
    "MinGPTCausalAttention",
    "MiniGPTBlock",
    "DeepSeek_MLA",
    "DeepSeek_MoE",
    "GroupedQueryAttention",
    "FP8_Matmul",
    "MoE_GatedGEMM",
    "INT4_Quantized_GEMM",
    "GatedDeltaNet",
    "KimiDeltaAttention",
)

# Tiny v3 bring-up subset. Keep this deliberately small and cheap.
KERNELBENCH_V3_SMOKE_PROBLEM_NAMES = ("Softmax",)


def select_problem_suffixes(
    prompts: list[dict[str, Any]],
    suffixes: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Keep prompts whose names end with one of the requested suffixes."""
    selected = [
        prompt
        for prompt in prompts
        if any(str(prompt.get("name", "")).endswith(suffix) for suffix in suffixes)
    ]
    if not selected:
        raise ValueError(f"No KernelBench prompts matched suffixes {suffixes!r}")
    return selected


def select_problem_names(
    prompts: list[dict[str, Any]],
    problem_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Keep prompts whose normalized problem_name matches one of the requested names."""
    selected = [
        prompt
        for prompt in prompts
        if str(prompt.get("problem_name", prompt.get("name", ""))) in problem_names
    ]
    if not selected:
        raise ValueError(f"No KernelBench prompts matched problem_names {problem_names!r}")
    return selected


def load_kernelbench_smoke_prompts(*, backend: str = "cuda") -> list[dict[str, Any]]:
    """Load the canonical easy smoke subset for KernelBench eval bring-up."""
    prompts = load_kernelbench_prompts(levels=[1], backend=backend)
    return select_problem_suffixes(prompts, SMOKE_PROBLEM_SUFFIXES)


def load_kernelbench_v3_elliot_prompts(
    *,
    backend: str = "cuda",
    root_path: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Load Elliot's curated 41-problem KernelBench-v3 subset."""
    prompts = load_kernelbench_v3_prompts(root_path=root_path, backend=backend)
    return select_problem_names(prompts, ELLIOT_V3_PROBLEM_NAMES)


def load_kernelbench_v3_smoke_prompts(
    *,
    backend: str = "cuda",
    root_path: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Load the tiny KernelBench-v3 smoke subset."""
    prompts = load_kernelbench_v3_prompts(root_path=root_path, backend=backend, levels=[1])
    return select_problem_names(prompts, KERNELBENCH_V3_SMOKE_PROBLEM_NAMES)
