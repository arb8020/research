"""Qwen3-Next-80B verification config.

MoE model: 80B total params, 3B active per token, 512 experts.
Uses FP8 version for single-GPU deployment (~80GB VRAM).

Architecture notes:
- 512 experts, 10 active per token
- hidden_size=2048, 48 layers
- Partial RoPE (25% of head_dim)
- GQA 8:1 ratio
"""

from tools.functional_extractor.config import DeploymentConfig, VerificationConfig

deployment = DeploymentConfig(
    vram_gb=80,
    gpu_filter="A100",
    gpu_count=1,  # FP8 fits on single 80GB GPU
    max_price=3.0,
    min_cpu_ram=64,
    container_disk=250,
)

verification = VerificationConfig(
    model_name="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",  # FP8 version
    forward_fn_name="qwen3_next_forward",
    test_inputs=[[1, 2, 3, 4, 5]],
    rtol=1e-3,  # Looser tolerance for FP8 quantization
    atol=1e-3,
    device_map="auto",  # Single GPU
)
