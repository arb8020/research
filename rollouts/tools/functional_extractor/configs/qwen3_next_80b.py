"""Qwen3-Next-80B verification config.

MoE model: 80B total params, 3B active per token, 512 experts.
Needs multi-GPU A100 setup.

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
    gpu_count=2,
    max_price=5.0,
    min_cpu_ram=64,
    container_disk=250,
)

verification = VerificationConfig(
    model_name="Qwen/Qwen3-Next-80B-A3B-Instruct",
    forward_fn_name="qwen3_next_forward",
    test_inputs=[[1, 2, 3, 4, 5]],
    rtol=1e-4,  # Looser tolerance for large MoE
    atol=1e-4,
    device_map="balanced",  # Multi-GPU balancing
)
