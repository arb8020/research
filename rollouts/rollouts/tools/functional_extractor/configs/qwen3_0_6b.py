"""Qwen3-0.6B verification config.

Small dense model, cheap GPU is fine.
~1.2GB model size, runs on 16GB VRAM easily.
"""

from tools.functional_extractor.config import DeploymentConfig, VerificationConfig

deployment = DeploymentConfig(
    vram_gb=16,
    max_price=0.30,
)

verification = VerificationConfig(
    model_name="Qwen/Qwen3-0.6B",
    forward_fn_name="qwen3_forward",
    test_inputs=[[1, 2, 3, 4, 5], [100, 200, 300, 400, 500]],
)
