"""Configuration for functional extraction verification.

Defines deployment and verification configs for running on remote GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DeploymentConfig:
    """Remote GPU deployment configuration."""

    vram_gb: int = 24
    gpu_filter: str | None = None  # "A100", "H100", "4090", etc.
    gpu_count: int = 1
    max_price: float = 0.50
    min_cpu_ram: int = 32
    container_disk: int = 50
    volume_disk: int = 0
    cloud_type: str = "secure"
    ssh_timeout: int = 300


@dataclass
class VerificationConfig:
    """Functional verification configuration."""

    model_name: str
    forward_fn_name: str = "forward"
    test_inputs: list[list[int]] = field(default_factory=lambda: [[1, 2, 3, 4, 5]])
    rtol: float = 1e-5
    atol: float = 1e-5
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
