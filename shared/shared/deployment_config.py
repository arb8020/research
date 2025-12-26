"""Shared deployment configuration components.

Casey Muratori: Extract ONLY what's actually duplicated, not what looks similar.

After writing both qwen3_next and clicker, the ONLY true duplication is GPU selection.
Everything else is superficially similar but actually different.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GPUConfig:
    """GPU hardware specification - the ONE thing truly shared.

    Both qwen3_next and clicker need to:
    - Specify which GPUs to use (gpu_ranks)
    - Track GPU type for memory planning
    - Specify Python/CUDA versions for venv setup

    This is proven duplication after implementing both systems.
    """

    # GPU selection
    gpu_ranks: list[int] = field(default_factory=lambda: [0])  # Which GPUs to use
    gpu_type: str = "H100"  # For memory estimation: "H100", "B200", "A100", etc.

    # Runtime environment (for venv bootstrap)
    python_version: str = "3.11"  # Python version for venv
    cuda_version: str = "12.4"  # CUDA version

    def __post_init__(self) -> None:
        """Validate GPU config."""
        assert len(self.gpu_ranks) > 0, "gpu_ranks cannot be empty"
        assert all(r >= 0 for r in self.gpu_ranks), (
            f"gpu_ranks must be non-negative: {self.gpu_ranks}"
        )
        assert len(self.gpu_ranks) == len(set(self.gpu_ranks)), (
            f"gpu_ranks has duplicates: {self.gpu_ranks}"
        )

    def get_cuda_visible_devices(self) -> str:
        """Get CUDA_VISIBLE_DEVICES string.

        Returns:
            Comma-separated GPU indices (e.g., "0", "2,3")
        """
        return ",".join(str(r) for r in self.gpu_ranks)
