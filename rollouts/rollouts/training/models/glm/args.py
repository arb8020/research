"""GLM model arguments for GLM-4.7-Flash and GLM-5.

GLM-4.7-Flash: 30B MoE, 3.6B active (64 routed experts, 4 active per token)
GLM-5: 744B MoE, 40B active (256 routed experts, 8 active per token)

Key architectural features:
- QKNorm: RMSNorm on Q and K after projection
- Sigmoid-gated MoE routing with e_score_correction_bias
- Shared expert that always runs on all tokens
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import nn
from torchtitan.config import JobConfig
from torchtitan.models.moe import MoEArgs
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.train_spec import BaseModelArgs
from torchtitan.tools.logging import logger


@dataclass
class GLMModelArgs(BaseModelArgs):
    """Configuration for GLM-4.7-Flash and GLM-5 models.

    Key differences from Llama/Qwen:
    - QKNorm: RMSNorm applied to Q and K after projection (before RoPE)
    - Attention bias: Q/K/V projections have bias, output has no bias
    - MoE routing: Sigmoid-gated with e_score_correction_bias
    - Shared expert: Always runs in addition to routed experts
    """

    # Core architecture
    dim: int = 2048
    n_layers: int = 47
    n_heads: int = 20
    n_kv_heads: int = 20
    vocab_size: int = 154880
    head_dim: int = 128
    hidden_dim: int = 10240  # Dense MLP intermediate

    # GLM-specific normalization
    norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    qk_norm: bool = True
    attn_bias: bool = True  # GLM uses bias on Q/K/V (not output)
    max_seq_len: int = 202752
    depth_init: bool = True

    # Attention
    attn_type: str = "sdpa"
    attn_mask_type: str = "causal"

    # Weight tying
    enable_weight_tying: bool = True

    # MoE config
    moe_enabled: bool = True
    moe_inter_dim: int = 1536
    n_routed_experts: int = 64
    n_shared_experts: int = 1
    num_experts_per_tok: int = 4
    routed_scaling_factor: float = 1.8
    topk_method: str = "noaux_tc"
    moe_args: MoEArgs = field(default_factory=MoEArgs)

    def __post_init__(self) -> None:
        """Sync moe_args from top-level fields."""
        if self.moe_enabled:
            self.moe_args = MoEArgs(
                num_experts=self.n_routed_experts,
                num_shared_experts=self.n_shared_experts,
                top_k=self.num_experts_per_tok,
                score_func="sigmoid",  # GLM uses sigmoid, not softmax
                route_norm=True,
                route_scale=self.routed_scaling_factor,
                score_before_experts=False,
            )

    def update_from_config(self, job_config: JobConfig, **kwargs: object) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        if self.moe_enabled:
            self.moe_args._debug_force_load_balance = job_config.debug.moe_force_load_balance

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        if self.moe_enabled:
            return get_moe_model_nparams_and_flops(self, model, 2 * self.head_dim, seq_len)
        else:
            n_params = sum(p.numel() for p in model.parameters())
            flops = 6 * n_params * seq_len
            return n_params, flops


# Pre-defined model configurations
GLM_4_7_FLASH = GLMModelArgs(
    # GLM-4.7-Flash: 30B total, 3.6B active
    dim=2048,
    n_layers=47,
    n_heads=20,
    n_kv_heads=20,
    vocab_size=154880,
    head_dim=128,
    hidden_dim=10240,
    norm_eps=1e-5,
    rope_theta=1000000.0,
    max_seq_len=202752,
    moe_enabled=True,
    moe_inter_dim=1536,
    n_routed_experts=64,
    n_shared_experts=1,
    num_experts_per_tok=4,
    routed_scaling_factor=1.8,
)

GLM_5 = GLMModelArgs(
    # GLM-5: 744B total, 40B active
    dim=6144,
    n_layers=78,
    n_heads=64,
    n_kv_heads=64,
    vocab_size=154880,
    head_dim=64,  # GLM-5 uses smaller head_dim
    hidden_dim=12288,
    norm_eps=1e-5,
    rope_theta=1000000.0,
    max_seq_len=202752,
    moe_enabled=True,
    moe_inter_dim=2048,
    n_routed_experts=256,
    n_shared_experts=1,
    num_experts_per_tok=8,
    routed_scaling_factor=2.5,
)

# Debug model for testing
GLM_DEBUG = GLMModelArgs(
    dim=256,
    n_layers=4,
    n_heads=4,
    n_kv_heads=4,
    vocab_size=1024,
    head_dim=64,
    hidden_dim=512,
    max_seq_len=1024,
    moe_enabled=True,
    moe_inter_dim=128,
    n_routed_experts=8,
    n_shared_experts=1,
    num_experts_per_tok=2,
    routed_scaling_factor=1.8,
)
