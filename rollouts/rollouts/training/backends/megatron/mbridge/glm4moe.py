"""GLM-4.7-Flash MoE model bridge for mbridge.

Maps between HuggingFace GLM-4.7-Flash weights and Megatron-Core format.
GLM-4.7-Flash is a 30B-A3B MoE model with:
- 64 routed experts, top-4 routing
- 1 shared expert
- Sigmoid router with expert bias
- Multi-Latent Attention (MLA) for efficient KV compression

Ported from SLIME: slime_plugins/mbridge/glm4moe.py
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

_BRIDGE_REGISTERED = False


def _register() -> bool:
    """Register GLM4MoEBridge with mbridge. Returns True if successful."""
    global _BRIDGE_REGISTERED
    if _BRIDGE_REGISTERED:
        return True

    try:
        from mbridge.core import register_model
        from mbridge.models import Qwen2Bridge, Qwen2MoEBridge
    except ImportError:
        return False

    @register_model("glm4_moe")
    class GLM4MoEBridge(Qwen2MoEBridge):
        """Bridge for GLM-4.7-Flash MoE model.

        Extends Qwen2MoEBridge with GLM-specific MoE mappings:
        - Router expert bias
        - Shared experts
        - MTP (Multi-Token Prediction) layers
        """

        _MLP_MAPPING = {
            **(Qwen2MoEBridge._MLP_MAPPING),
            **(Qwen2Bridge._MLP_MAPPING),
            "mlp.router.expert_bias": [
                "model.layers.{layer_number}.mlp.gate.e_score_correction_bias"
            ],
            "shared_experts.linear_fc1.weight": [
                "model.layers.{layer_number}.mlp.shared_experts.gate_proj.weight",
                "model.layers.{layer_number}.mlp.shared_experts.up_proj.weight",
            ],
            "shared_experts.linear_fc2.weight": [
                "model.layers.{layer_number}.mlp.shared_experts.down_proj.weight"
            ],
        }

        _MTP_MAPPING = {
            "enorm.weight": ["model.layers.{layer_number}.enorm.weight"],
            "hnorm.weight": ["model.layers.{layer_number}.hnorm.weight"],
            "eh_proj.weight": ["model.layers.{layer_number}.eh_proj.weight"],
            "final_layernorm.weight": ["model.layers.{layer_number}.shared_head.norm.weight"],
        }

        def _weight_name_mapping_mtp(self, name: str, num_layers: int) -> list[str]:
            """Map MTP layer weight names."""
            convert_names: list[str] = []
            for keyword, mapping_names in self._MTP_MAPPING.items():
                if keyword in name:
                    convert_names.extend([x.format(layer_number=num_layers) for x in mapping_names])
                    break
                elif "mlp" in name:
                    mtp_layer_index = int(re.findall(r"mtp\.layers\.(\d+)\.", name)[0])
                    name_ = re.sub(
                        r"^mtp\.layers.\d+.transformer_layer",
                        f"model.layers.{num_layers + mtp_layer_index}",
                        name,
                    )
                    convert_names = self._weight_name_mapping_mlp(name_)
                    break
                elif "self_attention" in name:
                    mtp_layer_index = int(re.findall(r"mtp\.layers.(\d+)\.", name)[0])
                    name_ = re.sub(
                        r"^mtp\.layers.\d+.transformer_layer",
                        f"model.layers.{num_layers + mtp_layer_index}",
                        name,
                    )
                    convert_names = self._weight_name_mapping_attention(name_)
                    break

            if len(convert_names) == 0:
                raise NotImplementedError(f"Unsupported parameter name: {name}")
            return convert_names

        def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
            """Map Megatron-Core weight names to HuggingFace names."""
            assert "_extra_state" not in mcore_weights_name

            direct_name_mapping = {
                "embedding.word_embeddings.weight": "model.embed_tokens.weight",
                "decoder.final_layernorm.weight": "model.norm.weight",
                "output_layer.weight": "lm_head.weight",
            }
            if mcore_weights_name in direct_name_mapping:
                return [direct_name_mapping[mcore_weights_name]]

            if "mtp" in mcore_weights_name:
                return self._weight_name_mapping_mtp(mcore_weights_name, self.config.num_layers)
            elif "self_attention" in mcore_weights_name:
                return self._weight_name_mapping_attention(mcore_weights_name)
            elif "mlp" in mcore_weights_name:
                return self._weight_name_mapping_mlp(mcore_weights_name)
            else:
                raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

        def _build_config(self) -> Any:
            """Build TransformerConfig for GLM-4.7-Flash MoE."""
            return self._build_base_config(
                use_cpu_initialization=False,
                # MoE specific
                moe_ffn_hidden_size=self.hf_config.moe_intermediate_size,
                moe_router_bias_update_rate=0.001,
                moe_router_topk=self.hf_config.num_experts_per_tok,
                num_moe_experts=self.hf_config.n_routed_experts,
                moe_router_load_balancing_type="none",  # No aux loss for RL
                moe_grouped_gemm=True,
                moe_router_score_function="sigmoid",
                moe_router_enable_expert_bias=True,
                moe_router_pre_softmax=True,
                # Other optimizations
                # persist_layer_norm requires Transformer Engine; disable for torch LayerNorm
                persist_layer_norm=False,
                bias_activation_fusion=True,
                bias_dropout_fusion=True,
                # GLM specific
                qk_layernorm=self.hf_config.use_qk_norm,
                add_qkv_bias=True,
                add_bias_linear=False,
                rotary_interleaved=True,
            )

    _BRIDGE_REGISTERED = True
    return True


# Try to register on import
_register()
