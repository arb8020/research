"""GLM-4.7 MoE Lite model bridge for mbridge.

Maps between HuggingFace GLM-4.7-Flash weights and Megatron-Core format.
GLM-4.7-Flash uses `glm4_moe_lite` model type which differs from `glm4_moe`:
- rope_theta comes from rope_scaling['rope_theta'] (not top-level)
- Uses mlp_layer_types config field
- Has MLA (Multi-Latent Attention) support

Ported from SLIME: slime_plugins/mbridge/glm4moe_lite.py
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

_BRIDGE_REGISTERED = False
_REGISTERED_BRIDGE: type[Any] | None = None


def _register() -> bool:
    """Register GLM4MoELiteBridge with mbridge. Returns True if successful."""
    global _BRIDGE_REGISTERED
    if _BRIDGE_REGISTERED:
        return True

    try:
        from mbridge.core import register_model
        from mbridge.models import DeepseekV3Bridge
    except ImportError:
        return False

    try:
        from megatron.core.transformer.enums import AttnBackend
    except ImportError:
        return False

    @register_model("glm4_moe_lite")
    class GLM4MoELiteBridge(DeepseekV3Bridge):
        """Bridge for GLM-4.7 MoE Lite models (e.g. GLM-4.7-Flash).

        Glm4MoeLiteConfig differs from DeepseekV3Config:
          - rope_theta comes from rope_scaling['rope_theta'] (not top-level)
          - Uses mlp_layer_types config field
        """

        _EXPECTED_NUM_LAYERS = 47

        @property
        def rope_theta(self) -> float:
            return self.hf_config.rope_scaling["rope_theta"]

        @property
        def moe_layer_freq(self) -> list[int]:
            return [0 if t == "dense" else 1 for t in self.hf_config.mlp_layer_types]

        def _build_config(self) -> Any:
            hf_config = self.hf_config
            mla_rope_config = {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 1,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 4096,
                "type": "rope",
            }
            rope_scaling = getattr(hf_config, "rope_scaling", None)
            if rope_scaling is not None:
                mla_rope_config.update(rope_scaling)

            base_config = {
                # RMSNorm models hit FusedLayerNorm assertions under the
                # Transformer Engine path in current Megatron-Core.
                "transformer_impl": "local",
                # Current Modal/Megatron/TE builds do not reliably provide a
                # valid fused/TE dot-product backend for GLM MLA. Prefer the
                # local backend for now so the model semantics stay live while
                # we sort out a truthful fast-path choice later.
                "attention_backend": AttnBackend.local,
                "layernorm_epsilon": hf_config.rms_norm_eps,
                "ffn_hidden_size": hf_config.intermediate_size,
                "qk_layernorm": True,
                # MoE
                "moe_ffn_hidden_size": hf_config.moe_intermediate_size,
                "moe_token_dispatcher_type": "alltoall",
                "moe_router_bias_update_rate": 0.001,
                "moe_router_enable_expert_bias": True,
                "moe_router_topk": hf_config.num_experts_per_tok,
                "num_moe_experts": hf_config.n_routed_experts,
                "moe_shared_expert_intermediate_size": hf_config.moe_intermediate_size
                * getattr(hf_config, "n_shared_experts", 1),
                "moe_aux_loss_coeff": getattr(hf_config, "aux_loss_alpha", 0.001),
                "moe_router_load_balancing_type": "none",
                "moe_shared_expert_overlap": True,
                # grouped_gemm not installed in Modal image; disable for now
                "moe_grouped_gemm": False,
                "moe_router_score_function": "sigmoid",
                "moe_router_pre_softmax": True,
                "moe_router_topk_scaling_factor": getattr(hf_config, "routed_scaling_factor", 1.0),
                "moe_layer_freq": self.moe_layer_freq,
                # MLA (Multi-Latent Attention)
                "q_lora_rank": hf_config.q_lora_rank,
                "kv_lora_rank": hf_config.kv_lora_rank,
                "qk_head_dim": hf_config.qk_nope_head_dim,
                "qk_pos_emb_head_dim": hf_config.qk_rope_head_dim,
                "v_head_dim": hf_config.v_head_dim,
                "rotary_base": self.rope_theta,
                "rotary_scaling_factor": mla_rope_config["factor"],
                "rope_type": mla_rope_config["type"],
                "mscale": mla_rope_config["mscale"],
                "mscale_all_dim": mla_rope_config["mscale_all_dim"],
                "beta_fast": mla_rope_config["beta_fast"],
                "beta_slow": mla_rope_config["beta_slow"],
                # Megatron-Core MoE options
                "moe_router_dtype": "fp32",
                "disable_bf16_reduced_precision_matmul": True,
                # Other
                # persist_layer_norm requires Transformer Engine; disable for torch LayerNorm
                "persist_layer_norm": False,
                "bias_activation_fusion": True,
                "bias_dropout_fusion": True,
            }

            import megatron.core

            megatron_version = getattr(megatron.core, "__version__", "0.0")
            if megatron_version >= "0.14":
                base_config["original_max_position_embeddings"] = mla_rope_config[
                    "original_max_position_embeddings"
                ]
            else:
                base_config["max_position_embeddings"] = mla_rope_config[
                    "original_max_position_embeddings"
                ]

            # MTP (Multi-Token Prediction)
            num_nextn_predict_layers = getattr(hf_config, "num_nextn_predict_layers", None)
            if num_nextn_predict_layers is not None:
                base_config["mtp_num_layers"] = num_nextn_predict_layers
                base_config["mtp_loss_scaling_factor"] = 0.1

            return self._build_base_config(**base_config)

        def _get_gptmodel_args(self) -> dict[str, Any]:
            return dict(
                vocab_size=self.hf_config.vocab_size,
                max_sequence_length=self.hf_config.max_position_embeddings,
                # MLA layers consume RoPE internally from the transformer config.
                # Passing a top-level GPT rotary embedding path causes Megatron to
                # thread rotary_pos_emb into MultiLatentAttention, which current
                # Megatron rejects.
                position_embedding_type="none",
            )

        def _weight_name_mapping_mlp(self, name: str) -> list[str]:
            local_expert_match = re.match(
                r"decoder\.layers\.(\d+)\.mlp\.experts\.local_experts\.(\d+)\.(linear_fc[12])\.weight",
                name,
            )
            if local_expert_match:
                layer_number, expert_number, proj = local_expert_match.groups()
                if proj == "linear_fc1":
                    return [
                        f"model.layers.{layer_number}.mlp.experts.{expert_number}.gate_proj.weight",
                        f"model.layers.{layer_number}.mlp.experts.{expert_number}.up_proj.weight",
                    ]
                return [f"model.layers.{layer_number}.mlp.experts.{expert_number}.down_proj.weight"]

            shared_expert_match = re.match(
                r"decoder\.layers\.(\d+)\.mlp\.shared_experts\.(linear_fc[12])\.weight",
                name,
            )
            if shared_expert_match:
                layer_number, proj = shared_expert_match.groups()
                if proj == "linear_fc1":
                    return [
                        f"model.layers.{layer_number}.mlp.shared_experts.gate_proj.weight",
                        f"model.layers.{layer_number}.mlp.shared_experts.up_proj.weight",
                    ]
                return [f"model.layers.{layer_number}.mlp.shared_experts.down_proj.weight"]

            router_bias_match = re.match(
                r"decoder\.layers\.(\d+)\.mlp\.router\.expert_bias",
                name,
            )
            if router_bias_match:
                layer_number = router_bias_match.group(1)
                return [f"model.layers.{layer_number}.mlp.gate.e_score_correction_bias"]

            return super()._weight_name_mapping_mlp(name)

        def _convert_mtp_param(self, name: str) -> list[str]:
            assert self.config.mtp_num_layers == 1, "only support one mtp layer for now"
            assert self.config.num_layers == self._EXPECTED_NUM_LAYERS, (
                f"glm4_moe_lite only supports {self._EXPECTED_NUM_LAYERS} layers for now"
            )

            mtp_layer_id = self.config.num_layers
            direct_name_mapping = {
                "mtp.layers.0.enorm.weight": f"model.layers.{mtp_layer_id}.enorm.weight",
                "mtp.layers.0.hnorm.weight": f"model.layers.{mtp_layer_id}.hnorm.weight",
                "mtp.layers.0.eh_proj.weight": f"model.layers.{mtp_layer_id}.eh_proj.weight",
                "mtp.layers.0.final_layernorm.weight": f"model.layers.{mtp_layer_id}.shared_head.norm.weight",
            }
            if name in direct_name_mapping:
                return [direct_name_mapping[name]]

            assert "mtp.layers.0.transformer_layer" in name, "mtp not found"
            proxy_name = name.replace(
                "mtp.layers.0.transformer_layer", f"decoder.layers.{mtp_layer_id}"
            )
            if "self_attention" in proxy_name or "input_layernorm.weight" in proxy_name:
                return self._weight_name_mapping_attention(proxy_name)
            if "mlp" in proxy_name:
                return self._weight_name_mapping_mlp(proxy_name)
            raise NotImplementedError(f"Unsupported parameter name: {name}")

        def _weight_to_hf_format(
            self, mcore_weights_name: str, mcore_weights: torch.Tensor
        ) -> tuple[list[str], list[torch.Tensor]]:
            if self.config.mtp_num_layers == 1:
                assert self.config.num_layers == self._EXPECTED_NUM_LAYERS, (
                    f"glm4_moe_lite only supports {self._EXPECTED_NUM_LAYERS} layers for now"
                )
                mtp_layer_id = self.config.num_layers
                shared_state_dict_mapping = {
                    "embedding.word_embeddings.weight": [
                        "model.embed_tokens.weight",
                        f"model.layers.{mtp_layer_id}.embed_tokens.weight",
                    ],
                    "output_layer.weight": [
                        "lm_head.weight",
                        f"model.layers.{mtp_layer_id}.shared_head.head.weight",
                    ],
                }
                if mcore_weights_name in shared_state_dict_mapping:
                    hf_names = shared_state_dict_mapping[mcore_weights_name]
                    return hf_names, [mcore_weights] * len(hf_names)
            return super()._weight_to_hf_format(mcore_weights_name, mcore_weights)

    global _REGISTERED_BRIDGE
    _REGISTERED_BRIDGE = GLM4MoELiteBridge
    _BRIDGE_REGISTERED = True
    return True


# Try to register on import
_register()
