"""GLM-4 dense model bridge for mbridge.

Maps between HuggingFace GLM-4 weights and Megatron-Core format.
GLM-4 uses a Qwen2-like architecture with some differences:
- QKV bias enabled
- QK layernorm
- Post-MLP and post-attention layernorms
- Interleaved rotary embeddings

Ported from SLIME: slime_plugins/mbridge/glm4.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

# Lazy imports - mbridge may not be installed
_BRIDGE_REGISTERED = False


def _register() -> bool:
    """Register GLM4Bridge with mbridge. Returns True if successful."""
    global _BRIDGE_REGISTERED
    if _BRIDGE_REGISTERED:
        return True

    try:
        from mbridge.core import LLMBridge, register_model
    except ImportError:
        return False

    try:
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
        )
    except ImportError:
        # Megatron not installed, can't register
        return False

    @register_model("glm4")
    class GLM4Bridge(LLMBridge):
        """Bridge for GLM-4 dense models (e.g., GLM-Z1-9B).

        Extends LLMBridge with GLM-4 specific weight mappings and config.
        """

        _DIRECT_MAPPING = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }

        _ATTENTION_MAPPING = {
            "self_attention.linear_proj.weight": [
                "model.layers.{layer_number}.self_attn.o_proj.weight"
            ],
            "self_attention.linear_qkv.layer_norm_weight": [
                "model.layers.{layer_number}.input_layernorm.weight"
            ],
            "self_attention.q_layernorm.weight": [
                "model.layers.{layer_number}.self_attn.q_norm.weight"
            ],
            "self_attention.k_layernorm.weight": [
                "model.layers.{layer_number}.self_attn.k_norm.weight"
            ],
            "self_attention.linear_qkv.weight": [
                "model.layers.{layer_number}.self_attn.q_proj.weight",
                "model.layers.{layer_number}.self_attn.k_proj.weight",
                "model.layers.{layer_number}.self_attn.v_proj.weight",
            ],
            "self_attention.linear_qkv.bias": [
                "model.layers.{layer_number}.self_attn.q_proj.bias",
                "model.layers.{layer_number}.self_attn.k_proj.bias",
                "model.layers.{layer_number}.self_attn.v_proj.bias",
            ],
        }

        _MLP_MAPPING = {
            "mlp.linear_fc1.weight": [
                "model.layers.{layer_number}.mlp.gate_up_proj.weight",
            ],
            "mlp.linear_fc1.layer_norm_weight": [
                "model.layers.{layer_number}.post_attention_layernorm.weight"
            ],
            "mlp.linear_fc2.weight": ["model.layers.{layer_number}.mlp.down_proj.weight"],
        }

        def _build_config(self) -> Any:
            """Build TransformerConfig for GLM-4."""
            return self._build_base_config(
                add_qkv_bias=True,
                qk_layernorm=False,
                post_mlp_layernorm=True,
                post_self_attn_layernorm=True,
                rotary_interleaved=True,
            )

        def _get_transformer_layer_spec(self) -> Any:
            """Get transformer layer spec with post-layernorms."""
            return get_gpt_layer_with_transformer_engine_spec(
                post_self_attn_layernorm=True,
                post_mlp_layernorm=True,
            )

        def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
            """Map Megatron-Core weight names to HuggingFace names."""
            assert "_extra_state" not in mcore_weights_name

            if mcore_weights_name in self._DIRECT_MAPPING:
                return [self._DIRECT_MAPPING[mcore_weights_name]]

            if "post_self_attn_layernorm" in mcore_weights_name:
                layer_number = mcore_weights_name.split(".")[2]
                return [f"model.layers.{layer_number}.post_self_attn_layernorm.weight"]
            elif "post_mlp_layernorm" in mcore_weights_name:
                layer_number = mcore_weights_name.split(".")[2]
                return [f"model.layers.{layer_number}.post_mlp_layernorm.weight"]
            elif "self_attention" in mcore_weights_name:
                return self._weight_name_mapping_attention(mcore_weights_name)
            elif "mlp" in mcore_weights_name:
                return self._weight_name_mapping_mlp(mcore_weights_name)
            else:
                raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

    _BRIDGE_REGISTERED = True
    return True


# Try to register on import
_register()
