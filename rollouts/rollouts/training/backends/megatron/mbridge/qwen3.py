# ruff: noqa: ANN202
"""Custom mbridge bridge for plain Qwen3 models on our local Megatron spec."""

from __future__ import annotations

from mbridge.core import register_model
from mbridge.models import Qwen2Bridge


@register_model("qwen3")
class Qwen3Bridge(Qwen2Bridge):
    """Bridge for plain Qwen3 models.

    Our `custom_spec:qwen3` path uses Megatron local layer specs that expose
    `input_layernorm.weight` and `pre_mlp_layernorm.weight` directly, instead of
    the older `linear_qkv.layer_norm_weight` / `linear_fc1.layer_norm_weight`
    names that upstream `Qwen2Bridge` expects. The HF checkpoint semantics are
    the same; only the Megatron-side parameter naming differs.
    """

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        assert "_extra_state" not in mcore_weights_name

        if mcore_weights_name in self._DIRECT_MAPPING:
            return [self._DIRECT_MAPPING[mcore_weights_name]]

        parts = mcore_weights_name.split(".")
        if len(parts) >= 4 and parts[0] == "decoder" and parts[1] == "layers":
            layer_number = parts[2]
            rest = ".".join(parts[3:])
            if rest == "input_layernorm.weight":
                return [f"model.layers.{layer_number}.input_layernorm.weight"]
            if rest == "pre_mlp_layernorm.weight":
                return [f"model.layers.{layer_number}.post_attention_layernorm.weight"]

        if "self_attention" in mcore_weights_name:
            return self._weight_name_mapping_attention(mcore_weights_name)
        if "mlp" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")
