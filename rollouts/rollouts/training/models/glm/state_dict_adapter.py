"""State dict adapter for GLM models.

Converts between HuggingFace checkpoint format and our torchtitan-style format.
"""

from __future__ import annotations

import re
from typing import Any

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .args import GLMModelArgs


class GLMStateDictAdapter(StateDictAdapter):
    """Convert between HF GLM checkpoints and our model format.

    Inherits from StateDictAdapter which provides:
    - fqn_to_index_mapping for multi-file safetensors
    - get_hf_storage_reader() implementation

    HuggingFace GLM naming:
        model.embed_tokens.weight
        model.layers.{i}.self_attn.q_proj.weight/bias
        model.layers.{i}.self_attn.k_proj.weight/bias
        model.layers.{i}.self_attn.v_proj.weight/bias
        model.layers.{i}.self_attn.o_proj.weight
        model.layers.{i}.self_attn.q_norm.weight
        model.layers.{i}.self_attn.k_norm.weight
        model.layers.{i}.input_layernorm.weight
        model.layers.{i}.post_attention_layernorm.weight
        model.layers.{i}.mlp.gate.weight  (router)
        model.layers.{i}.mlp.gate.e_score_correction_bias
        model.layers.{i}.mlp.experts.{j}.gate_proj.weight
        model.layers.{i}.mlp.experts.{j}.up_proj.weight
        model.layers.{i}.mlp.experts.{j}.down_proj.weight
        model.layers.{i}.mlp.shared_experts.gate_proj.weight
        model.layers.{i}.mlp.shared_experts.up_proj.weight
        model.layers.{i}.mlp.shared_experts.down_proj.weight
        model.norm.weight
        lm_head.weight

    Our naming:
        tok_embeddings.weight
        layers.{i}.attention.wq.weight/bias
        layers.{i}.attention.wk.weight/bias
        layers.{i}.attention.wv.weight/bias
        layers.{i}.attention.wo.weight
        layers.{i}.attention.q_norm.weight
        layers.{i}.attention.k_norm.weight
        layers.{i}.attention_norm.weight
        layers.{i}.ffn_norm.weight
        layers.{i}.moe.router.weight
        layers.{i}.moe.e_score_correction_bias
        layers.{i}.moe.experts.{j}.w1.weight (gate_proj)
        layers.{i}.moe.experts.{j}.w3.weight (up_proj)
        layers.{i}.moe.experts.{j}.w2.weight (down_proj)
        layers.{i}.moe.shared_expert.gate_proj.weight
        layers.{i}.moe.shared_expert.up_proj.weight
        layers.{i}.moe.shared_expert.down_proj.weight
        norm.weight
        output.weight
    """

    def __init__(self, model_args: GLMModelArgs, hf_assets_path: str | None) -> None:
        super().__init__(model_args, hf_assets_path)
        self.model_args = model_args

        # HF -> our format mapping (single layer index)
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            # Attention
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.wq.bias",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.wk.bias",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.wv.bias",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            # Layer norms
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # MoE router
            "model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.weight",
            "model.layers.{}.mlp.gate.e_score_correction_bias": "layers.{}.moe.e_score_correction_bias",
            # Shared expert
            "model.layers.{}.mlp.shared_experts.gate_proj.weight": "layers.{}.moe.shared_expert.gate_proj.weight",
            "model.layers.{}.mlp.shared_experts.up_proj.weight": "layers.{}.moe.shared_expert.up_proj.weight",
            "model.layers.{}.mlp.shared_experts.down_proj.weight": "layers.{}.moe.shared_expert.down_proj.weight",
            # Final
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HuggingFace state dict to our format."""
        state_dict = {}

        # Handle weight tying
        if self.model_args.enable_weight_tying and "lm_head.weight" not in hf_state_dict:
            if "model.embed_tokens.weight" in hf_state_dict:
                hf_state_dict["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"]

        for hf_key, value in hf_state_dict.items():
            new_key = self._convert_key_from_hf(hf_key)
            if new_key is not None:
                state_dict[new_key] = value

        return state_dict

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert our state dict to HuggingFace format."""
        to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
        hf_state_dict = {}

        for key, value in state_dict.items():
            new_key = self._convert_key_to_hf(key, to_hf_map)
            if new_key is not None:
                if self.model_args.enable_weight_tying and key == "output.weight":
                    continue
                hf_state_dict[new_key] = value

        return hf_state_dict

    def _convert_key_from_hf(self, hf_key: str) -> str | None:
        """Convert a single HF key to our format."""
        # Handle expert keys (two layer indices: layer and expert)
        if "mlp.experts" in hf_key:
            match = re.match(
                r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight",
                hf_key,
            )
            if match:
                layer_idx, expert_idx, proj_name = match.groups()
                proj_map = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
                return f"layers.{layer_idx}.moe.experts.{expert_idx}.{proj_map[proj_name]}.weight"
            return None

        # Handle single layer index keys
        for hf_pattern, our_pattern in self.from_hf_map.items():
            if our_pattern is None:
                continue

            hf_placeholders = hf_pattern.count("{}")
            our_placeholders = our_pattern.count("{}")

            if hf_placeholders == 1 and our_placeholders == 1:
                regex_pattern = hf_pattern.replace(".", r"\.").replace("{}", r"(\d+)")
                match = re.match(f"^{regex_pattern}$", hf_key)
                if match:
                    layer_idx = match.group(1)
                    return our_pattern.format(layer_idx)
            elif hf_placeholders == 0 and our_placeholders == 0:
                if hf_key == hf_pattern:
                    return our_pattern

        return None

    def _convert_key_to_hf(self, our_key: str, to_hf_map: dict[str, str]) -> str | None:
        """Convert a single key from our format to HF format."""
        # Handle expert keys
        if "moe.experts" in our_key:
            match = re.match(r"layers\.(\d+)\.moe\.experts\.(\d+)\.(w1|w2|w3)\.weight", our_key)
            if match:
                layer_idx, expert_idx, proj_name = match.groups()
                proj_map = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
                return f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_map[proj_name]}.weight"
            return None

        # Handle single layer index keys
        for our_pattern, hf_pattern in to_hf_map.items():
            our_placeholders = our_pattern.count("{}")

            if our_placeholders == 1:
                regex_pattern = our_pattern.replace(".", r"\.").replace("{}", r"(\d+)")
                match = re.match(f"^{regex_pattern}$", our_key)
                if match:
                    layer_idx = match.group(1)
                    return hf_pattern.format(layer_idx)
            elif our_placeholders == 0:
                if our_key == our_pattern:
                    return hf_pattern

        return None
