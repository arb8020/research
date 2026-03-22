from __future__ import annotations

import torch

from rollouts.training.backends.megatron.export_runtime import (
    _globalize_local_expert_name,
    _materialize_ep_local_param,
)
from rollouts.training.backends.megatron.weight_conversion.deepseekv3 import (
    convert_deepseekv3_to_hf,
)


def test_globalize_local_expert_name_uses_ep_rank_offset() -> None:
    assert (
        _globalize_local_expert_name(
            "decoder.layers.7.mlp.experts.local_experts.3.linear_fc1.weight",
            ep_rank=2,
            local_experts_per_rank=16,
        )
        == "decoder.layers.7.mlp.experts.35.linear_fc1.weight"
    )


def test_deepseekv3_converter_accepts_runtime_local_expert_names() -> None:
    class Args:
        hidden_size = 8
        num_attention_heads = 1
        num_query_groups = 1
        kv_channels = None

    param = torch.arange(16, dtype=torch.float32).view(4, 4)
    converted = convert_deepseekv3_to_hf(
        Args(),
        "decoder.layers.1.mlp.experts.local_experts.0.linear_fc1.weight",
        param,
    )

    assert [name for name, _ in converted] == [
        "model.layers.1.mlp.experts.0.gate_proj.weight",
        "model.layers.1.mlp.experts.0.up_proj.weight",
    ]
    assert converted[0][1].shape == (2, 4)
    assert converted[1][1].shape == (2, 4)


def test_materialize_ep_local_param_moves_tensor_to_cpu() -> None:
    param = torch.arange(6, dtype=torch.float32).view(2, 3)

    materialized = _materialize_ep_local_param(param)

    assert materialized.device.type == "cpu"
    assert torch.equal(materialized, param.cpu())
