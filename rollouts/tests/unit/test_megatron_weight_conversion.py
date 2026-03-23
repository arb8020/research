from __future__ import annotations

import torch

from rollouts.training.backends.megatron.weight_conversion import convert_megatron_to_hf


def test_qwen3_local_layernorm_names_convert_to_hf_runtime_names() -> None:
    param = torch.randn(8)

    input_layernorm = convert_megatron_to_hf(
        model_name="PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT",
        name="decoder.layers.3.input_layernorm.weight",
        param=param,
        vocab_size=151936,
        num_layers=28,
        num_attention_heads=16,
        hidden_size=1024,
        num_query_groups=8,
        kv_channels=64,
    )
    pre_mlp_layernorm = convert_megatron_to_hf(
        model_name="PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT",
        name="decoder.layers.3.pre_mlp_layernorm.weight",
        param=param,
        vocab_size=151936,
        num_layers=28,
        num_attention_heads=16,
        hidden_size=1024,
        num_query_groups=8,
        kv_channels=64,
    )

    assert input_layernorm == [("model.layers.3.input_layernorm.weight", param)]
    assert pre_mlp_layernorm == [("model.layers.3.post_attention_layernorm.weight", param)]
