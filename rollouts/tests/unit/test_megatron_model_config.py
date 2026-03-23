from types import SimpleNamespace

from rollouts.training.backends.megatron.qwen import normalize_te_only_megatron_config


def test_normalize_te_only_transformer_config_fields_clears_local_flags() -> None:
    config = SimpleNamespace(
        transformer_impl="local",
        persist_layer_norm=True,
        apply_rope_fusion=True,
        gradient_accumulation_fusion=True,
    )

    normalize_te_only_megatron_config(config)

    assert config.persist_layer_norm is False
    assert config.apply_rope_fusion is False
    assert config.gradient_accumulation_fusion is False
