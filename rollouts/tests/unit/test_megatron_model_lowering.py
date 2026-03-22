import os

from rollouts.training.backends.megatron.runtime_env import (
    configure_transformer_engine_attention_env,
)
from rollouts.training.models.backend_lowering import lower_model_to_megatron
from rollouts.training.models.denotation import (
    CheckpointSemantics,
    HFModelSource,
    ModelArchitectureSemantics,
    ModelDenotation,
    MoESemantics,
)


def _glm4_moe_denotation() -> ModelDenotation:
    source = HFModelSource(name_or_path="zai-org/GLM-4.7-Flash")
    return ModelDenotation(
        source=source,
        architecture=ModelArchitectureSemantics(
            family="glm4_moe",
            hidden_size=2048,
            num_layers=47,
            num_attention_heads=20,
            ffn_hidden_size=10240,
            vocab_size=154880,
            max_sequence_length=202752,
            norm="rmsnorm",
            norm_epsilon=1e-5,
            attention="mla",
            num_kv_heads=20,
            moe=MoESemantics(num_experts=64, experts_per_token=4, shared_expert=True),
            tie_embeddings=False,
            uses_bias_linear=False,
        ),
        checkpoint=CheckpointSemantics(format="hf_pretrained", source=source),
        variant="4.7-flash",
    )


def test_glm4_moe_lowers_to_custom_spec() -> None:
    lowering = lower_model_to_megatron(
        _glm4_moe_denotation(),
        bridge_supports_provider=False,
    )

    assert lowering.adapter_kind == "custom_spec"
    assert any("bridge-native transformer layer spec" in note for note in lowering.validation_notes)


def test_glm4_mla_sets_te_flash_attention_env(monkeypatch) -> None:
    monkeypatch.delenv("NVTE_FUSED_ATTN", raising=False)
    monkeypatch.delenv("NVTE_FLASH_ATTN", raising=False)

    updates = configure_transformer_engine_attention_env(_glm4_moe_denotation())

    assert updates == {
        "NVTE_FUSED_ATTN": "0",
        "NVTE_FLASH_ATTN": "1",
    }
    assert os.environ["NVTE_FUSED_ATTN"] == "0"
    assert os.environ["NVTE_FLASH_ATTN"] == "1"


def test_glm4_mla_respects_existing_te_attention_env(monkeypatch) -> None:
    monkeypatch.setenv("NVTE_FUSED_ATTN", "1")
    monkeypatch.setenv("NVTE_FLASH_ATTN", "0")

    updates = configure_transformer_engine_attention_env(_glm4_moe_denotation())

    assert updates == {}
    assert os.environ["NVTE_FUSED_ATTN"] == "1"
    assert os.environ["NVTE_FLASH_ATTN"] == "0"
