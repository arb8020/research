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
