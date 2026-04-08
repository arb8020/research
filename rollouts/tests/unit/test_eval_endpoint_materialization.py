from rollouts.eval.configs import (
    EndpointCapabilities,
    ExternalEndpoint,
    OwnedEndpoint,
    materialize_endpoint,
)


def test_materialize_external_endpoint_preserves_extra_params() -> None:
    endpoint = ExternalEndpoint(
        url="http://localhost:30000/v1",
        model="Qwen/Qwen3-0.6B",
        provider="sglang",
        extra_params={"chat_template_kwargs": {"enable_thinking": False}},
    )

    materialized = materialize_endpoint(endpoint)

    assert materialized.extra_params == {"chat_template_kwargs": {"enable_thinking": False}}


def test_materialize_owned_endpoint_preserves_extra_params() -> None:
    endpoint = OwnedEndpoint(
        spec="slime-sglang",
        model="Qwen/Qwen3-0.6B",
        cuda_device_ids=(0,),
        port=30000,
        capabilities=EndpointCapabilities(weight_sync=None),
        extra_params={"chat_template_kwargs": {"enable_thinking": False}},
    )

    materialized = materialize_endpoint(endpoint)

    assert materialized.extra_params == {"chat_template_kwargs": {"enable_thinking": False}}
