from rollouts.eval.configs import (
    EndpointCapabilities,
    ExternalEndpoint,
    OwnedEndpoint,
    materialize_endpoint,
)
from rollouts.eval.endpoint_realization import _remote_service_spec


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


def test_owned_endpoint_health_url_uses_readiness_path() -> None:
    endpoint = OwnedEndpoint(
        spec="custom-http",
        model="Qwen/Qwen3-0.6B",
        cuda_device_ids=(0,),
        port=30000,
        capabilities=EndpointCapabilities(weight_sync=None),
        readiness_path="/v1",
    )

    assert endpoint.health_url == "http://localhost:30000/v1"


def test_remote_service_spec_uses_owned_endpoint_readiness_path_for_launch_module() -> None:
    owned_endpoint = OwnedEndpoint(
        spec="custom-http",
        model="Qwen/Qwen3-0.6B",
        cuda_device_ids=(0,),
        port=30002,
        capabilities=EndpointCapabilities(weight_sync=None),
        launch_module="minisgl",
        readiness_path="/v1",
    )

    launch_cmd, readiness_target = _remote_service_spec(
        worker=None,
        output_dir=None,
        remote_python="/opt/venvs/rollouts/bin/python",
        owned_endpoint=owned_endpoint,
    )

    assert "python -m minisgl" in launch_cmd
    assert readiness_target == "/v1"
