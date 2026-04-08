"""Fast smoke witness for the patched qed-vllm realization."""

from examples.inference.smoke_witness_lib import make_smoke_eval_task
from rollouts.eval.configs import EndpointCapabilities, OwnedEndpoint
from rollouts.training.configs import DepsConfig, HardwareConfig

MODEL = "Qwen/Qwen3-0.6B"
PORT = 30001

hardware = HardwareConfig(
    provider="modal",
    gpu_type="A100",
    gpu_count=1,
    use_torchrun=False,
    keep_alive=True,
    deps=DepsConfig(
        bootstrap_commands=(
            "~/.local/bin/uv pip install --python /opt/venvs/rollouts/bin/python "
            "torch transformers accelerate fastapi uvicorn trio hf-transfer "
            "'vllm>=0.13.0,<0.14.0'",
        ),
    ),
)

endpoint = OwnedEndpoint(
    spec="qed-vllm",
    model=MODEL,
    cuda_device_ids=(0,),
    port=PORT,
    capabilities=EndpointCapabilities(weight_sync=None),
    mem_fraction=0.6,
    startup_timeout=300.0,
    max_tokens=64,
    extra_params={"chat_template_kwargs": {"enable_thinking": False}},
)

eval_task = make_smoke_eval_task(
    endpoint=endpoint,
    hardware=hardware,
    experiment_name="qed_vllm_smoke_eval",
)
