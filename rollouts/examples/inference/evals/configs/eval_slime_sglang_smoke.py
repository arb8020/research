"""Fast smoke witness for the slime-sglang realization.

This uses the named OwnedEndpoint path so eval owns the lifecycle and emits the
same startup/health/stall telemetry we care about for inference-engine
iteration.

Usage:
    python -m argus run --config inference.eval_slime_sglang_smoke

    # Monitor progress:
    tail -f results/eval/<run>/events.jsonl | jq .

    # Direct invocation (interactive debug mode):
    python -m rollouts.eval.run --config inference.eval_slime_sglang_smoke
"""

from examples.inference.smoke_witness_lib import make_smoke_eval_task
from rollouts.eval.configs import EndpointCapabilities, OwnedEndpoint
from rollouts.training.configs import DepsConfig, HardwareConfig

MODEL = "Qwen/Qwen3-0.6B"
PORT = 30000

hardware = HardwareConfig(
    provider="modal",
    gpu_type="A100",
    gpu_count=1,
    use_torchrun=False,
    keep_alive=True,
    deps=DepsConfig(
        bootstrap_commands=(
            "~/.local/bin/uv pip install --python /opt/venvs/rollouts/bin/python "
            "torch transformers accelerate fastapi uvicorn "
            "'sglang[all] @ git+https://github.com/sgl-project/sglang.git@main#subdirectory=python'",
        ),
    ),
)

endpoint = OwnedEndpoint(
    spec="slime-sglang",
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
    experiment_name="slime_sglang_smoke_eval",
)
