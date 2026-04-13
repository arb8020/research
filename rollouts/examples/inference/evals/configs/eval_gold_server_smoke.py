"""Fast smoke witness for the HuggingFace gold inference server.

Usage:
    python -m argus run --config examples/inference/evals/configs/eval_gold_server_smoke.py

    # Monitor progress:
    tail -f results/eval/<run>/events.jsonl | jq .

    # Direct invocation (interactive debug mode):
    python -m rollouts.eval.run --config inference.eval_gold_server_smoke
"""

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
            "~/.local/bin/uv pip install --python /opt/venvs/rollouts/bin/python torch transformers accelerate uvicorn fastapi",
        ),
    ),
)

endpoint = OwnedEndpoint(
    spec="custom-http",
    model=MODEL,
    cuda_device_ids=(0,),
    port=PORT,
    capabilities=EndpointCapabilities(weight_sync=None),
    launch_module="rollouts.inference.gold_server",
    startup_timeout=180.0,
    max_tokens=64,
)

eval_task = make_smoke_eval_task(
    endpoint=endpoint,
    hardware=hardware,
    experiment_name="gold_server_smoke_eval",
)
