"""Fast smoke witness for mini-sglang.

This intentionally uses the generic custom-http launch path today.

TODO(mini-sglang-owned-endpoint): the named ``mini-sglang`` inference spec in
``rollouts`` still lowers through the SGLang launcher contract, which hardcodes
SGLang CLI flags like ``--model-path`` and ``--mem-fraction-static``. mini-sglang
has a different CLI surface, so keep this ugly-but-honest compatibility path
until the shared launcher learns the real mini-sglang process contract.
"""

from examples.inference.smoke_witness_lib import make_smoke_eval_task
from rollouts.eval.configs import EndpointCapabilities, OwnedEndpoint
from rollouts.training.configs import DepsConfig, HardwareConfig

MODEL = "Qwen/Qwen3-0.6B"
PORT = 30002

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
            "'minisgl @ git+https://github.com/sgl-project/mini-sglang.git'",
        ),
    ),
)

endpoint = OwnedEndpoint(
    spec="custom-http",
    model=MODEL,
    cuda_device_ids=(0,),
    port=PORT,
    capabilities=EndpointCapabilities(weight_sync=None),
    launch_module="minisgl",
    startup_timeout=180.0,
    max_tokens=64,
)

eval_task = make_smoke_eval_task(
    endpoint=endpoint,
    hardware=hardware,
    experiment_name="mini_sglang_smoke_eval",
)
