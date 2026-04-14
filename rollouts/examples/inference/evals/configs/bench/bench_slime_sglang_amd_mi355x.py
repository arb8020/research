"""Throughput/latency benchmark for SGLang on AMD Instinct MI355X (8x GPU node).

Uses AMD's official ROCm 7.0 + SGLang Docker image rather than bare pip install.
The image has the correct ROCm/PyTorch/SGLang version matrix pre-solved.

Hardware: root@66.42.120.238, 8x AMD Instinct MI355X (gfx950), ROCm 7.0
SSH key:  ~/.ssh/id_ed25519

Usage:
    # SSH eval endpoint not yet implemented — see docs/handoffs/ssh_eval_endpoint.md
    # Once implemented:
    python -m argus run --config examples/inference/evals/configs/bench/bench_slime_sglang_amd_mi355x.py

    # Monitor progress:
    tail -f results/eval/<run>/events.jsonl | jq .

    # Direct invocation (interactive debug mode):
    python -m rollouts.eval.run --config examples/inference/evals/configs/bench/bench_slime_sglang_amd_mi355x.py

Reference:
    InferenceX by SemiAnalysis: https://github.com/SemiAnalysisAI/InferenceX
    Docker image: rocm/sgl-dev:sglang-0.5.6.post1-rocm700-mi35x-mori-1224
    ROCm SGLang docs: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/sglang-mori-distributed.html
"""

from examples.inference.bench_config_lib import (
    SUMMARY_DISTRIBUTION_PERCENTILES as DEFAULT_SUMMARY_DISTRIBUTION_PERCENTILES,
)
from examples.inference.bench_config_lib import (
    InferenceBenchSLA,
    InferenceBenchWorkload,
    build_bench_tasks,
)
from examples.inference.bench_workload_lib import prepare_bench_messages
from rollouts.core import Score
from rollouts.eval import AgentRunSpec, EvalOutputConfig, EvalRunConfig, EvalTaskSpec
from rollouts.eval.configs import EndpointCapabilities, OwnedEndpoint
from rollouts.training.configs import DepsConfig, HardwareConfig
from rollouts.training.scoring import FunctionScorer

_no_op_scorer = FunctionScorer(lambda attempt, _ctx: Score(metrics=()))

MODEL = "Qwen/Qwen3-0.6B"
PORT = 30000

# AMD's official Docker image for MI355X: ROCm 7.0, SGLang 0.5.6, MoRI backend.
# No pip version matrix to manage — the image is the reproducible environment.
_SGLANG_IMAGE = "rocm/sgl-dev:sglang-0.5.6.post1-rocm700-mi35x-mori-1224"

# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------

WORKLOAD = InferenceBenchWorkload(
    kind="random",
    num_prompts=200,
    input_len=512,
    output_len=256,
    seed=42,
    max_concurrent=16,
)
SLA = InferenceBenchSLA()
SUMMARY_DISTRIBUTION_PERCENTILES = DEFAULT_SUMMARY_DISTRIBUTION_PERCENTILES

tasks = build_bench_tasks(WORKLOAD)

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

hardware = HardwareConfig(
    provider="ssh",
    ssh="root@66.42.120.238:22",
    ssh_key_path="~/.ssh/id_ed25519",
    gpu_type="MI355X",
    gpu_count=2,
    use_torchrun=False,
    deps=DepsConfig(
        bootstrap_commands=(f"docker pull {_SGLANG_IMAGE}",),
    ),
)

# ---------------------------------------------------------------------------
# Endpoint
# launch_cmd runs SGLang inside Docker with ROCm device access.
# ROCR_VISIBLE_DEVICES=0,1 scopes to the first two GPUs.
# --network host binds directly to port 30000; SSH tunnel handles local access.
# ---------------------------------------------------------------------------

_docker_run = (
    f"docker run --rm --detach"
    f" --device /dev/kfd --device /dev/dri"
    f" --group-add video"
    f" --ipc host --network host"
    f" --shm-size 16G"
    f" --env ROCR_VISIBLE_DEVICES=0,1"
    f" --name sglang_bench_{PORT}"
    f" {_SGLANG_IMAGE}"
    f" python -m sglang.launch_server"
    f" --model-path {MODEL}"
    f" --host 0.0.0.0"
    f" --port {PORT}"
    f" --dtype bfloat16"
    f" --tp 2"
    f" --trust-remote-code"
)

endpoint = OwnedEndpoint(
    spec="custom-http",
    launch_cmd=_docker_run,
    cuda_device_ids=(0, 1),
    model=MODEL,
    port=PORT,
    capabilities=EndpointCapabilities(weight_sync=None),
    startup_timeout=300.0,
    max_tokens=WORKLOAD.output_len,
)

# ---------------------------------------------------------------------------
# Eval task
# ---------------------------------------------------------------------------

eval_task = EvalTaskSpec(
    tasks=tasks,
    run_spec=AgentRunSpec(
        endpoint=endpoint,
        prepare_messages=prepare_bench_messages,
    ),
    scorer=_no_op_scorer,
    run=EvalRunConfig(
        max_concurrent=WORKLOAD.max_concurrent,
        max_samples=WORKLOAD.num_prompts,
        max_turns=1,
        verbose=False,
        show_progress=True,
    ),
    output=EvalOutputConfig(experiment_name="bench_slime_sglang_amd_mi355x"),
    hardware=hardware,
    summary_distribution_percentiles=SUMMARY_DISTRIBUTION_PERCENTILES,
)
