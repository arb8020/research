"""Throughput/latency benchmark for DeepSeek-V3.2 on MI355X using the older ROCm image.

This is an experimental variant of the main DeepSeek V3.2 bench config. Its purpose is
to isolate the container/image difference by swapping in the older ROCm image referenced
by InferenceX / SemiAnalysis without perturbing the rest of the rollouts harness.

Important:
- This config is intentionally separate from the working `dsv32-rocm` config.
- It keeps the same benchmark workload and DeepSeek-specific chat template so results are
  easier to compare against the current serving path.
- It may fail at launch if the older image is missing DeepSeek V3.2 runtime support.
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

MODEL = "deepseek-ai/DeepSeek-V3.2"
PORT = 30000

# Older ROCm image referenced by InferenceX / SemiAnalysis material.
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
    max_concurrent=32,
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
    gpu_count=8,
    use_torchrun=False,
    deps=DepsConfig(
        bootstrap_commands=(
            "mount /dev/nvme0n1 /models 2>/dev/null || true",
            f"docker pull {_SGLANG_IMAGE}",
        ),
    ),
)

# ---------------------------------------------------------------------------
# Endpoint
# This variant is meant to test the image difference first, not the tool-calling path.
# Keep launch args close to the working bench config, but do not rely on parser-specific
# flags that may not exist on the older image.
#
# DeepSeek V3.2 NSA still needs tilelang. The older image does not have it installed as a
# package, but it does ship the source tree under /opt/tilelang.
# ---------------------------------------------------------------------------

_docker_run = (
    f"docker run --rm"
    f" --device /dev/kfd --device /dev/dri"
    f" --group-add video"
    f" --ipc host --network host"
    f" --shm-size 128G"
    f" --volume /models:/models"
    f" --env ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
    f" --env HF_HOME=/models/hf_cache"
    f" --env SGLANG_NSA_FUSE_TOPK=false"
    f" --env SGLANG_NSA_KV_CACHE_STORE_FP8=false"
    f" --env SGLANG_NSA_USE_REAL_INDEXER=true"
    f" --env SGLANG_NSA_USE_TILELANG_PREFILL=True"
    f" --name sglang_bench_{PORT}"
    f" {_SGLANG_IMAGE}"
    f" bash -c 'USE_ROCM=true ROCM_HOME=/opt/rocm pip install -q /opt/tilelang && python -m sglang.launch_server"
    f" --model-path {MODEL}"
    f" --host 0.0.0.0"
    f" --port {PORT}"
    f" --tp 8"
    f" --trust-remote-code"
    f" --chat-template /sgl-workspace/sglang/examples/chat_template/tool_chat_template_deepseekv32.jinja"
    f" --disable-cuda-graph"
    f" --mem-fraction-static 0.85"
    f" --page-size 64"
    f" --nsa-prefill tilelang"
    f" --nsa-decode aiter"
    f"'"
)

endpoint = OwnedEndpoint(
    spec="custom-http",
    launch_cmd=_docker_run,
    cuda_device_ids=(0, 1, 2, 3, 4, 5, 6, 7),
    model=MODEL,
    port=PORT,
    capabilities=EndpointCapabilities(weight_sync=None),
    startup_timeout=7200.0,
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
    output=EvalOutputConfig(experiment_name="bench_deepseek_v3_2_inferencex_image_amd_mi355x"),
    hardware=hardware,
    summary_distribution_percentiles=SUMMARY_DISTRIBUTION_PERCENTILES,
)
