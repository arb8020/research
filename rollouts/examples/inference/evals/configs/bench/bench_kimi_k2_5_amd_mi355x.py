"""Throughput/latency benchmark for Kimi K2.5 on AMD Instinct MI355X (8x GPU).

Uses SGLang's MI355X ROCm image. vLLM ROCm fails with
assert num_head_qo % 16 == 0 at tp=8 for Kimi K2.5's attention head count.
Weights downloaded to /models/hf_cache on the NVMe drive (7TB, mounted at /models).

Hardware: root@66.42.120.238
  - 8x AMD Instinct MI355X, 288GB VRAM each (2.3TB total)
  - /dev/nvme0n1 mounted at /models (6.6TB free)

Model: moonshotai/Kimi-K2.5
  - MoE architecture, FP8 weights
  - Requires all 8 GPUs (--tensor-parallel-size 8)

First run will download model weights — expect 30-60min before server starts.
Subsequent runs reuse the cached weights from /models/hf_cache.

Usage:
    python -m argus run --config examples/inference/evals/configs/bench/bench_kimi_k2_5_amd_mi355x.py --force-deploy-committed

    # Monitor:
    tail -f results/eval/<run>/run.jsonl | jq 'select(.event | test("inference_startup|health|eval_end"))'

Reference:
    SGLang cookbook: https://cookbook.sglang.io/autoregressive/Moonshotai/Kimi-K2.5
    Docker image: lmsysorg/sglang:v0.5.9-rocm700-mi35x
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

MODEL = "moonshotai/Kimi-K2.5"
PORT = 30000

# dsv32-rocm is a newer build than v0.5.9-rocm700-mi35x and supports the
# DeepSeek V2 attention family (which Kimi K2.5 uses). v0.5.9 has a
# ForwardMetadata unpack bug in deepseek_v2.py that's fixed in this image.
_SGLANG_IMAGE = "lmsysorg/sglang:dsv32-rocm"

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
# SGLang flags per cookbook:
#   --reasoning-parser kimi_k2  — split thinking/content in output
#   --tool-call-parser kimi_k2  — structured tool calls
#   --dp 8 --enable-dp-attention — data-parallel attention for throughput
# ---------------------------------------------------------------------------

_docker_run = (
    f"docker run --rm"
    f" --device /dev/kfd --device /dev/dri"
    f" --group-add video"
    f" --shm-size 128G"
    f" --ipc host --network host"
    f" --volume /models:/models"
    f" --env ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
    f" --env HF_HOME=/models/hf_cache"
    f" --name sglang_bench_{PORT}"
    f" {_SGLANG_IMAGE}"
    f" bash -c 'USE_ROCM=true ROCM_HOME=/opt/rocm pip install -q /root/tilelang && python -m sglang.launch_server"
    f" --model-path {MODEL}"
    f" --host 0.0.0.0"
    f" --port {PORT}"
    f" --tp 8"
    f" --trust-remote-code"
    f" --reasoning-parser kimi"
    f" --tool-call-parser kimi_k2"
    f" --mem-fraction-static 0.85"
    f" --context-length 8192'"
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
    output=EvalOutputConfig(experiment_name="bench_kimi_k2_5_amd_mi355x"),
    hardware=hardware,
    summary_distribution_percentiles=SUMMARY_DISTRIBUTION_PERCENTILES,
)
