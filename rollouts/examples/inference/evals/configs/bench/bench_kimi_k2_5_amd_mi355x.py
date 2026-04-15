"""Throughput/latency benchmark for Kimi K2.5 on AMD Instinct MI355X (8x GPU).

Uses vLLM's official ROCm image — verified on 8x MI300X/MI355X per vLLM docs.
Weights downloaded to /models/hf_cache on the NVMe drive (7TB, mounted at /models).

Hardware: root@66.42.120.238
  - 8x AMD Instinct MI355X, 288GB VRAM each (2.3TB total)
  - /dev/nvme0n1 mounted at /models (6.6TB free)

Model: moonshotai/Kimi-K2.5
  - MoE architecture, FP8 weights
  - Requires all 8 GPUs (--tensor-parallel-size 8)
  - Verified on 8x MI300X/MI355X per vLLM recipe docs

First run will download model weights — expect 30-60min before server starts.
Subsequent runs reuse the cached weights from /models/hf_cache.

Usage:
    python -m argus run --config examples/inference/evals/configs/bench/bench_kimi_k2_5_amd_mi355x.py --force-deploy-committed

    # Monitor:
    tail -f results/eval/<run>/run.jsonl | jq 'select(.event | test("inference_startup|health|eval_end"))'

Reference:
    vLLM ROCm recipe: https://docs.vllm.ai/projects/recipes/en/latest/moonshotai/Kimi-K2.5.html
    Docker image: vllm/vllm-openai-rocm:latest
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

# vLLM's official ROCm image — verified on MI300X/MI355X.
_VLLM_IMAGE = "vllm/vllm-openai-rocm:latest"

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
            f"docker pull {_VLLM_IMAGE}",
        ),
    ),
)

# ---------------------------------------------------------------------------
# Endpoint
# vLLM ROCm flags per official recipe:
#   VLLM_ROCM_USE_AITER=1              — AITER attention/tensor optimizations
#   VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 — faster all-reduce
#   VLLM_ROCM_USE_AITER_RMSNORM=0      — disable AITER for RMSNorm (stability)
#   --block-size=1                     — required for ROCm prefix caching
# ---------------------------------------------------------------------------

_docker_run = (
    f"docker run --rm"
    f" --device /dev/kfd --device /dev/dri"
    f" --group-add video"
    f" --security-opt seccomp=unconfined"
    f" --ipc host --network host"
    f" --volume /models:/models"
    f" --env ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
    f" --env HF_HOME=/models/hf_cache"
    f" --env VLLM_ROCM_USE_AITER=1"
    f" --env VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4"
    f" --env VLLM_ROCM_USE_AITER_RMSNORM=0"
    f" --name vllm_bench_{PORT}"
    f" {_VLLM_IMAGE}"
    f" {MODEL}"
    f" --tensor-parallel-size 8"
    f" --mm-encoder-tp-mode data"
    f" --block-size 1"
    f" --tool-call-parser kimi_k2"
    f" --reasoning-parser kimi_k2"
    f" --enable-auto-tool-choice"
    f" --enable-prefix-caching"
    f" --trust-remote-code"
    f" --host 0.0.0.0"
    f" --port {PORT}"
    f" --max-model-len 8192"
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
