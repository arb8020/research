"""Throughput/latency benchmark for DeepSeek-V3.2 on AMD Instinct MI355X (8x GPU).

Uses AMD's official ROCm 7.0 + SGLang Docker image.
Weights downloaded to /models/hf_cache on the NVMe drive (7TB, mounted at /models).

Hardware: root@66.42.120.238
  - 8x AMD Instinct MI355X, 288GB VRAM each (2.3TB total)
  - /dev/nvme0n1 mounted at /models (6.6TB free)

Model: deepseek-ai/DeepSeek-V3.2
  - 685B params MoE, ~37B active per token
  - FP8 weights, ~690GB on disk
  - Needs all 8 GPUs (8 x 288GB = 2.3TB for weights + KV cache)

First run will download ~690GB of weights — expect 30-60min before SGLang starts.
Subsequent runs reuse the cached weights from /models/hf_cache.

Usage:
    python -m argus run --config examples/inference/evals/configs/bench/bench_deepseek_v3_2_amd_mi355x.py --force-deploy-committed

    # Monitor progress:
    tail -f results/eval/<run>/run.jsonl | jq 'select(.event | test("inference_startup|health|eval_end"))'

Reference:
    InferenceX by SemiAnalysis: https://github.com/SemiAnalysisAI/InferenceX
    Docker image: rocm/sgl-dev:sglang-0.5.6.post1-rocm700-mi35x-mori-1224
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

# v0.5.9 ships the `deepseekv32` tool-call parser (matches DSV3.2's DSML tag
# output format). Earlier `dsv32-rocm` image only had `deepseekv31`, which
# matches V3.1's different wire format and silently drops V3.2 tool calls as
# plain text content. v0.5.9 also has tilelang pre-installed.
_SGLANG_IMAGE = "lmsysorg/sglang:v0.5.9-rocm700-mi35x"

# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------

WORKLOAD = InferenceBenchWorkload(
    kind="random",
    num_prompts=200,
    input_len=512,
    output_len=256,
    seed=42,
    max_concurrent=32,  # higher concurrency to stress the larger model
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
            # Ensure NVMe is mounted — idempotent, fails silently if already mounted
            "mount /dev/nvme0n1 /models 2>/dev/null || true",
            # Pull the Docker image if not already cached.
            f"docker pull {_SGLANG_IMAGE}",
        ),
    ),
)

# ---------------------------------------------------------------------------
# Endpoint
# All 8 MI355X GPUs, tp=8, FP8 precision.
# Weights cached to /models/hf_cache on the 7TB NVMe (bind-mounted into container).
# First run downloads ~690GB — startup_timeout is set high accordingly.
# --enable-dp-attention improves MoE throughput on multi-GPU.
# ---------------------------------------------------------------------------

# TODO(nix): f-string-as-launch-spec pain — see the matching TODO in
# examples/serving/kimi_verifier_deepseek_v32_mi355x_smoke.py. When nix
# fixes this, the NSA env var block, MI355X device flags, and sglang
# launch-arg fragments should become named structured fragments shared
# across all DSV3.2 configs.
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
    # SemiAnalysis's working MI355X config (InferenceX benchmarks/single_node/
    # glm5_fp8_mi355x.sh) disables fused decode MLA alongside tilelang decode
    # backend. Keeps us on the known-good path.
    f" --env SGLANG_ROCM_FUSED_DECODE_MLA=0"
    # First-run aiter MoE kernel JIT compile on v0.5.9 overruns SGLang's
    # default 600s warmup timeout; bump to 30 min.
    f" --env SGLANG_WARMUP_TIMEOUT=1800"
    f" --name sglang_bench_{PORT}"
    f" {_SGLANG_IMAGE}"
    f" python -m sglang.launch_server"
    f" --model-path {MODEL}"
    f" --host 0.0.0.0"
    f" --port {PORT}"
    f" --tp 8"
    f" --trust-remote-code"
    f" --tool-call-parser deepseekv32"
    f" --chat-template /sgl-workspace/sglang/examples/chat_template/tool_chat_template_deepseekv32.jinja"
    f" --disable-cuda-graph"
    f" --mem-fraction-static 0.85"
    f" --page-size 64"
    # `--nsa-decode-backend aiter` hits a sglang↔aiter version-skew bug on
    # v0.5.9 (softmax_scale float leaks into page_size int slot); tilelang
    # decode avoids the broken aiter kernel path. Matches SemiAnalysis.
    f" --nsa-prefill-backend tilelang"
    f" --nsa-decode-backend tilelang"
    f" --enable-cache-report"
    # Same rationale as warmup timeout: first-run forward batches run during
    # aiter JIT and can easily exceed SGLang's default ~300s watchdog.
    f" --watchdog-timeout 1800"
)

endpoint = OwnedEndpoint(
    spec="custom-http",
    launch_cmd=_docker_run,
    cuda_device_ids=(0, 1, 2, 3, 4, 5, 6, 7),
    model=MODEL,
    port=PORT,
    capabilities=EndpointCapabilities(weight_sync=None),
    startup_timeout=7200.0,  # 2h — first run downloads 690GB of weights
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
    output=EvalOutputConfig(experiment_name="bench_deepseek_v3_2_amd_mi355x"),
    hardware=hardware,
    summary_distribution_percentiles=SUMMARY_DISTRIBUTION_PERCENTILES,
)
