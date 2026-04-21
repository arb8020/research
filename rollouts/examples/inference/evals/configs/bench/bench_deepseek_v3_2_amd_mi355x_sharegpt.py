"""ShareGPT-driven throughput/latency benchmark for DeepSeek-V3.2 on MI355X.

Research-side twin of courier/bench/bench_deepseek_v3_2_amd_mi355x_sharegpt.py.
Kept in ~/research because courier's argus-deploy path is broken (REPO_ROOT
leak in vendored rollouts — see memory project_courier_argus_vendoring).

Same docker/hardware setup as bench_deepseek_v3_2_amd_mi355x.py (random
synthetic prompts). The only delta is the workload: real ShareGPT first-turn
human prompts instead of synthetic word-pool draws.

Dataset: anon8231489123/ShareGPT_Vicuna_unfiltered (ShareGPT_V3_unfiltered_cleaned_split).
Staged locally under ~/research/data/ as a JSONL (symlinked from courier).

Usage:
    python -m argus run --config rollouts/examples/inference/evals/configs/bench/bench_deepseek_v3_2_amd_mi355x_sharegpt.py --force-deploy-committed
"""

from pathlib import Path

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
NODE_COST_USD_PER_HOUR = 22.18

_SGLANG_IMAGE = "lmsysorg/sglang:v0.5.9-rocm700-mi35x"

# Resolve dataset path relative to the research repo root (walk up from this
# config file past rollouts/examples/inference/evals/configs/bench/ to ~/research).
_RESEARCH_ROOT = Path(__file__).resolve().parents[6]
_SHAREGPT_PATH = str(_RESEARCH_ROOT / "data" / "ShareGPT_V3_unfiltered_cleaned_split.jsonl")

WORKLOAD = InferenceBenchWorkload(
    kind="sharegpt",
    num_prompts=200,
    input_len=512,  # sentinel; sharegpt uses real prompt lengths from the dataset
    output_len=256,
    dataset_path=_SHAREGPT_PATH,
    seed=42,
    max_concurrent=32,
)
SLA = InferenceBenchSLA()
SUMMARY_DISTRIBUTION_PERCENTILES = DEFAULT_SUMMARY_DISTRIBUTION_PERCENTILES

tasks = build_bench_tasks(WORKLOAD)

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
    f" --env SGLANG_ROCM_FUSED_DECODE_MLA=0"
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
    f" --reasoning-parser deepseek-v3"
    f" --disable-cuda-graph"
    f" --mem-fraction-static 0.85"
    f" --page-size 64"
    f" --nsa-prefill-backend tilelang"
    f" --nsa-decode-backend tilelang"
    f" --enable-cache-report"
    f" --watchdog-timeout 1800"
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
    output=EvalOutputConfig(experiment_name="bench_deepseek_v3_2_amd_mi355x_sharegpt"),
    hardware=hardware,
    summary_distribution_percentiles=SUMMARY_DISTRIBUTION_PERCENTILES,
)
