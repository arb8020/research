"""Mixed-workload serving scenario against one DSV3.2 endpoint on MI355X.

Three workloads run in parallel through a single shared sglang endpoint,
exercising the multi-workload serving path end-to-end:

- tau2 retail: multi-turn agent with user simulator + tool use
- kimi_verifier (K2VV): single-turn tool-call conformance requests
- sharegpt bench: raw throughput load from real first-turn human prompts

This is the "can we serve a realistic mixture at all, observably" smoke.
Sample counts are intentionally small so the whole mixture finishes in a
few minutes after sglang warmup; the point is mixture shape, not numbers.

Artifacts per workload land under results/serving/<run>/workloads/<name>/:
  - report.json            eval-style summary (tau2, sharegpt)
  - engine_report.json     K2VV-style summary (kimi_verifier)
  - engine.jsonl           per-sample or per-request stream
  - events.jsonl           agent-loop events (tau2, sharegpt)

Usage:
    python -m argus run \\
        --config rollouts/examples/serving/mixture_mi355x_smoke.py \\
        --force-deploy-committed
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from examples.inference.bench_config_lib import (
    InferenceBenchWorkload,
    build_bench_tasks,
)
from examples.inference.bench_workload_lib import prepare_bench_messages
from rollouts.core import Score
from rollouts.eval import (
    AgentRunSpec,
    EvalOutputConfig,
    EvalRunConfig,
    EvalTaskSpec,
)
from rollouts.eval.configs import EndpointCapabilities, OwnedEndpoint
from rollouts.serving.configs import (
    EvalServingWorkload,
    ServingOutputConfig,
    ServingScenario,
    ToolCallVerifierWorkload,
)
from rollouts.training.configs import DepsConfig, HardwareConfig
from rollouts.training.scoring import FunctionScorer

# ---------------------------------------------------------------------------
# tau2 imports (same fragile sys.path dance as tau2_retail_c4_smoke.py)
# ---------------------------------------------------------------------------

_EVALS_ROOT = Path(__file__).resolve().parents[2] / "evals"
if str(_EVALS_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVALS_ROOT))


def _ensure_tau2_data_dir() -> None:
    if os.environ.get("TAU2_DATA_DIR"):
        return

    candidates = [
        *Path.home().glob(".cache/uv/git-v0/checkouts/*/*/data/tau2/domains/retail/tasks.json"),
        *Path.home().glob(
            ".cache/uv/archive-v0/*/inspect_evals/tau2/data/domains/retail/tasks.json"
        ),
    ]
    if not candidates:
        raise FileNotFoundError(
            "Could not locate tau2 retail task data. Set TAU2_DATA_DIR explicitly."
        )
    os.environ["TAU2_DATA_DIR"] = str(candidates[0].parents[3])


_ensure_tau2_data_dir()

from harbor_v0.config_types import ModalHarborHost  # noqa: E402
from harbor_v0.eval import make_environment as harbor_make_environment  # noqa: E402
from harbor_v0.eval import prepare_messages as harbor_prepare_messages  # noqa: E402
from harbor_v0.eval import score_sample as harbor_score_sample  # noqa: E402
from tau2_v0.eval import make_environment, prepare_messages, score_sample  # noqa: E402
from tau2_v0.prepare import DEFAULT_USER_ENDPOINT, build_sample_rows  # noqa: E402

from rollouts.environments.harbor_environment import attach_harbor_host_to_tasks  # noqa: E402

# ---------------------------------------------------------------------------
# Shared endpoint (DSV3.2 on MI355X, same docker config as the single-workload
# smokes). One endpoint, shared by all workloads below.
# ---------------------------------------------------------------------------

MODEL = "deepseek-ai/DeepSeek-V3.2"
PORT = 30000
NODE_COST_USD_PER_HOUR = 22.18

_SGLANG_IMAGE = "lmsysorg/sglang:v0.5.9-rocm700-mi35x"

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
    f" --name sglang_mixture_{PORT}"
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
    max_tokens=1024,
)

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
# Workload 1: tau2 retail (multi-turn agent with user simulator + tools)
# ---------------------------------------------------------------------------

_TAU2_TASKS = build_sample_rows(
    domain="retail",
    limit=2,
    user_endpoint=DEFAULT_USER_ENDPOINT,
)

# The scenario endpoint wins over eval_task.run_spec.endpoint at serve time
# (see rollouts.serving.run._rewrite_workload_eval_task), so this placeholder
# endpoint is only used to construct the EvalTaskSpec shape. Actual requests
# go to the MI355X sglang endpoint above.
_tau2_eval = EvalTaskSpec(
    tasks=_TAU2_TASKS,
    run_spec=AgentRunSpec(
        endpoint=endpoint,
        prepare_messages=prepare_messages,
        environment_factory=make_environment,
    ),
    scorer=FunctionScorer(score_sample),
    run=EvalRunConfig(
        max_concurrent=2,
        max_samples=2,
        max_turns=20,
        verbose=False,
        show_progress=False,
    ),
    output=EvalOutputConfig(experiment_name="tau2_retail"),
)

# ---------------------------------------------------------------------------
# Workload 2: kimi_verifier (K2VV single-turn tool-call conformance)
# ---------------------------------------------------------------------------

_KVV_DEFAULTS = {
    "temperature": 0.6,
    "top_p": 0.95,
}

# ---------------------------------------------------------------------------
# Workload 3: sharegpt bench (raw-throughput, real first-turn human prompts)
# ---------------------------------------------------------------------------

_RESEARCH_ROOT = Path(__file__).resolve().parents[2].parent
_SHAREGPT_PATH = str(_RESEARCH_ROOT / "data" / "ShareGPT_V3_unfiltered_cleaned_split.jsonl")

_SHAREGPT_WORKLOAD = InferenceBenchWorkload(
    kind="sharegpt",
    num_prompts=8,
    input_len=512,
    output_len=256,
    dataset_path=_SHAREGPT_PATH,
    seed=42,
    max_concurrent=8,
)
_sharegpt_tasks = build_bench_tasks(_SHAREGPT_WORKLOAD)
_no_op_scorer = FunctionScorer(lambda attempt, _ctx: Score(metrics=()))

_sharegpt_eval = EvalTaskSpec(
    tasks=_sharegpt_tasks,
    run_spec=AgentRunSpec(
        endpoint=endpoint,
        prepare_messages=prepare_bench_messages,
    ),
    scorer=_no_op_scorer,
    run=EvalRunConfig(
        max_concurrent=_SHAREGPT_WORKLOAD.max_concurrent,
        max_samples=_SHAREGPT_WORKLOAD.num_prompts,
        max_turns=1,
        verbose=False,
        show_progress=False,
    ),
    output=EvalOutputConfig(experiment_name="sharegpt_bench"),
)

# ---------------------------------------------------------------------------
# Workload 4: harbor (Modal-backed TB2 tool execution, agent multi-turn).
# DSV3.2 will likely underperform Sonnet on pass rate; we include it for
# traffic shape (multi-turn + Modal sandbox orchestration), not eval quality.
# ---------------------------------------------------------------------------

_HARBOR_TASKS = attach_harbor_host_to_tasks(
    [{"task_id": "cancel-async-tasks"}],
    ModalHarborHost(app_name="rollouts-harbor"),
)

_harbor_eval = EvalTaskSpec(
    tasks=_HARBOR_TASKS,
    run_spec=AgentRunSpec(
        endpoint=endpoint,
        prepare_messages=harbor_prepare_messages,
        environment_factory=harbor_make_environment,
    ),
    scorer=FunctionScorer(harbor_score_sample),
    run=EvalRunConfig(
        max_concurrent=1,
        max_samples=1,
        max_turns=20,
        verbose=False,
        show_progress=False,
    ),
    output=EvalOutputConfig(experiment_name="harbor_tb2"),
)

# ---------------------------------------------------------------------------
# The mixture
# ---------------------------------------------------------------------------

serving_scenario = ServingScenario(
    endpoint=endpoint,
    hardware=hardware,
    workloads=[
        EvalServingWorkload(
            name="tau2_retail",
            eval_task=_tau2_eval,
            concurrency=2,
            max_samples=2,
        ),
        ToolCallVerifierWorkload(
            name="kimi_verifier",
            concurrency=2,
            max_samples=4,
            extra_body=_KVV_DEFAULTS,
            request_timeout_s=600.0,
        ),
        EvalServingWorkload(
            name="sharegpt_bench",
            eval_task=_sharegpt_eval,
            concurrency=_SHAREGPT_WORKLOAD.max_concurrent,
            max_samples=_SHAREGPT_WORKLOAD.num_prompts,
        ),
        EvalServingWorkload(
            name="harbor_tb2",
            eval_task=_harbor_eval,
            concurrency=1,
            max_samples=1,
        ),
    ],
    output=ServingOutputConfig(
        experiment_name="mixture_mi355x_smoke",
    ),
    # Iterate fast: if sglang is already up at port 30000 on the MI355X node
    # from a prior run, tunnel to it instead of paying the ~10min warmup
    # cost. If the launch config changes (model / docker flags / image),
    # SSH in and `docker stop sglang_mixture_30000` first.
    reuse_running_endpoint=True,
)
