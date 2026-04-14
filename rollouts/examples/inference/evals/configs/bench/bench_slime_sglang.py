"""Throughput/latency benchmark for the slime-sglang realization.

Drives the server with concurrent synthetic load and reports:
  - requests_per_sec (wall-time)
  - total_output_tokens_per_sec (wall-time)
  - llm_ttft_ms_* (mean/p50/p95/max)
  - llm_output_tokens_per_sec_* (mean/p50/p95/max)
  - llm_duration_ms_* (mean/p50/p95/max)

No correctness scorer — this is a perf characterization run, not an accuracy check.

Usage:
    python -m argus run --config examples/inference/evals/configs/bench/bench_slime_sglang.py

    # Monitor progress:
    tail -f results/eval/<run>/events.jsonl | jq .

    # Direct invocation (interactive debug mode):
    python -m rollouts.eval.run --config bench.bench_slime_sglang

To benchmark with a real dataset instead of synthetic load:
    Change WORKLOAD.kind to "sharegpt" and set WORKLOAD.dataset_path.
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

# No correctness check — bench runs care only about throughput/latency metrics.
_no_op_scorer = FunctionScorer(lambda attempt, _ctx: Score(metrics=()))

MODEL = "Qwen/Qwen3-0.6B"
PORT = 30000

# ---------------------------------------------------------------------------
# Workload — swap to make_sharegpt_tasks for real data
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

# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

endpoint = OwnedEndpoint(
    spec="slime-sglang",
    model=MODEL,
    cuda_device_ids=(0,),
    port=PORT,
    capabilities=EndpointCapabilities(weight_sync=None),
    mem_fraction=0.6,
    startup_timeout=300.0,
    max_tokens=WORKLOAD.output_len,
    extra_params={"chat_template_kwargs": {"enable_thinking": False}},
)

# ---------------------------------------------------------------------------
# Eval task — no scorer, this is a perf run
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
    output=EvalOutputConfig(experiment_name="bench_slime_sglang"),
    hardware=hardware,
    summary_distribution_percentiles=SUMMARY_DISTRIBUTION_PERCENTILES,
)
