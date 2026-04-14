"""Throughput/latency benchmark for the slime-sglang realization over SSH.

Drives the server with concurrent synthetic load and reports:
  - requests_per_sec (wall-time)
  - total_output_tokens_per_sec (wall-time)
  - llm_ttft_ms_* (mean/p50/p95/max)
  - llm_output_tokens_per_sec_* (mean/p50/p95/max)
  - llm_duration_ms_* (mean/p50/p95/max)

Primary metrics to watch in `report.json`:
  - requests_per_sec
  - total_output_tokens_per_sec
  - llm_duration_ms_p50
  - llm_duration_ms_p95
  - llm_output_tokens_per_sec_mean

No correctness scorer — this is a perf characterization run, not an accuracy check.

Usage:
    python -m argus run --config examples/inference/evals/configs/bench/bench_slime_sglang_ssh.py

    # Inspect the benchmark summary:
    jq '.summary_metrics | {
      requests_per_sec,
      total_output_tokens_per_sec,
      llm_duration_ms_p50,
      llm_duration_ms_p95,
      llm_output_tokens_per_sec_mean
    }' results/eval/<run>/report.json
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

hardware = HardwareConfig(
    provider="ssh",
    ssh="ubuntu@146.88.195.10:2222",
    ssh_key_path="~/.ssh/id_ed25519",
    gpu_type="B200",
    gpu_count=2,
    use_torchrun=False,
    deps=DepsConfig(
        bootstrap_commands=(
            "set -euo pipefail && "
            "uv pip install --python .venv/bin/python "
            "torch transformers accelerate fastapi uvicorn "
            "'sglang[all] @ git+https://github.com/sgl-project/sglang.git@main#subdirectory=python'",
        ),
    ),
)

endpoint = OwnedEndpoint(
    spec="slime-sglang",
    model=MODEL,
    cuda_device_ids=(0, 1),
    port=PORT,
    capabilities=EndpointCapabilities(weight_sync=None),
    mem_fraction=0.6,
    startup_timeout=300.0,
    max_tokens=WORKLOAD.output_len,
    extra_params={"chat_template_kwargs": {"enable_thinking": False}},
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
    output=EvalOutputConfig(experiment_name="bench_slime_sglang_ssh"),
    hardware=hardware,
    summary_distribution_percentiles=SUMMARY_DISTRIBUTION_PERCENTILES,
)
