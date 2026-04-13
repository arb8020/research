"""Throughput/latency benchmark for the qed-vllm realization.

Drives the server with concurrent synthetic load and reports:
  - requests_per_sec (wall-time)
  - total_output_tokens_per_sec (wall-time)
  - llm_ttft_ms_* (mean/p50/p95/max)
  - llm_output_tokens_per_sec_* (mean/p50/p95/max)
  - llm_duration_ms_* (mean/p50/p95/max)

No correctness scorer — this is a perf characterization run, not an accuracy check.

Usage:
    python -m argus run --config examples/inference/evals/configs/bench/bench_qed_vllm.py

    # Monitor progress:
    tail -f results/eval/<run>/events.jsonl | jq .

    # Direct invocation (interactive debug mode):
    python -m rollouts.eval.run --config bench.bench_qed_vllm

To benchmark with a real dataset instead of synthetic load:
    Swap make_random_tasks(...) for make_sharegpt_tasks(path=..., num_prompts=200)
    and keep everything else the same.
"""

from examples.inference.bench_workload_lib import make_random_tasks, prepare_bench_messages
from rollouts.core import Score
from rollouts.eval import AgentRunSpec, EvalOutputConfig, EvalRunConfig, EvalTaskSpec
from rollouts.eval.configs import EndpointCapabilities, OwnedEndpoint
from rollouts.training.configs import DepsConfig, HardwareConfig
from rollouts.training.scoring import FunctionScorer

# No correctness check — bench runs care only about throughput/latency metrics.
_no_op_scorer = FunctionScorer(lambda attempt, _ctx: Score(metrics=()))

MODEL = "Qwen/Qwen3-0.6B"
PORT = 30001

# ---------------------------------------------------------------------------
# Workload — swap to make_sharegpt_tasks for real data
# ---------------------------------------------------------------------------

INPUT_LEN = 512
OUTPUT_LEN = 256
NUM_PROMPTS = 200

tasks = make_random_tasks(
    num_prompts=NUM_PROMPTS,
    input_len=INPUT_LEN,
    output_len=OUTPUT_LEN,
    seed=42,
)

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
            "torch 'vllm>=0.13.0,<0.14.0'",
        ),
    ),
)

# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

endpoint = OwnedEndpoint(
    spec="qed-vllm",
    model=MODEL,
    cuda_device_ids=(0,),
    port=PORT,
    capabilities=EndpointCapabilities(weight_sync=None),
    startup_timeout=300.0,
    max_tokens=OUTPUT_LEN,
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
        max_concurrent=16,
        max_samples=NUM_PROMPTS,
        max_turns=1,
        verbose=False,
        show_progress=True,
    ),
    output=EvalOutputConfig(experiment_name="bench_qed_vllm"),
    hardware=hardware,
)
