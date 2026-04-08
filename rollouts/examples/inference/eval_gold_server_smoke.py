"""Fast smoke witness for the HuggingFace gold inference server.

This is the cheap inner-loop witness for inference-engine iteration:
- owns the server lifecycle via OwnedEndpoint
- runs a small canonical reverse-text workload
- writes the usual eval artifacts plus per-call runtime telemetry

The report.json summary now includes metrics derived from eval-side LLM call
events such as:
- sample_duration_seconds_*
- llm_duration_ms_*
- llm_ttft_ms_*
- llm_tokens_in_total / llm_tokens_out_total

Usage:
    cd /Users/chiraagbalu/research/rollouts
    ../.venv/bin/python -m rollouts.eval.run --config examples/inference/eval_gold_server_smoke.py
"""

from rollouts.core import Message, Metric, Score
from rollouts.eval import (
    AgentRunSpec,
    EvalOutputConfig,
    EvalRunConfig,
    EvalTaskSpec,
)
from rollouts.eval.configs import EndpointCapabilities, OwnedEndpoint
from rollouts.training.configs import DepsConfig, HardwareConfig
from rollouts.training.scoring import FunctionScorer
from rollouts.training.types import RowAttempt

MODEL = "Qwen/Qwen3-0.6B"
PORT = 30001

hardware = HardwareConfig(
    provider="modal",
    gpu_type="A100",
    gpu_count=1,
    use_torchrun=False,
    keep_alive=True,
    deps=DepsConfig(
        bootstrap_commands=(
            "~/.local/bin/uv pip install --python /opt/venvs/rollouts/bin/python torch transformers accelerate uvicorn fastapi",
        ),
    ),
)

endpoint = OwnedEndpoint(
    spec="custom-http",
    model=MODEL,
    cuda_device_ids=(0,),
    port=PORT,
    capabilities=EndpointCapabilities(weight_sync=None),
    launch_module="rollouts.inference.gold_server",
    startup_timeout=180.0,
    max_tokens=64,
)

tasks = [
    {"text": "hello"},
    {"text": "world"},
    {"text": "attention is all you need"},
    {"text": "continuous batching"},
]


def prepare_messages(sample: dict) -> list[Message]:
    text = sample["text"]
    return [
        Message(
            role="user",
            content=(
                "Reverse the following text character-by-character. "
                "Put your answer in <answer> tags.\n\n"
                f"Text: {text}"
            ),
        )
    ]


def score_fn(sample: RowAttempt, _context: object) -> Score:
    import re

    expected = sample.input["text"][::-1]
    response = sample.response
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    parsed = match.group(1).strip() if match else response.strip()
    exact = parsed == expected
    return Score(
        metrics=(
            Metric("exact_match", 1.0 if exact else 0.0, weight=1.0),
            Metric("has_tags", 1.0 if match else 0.0, weight=0.0),
        )
    )


eval_task = EvalTaskSpec(
    tasks=tasks,
    run_spec=AgentRunSpec(
        endpoint=endpoint,
        prepare_messages=prepare_messages,
    ),
    scorer=FunctionScorer(score_fn),
    run=EvalRunConfig(
        max_concurrent=2,
        max_samples=len(tasks),
        max_turns=1,
        verbose=True,
        show_progress=True,
    ),
    output=EvalOutputConfig(
        experiment_name="gold_server_smoke_eval",
    ),
    hardware=hardware,
)
