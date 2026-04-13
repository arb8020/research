"""Eval config for testing a custom inference engine via skeleton_server.py.

This is the reference eval for the 'implement an inference engine' assignment.
It uses OwnedEndpoint to manage the server lifecycle - no manual server startup
required. The harness launches the server, waits for /health, runs the eval,
then shuts down.

The eval task is simple (reverse text) so results are deterministic and easy to
verify by hand. It's a functional correctness check, not a model quality check -
the untouched skeleton returns a deterministic stub answer with fixed logprobs,
which proves the serving contract while still scoring poorly.

Usage (against a real implementation):
    # First implement generate_reply in skeleton_server.py, then:
    python -m argus run --config inference.eval_skeleton_server

    # Monitor progress (stdout/stderr are noise - use the jsonl):
    tail -f results/eval/<run>/events.jsonl | jq .

    # Direct invocation (interactive debug mode):
    python -m rollouts.eval.run --config inference.eval_skeleton_server

Usage (against a running server you started manually):
    # If you want to test against an already-running server instead:
    # Change OwnedEndpoint → ExternalEndpoint with url="http://localhost:30000/v1"

The OwnedEndpoint approach is preferred for automated testing - it owns the
full lifecycle and tears down cleanly on failure.

Student-facing assignment docs:
    examples/inference/README.md
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

# Smoke test uses Qwen3-0.6B: small enough to load fast, proves lifecycle works.
# For the real assignment eval against GLM-4.7-Flash, see eval_gold_server.py.
MODEL = "Qwen/Qwen3-0.6B"
PORT = 30001  # offset from 30000 to avoid clashing with a running sglang server

# ---------------------------------------------------------------------------
# Hardware - Modal A100, installs transformers + torch
# ---------------------------------------------------------------------------

hardware = HardwareConfig(
    provider="modal",
    gpu_type="A100",
    gpu_count=1,
    use_torchrun=False,  # single inference server process, not distributed training
    deps=DepsConfig(
        bootstrap_commands=(
            "~/.local/bin/uv pip install --python /opt/venvs/rollouts/bin/python torch transformers accelerate uvicorn fastapi",
        ),
    ),
)

# ---------------------------------------------------------------------------
# Endpoint - we own the server lifecycle
# launch_module works locally and remotely (module path survives bifrost sync)
# ---------------------------------------------------------------------------

endpoint = OwnedEndpoint(
    spec="custom-http",
    model=MODEL,
    cuda_device_ids=(0,),
    port=PORT,
    capabilities=EndpointCapabilities(weight_sync=None),
    launch_module="rollouts.inference.skeleton_server",  # students implement generate_reply here
    startup_timeout=120.0,
)

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

tasks = [
    {"text": "hello"},
    {"text": "world"},
    {"text": "inference engine"},
    {"text": "attention is all you need"},
]

# ---------------------------------------------------------------------------
# Eval functions
# ---------------------------------------------------------------------------


def prepare_messages(sample: dict) -> list[Message]:
    text = sample["text"]
    return [
        Message(
            role="user",
            content=(
                f"Reverse the following text character-by-character. "
                f"Put your answer in <answer> tags.\n\nText: {text}"
            ),
        )
    ]


def score_fn(sample: RowAttempt, _context: object) -> Score:
    import re

    expected = sample.input["text"][::-1]
    response = ""
    if sample.trajectory:
        for msg in reversed(sample.trajectory.messages):
            if msg.role == "assistant":
                content = msg.content
                response = content if isinstance(content, str) else str(content)
                break

    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    parsed = match.group(1).strip() if match else response.strip()
    exact = parsed == expected

    return Score(
        metrics=(
            Metric("exact_match", 1.0 if exact else 0.0, weight=1.0),
            Metric("has_tags", 1.0 if match else 0.0, weight=0.0),
        )
    )


scorer = FunctionScorer(score_fn)

# ---------------------------------------------------------------------------
# Eval config
# ---------------------------------------------------------------------------

eval_task = EvalTaskSpec(
    tasks=tasks,
    run_spec=AgentRunSpec(
        endpoint=endpoint,
        prepare_messages=prepare_messages,
    ),
    scorer=scorer,
    run=EvalRunConfig(
        max_concurrent=2,
        max_samples=len(tasks),
        max_turns=1,
        verbose=True,
    ),
    output=EvalOutputConfig(
        experiment_name="skeleton_server_eval",
    ),
    hardware=hardware,
)
