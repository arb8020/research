"""Eval config for testing a custom inference engine via skeleton_server.py on B200 over SSH.

Same correctness eval as eval_skeleton_server.py but targets the B200 SSH host
instead of Modal A100. Use this to verify your inference engine implementation
on real B200 hardware.

Usage:
    python -m argus run --config examples/inference/evals/configs/eval_skeleton_server_b200_ssh.py

    # Monitor progress:
    tail -f results/eval/<run>/events.jsonl | jq .

    # Direct invocation (interactive debug mode):
    python -m rollouts.eval.run --config inference.eval_skeleton_server_b200_ssh
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

# ---------------------------------------------------------------------------
# Hardware - B200 over SSH
# ---------------------------------------------------------------------------

hardware = HardwareConfig(
    provider="ssh",
    ssh="ubuntu@146.88.195.10:2222",
    ssh_key_path="~/.ssh/id_ed25519",
    gpu_type="B200",
    gpu_count=1,
    use_torchrun=False,
    deps=DepsConfig(
        bootstrap_commands=(
            "uv pip install --python .venv/bin/python "
            "torch transformers accelerate fastapi uvicorn",
        ),
    ),
)

# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

endpoint = OwnedEndpoint(
    spec="custom-http",
    model=MODEL,
    cuda_device_ids=(0,),
    port=PORT,
    capabilities=EndpointCapabilities(weight_sync=None),
    launch_module="rollouts.inference.skeleton_server",
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
        experiment_name="skeleton_server_eval_b200_ssh",
    ),
    hardware=hardware,
)
