"""Eval config for testing the student inference engine on B200 over SSH.

Same correctness eval as eval_skeleton_server_b200_ssh.py but targets
student_server.py instead of skeleton_server.py.

Usage:
    .venv/bin/python -m argus run --config rollouts/examples/inference/evals/configs/eval_student_server_b200_ssh.py --force-deploy-committed

    # Monitor progress:
    tail -f results/eval/<run>/events.jsonl | jq .

    # Direct invocation (interactive debug mode):
    python -m rollouts.eval.run --config inference.eval_student_server_b200_ssh
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
    launch_module="rollouts.inference.student",
    startup_timeout=120.0,
    extra_launch_args=("--trace-path", "__output_dir__/engine_trace.jsonl"),
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
        experiment_name="student_server_eval_b200_ssh",
    ),
    hardware=hardware,
)
