"""Eval config for the GLM-4.7-Flash assignment target.

This is the eval graders run against student submissions. It uses OwnedEndpoint
to manage the server lifecycle - students just provide a server that implements
the skeleton_server.py contract, and this config handles start/stop.

Hardware: requires multi-GPU (GLM-4.7-Flash is 30B MoE, ~60GB BF16).
device_map="auto" in the gold baseline distributes naively across available GPUs.
Student implementations should target proper expert parallelism.

Two configs here:
    eval_task_correctness  - reverse text, checks output quality (fast, cheap)
    eval_task_throughput   - throughput/latency benchmark (TODO: wire into benchmark harness)

Usage:
    # Against the gold server (naive HF baseline)
    python -m rollouts.eval.run --config examples/inference/eval_gold_server.py --server gold

    # Against a student server
    python -m rollouts.eval.run --config examples/inference/eval_gold_server.py --server /path/to/student_server.py
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
from rollouts.training.types import AttemptResult

MODEL = "zai-org/GLM-4.7-Flash"
PORT = 30001

# ---------------------------------------------------------------------------
# Hardware - 4x A100 for GLM-4.7-Flash (30B MoE, ~60GB BF16)
# ---------------------------------------------------------------------------

hardware = HardwareConfig(
    provider="modal",
    gpu_type="A100",
    gpu_count=4,
    use_torchrun=False,  # single inference server process, not distributed training
    deps=DepsConfig(
        bootstrap_commands=(
            "~/.local/bin/uv pip install --python /opt/venvs/rollouts/bin/python torch transformers accelerate uvicorn fastapi",
        ),
    ),
)

# ---------------------------------------------------------------------------
# Endpoint - we own the server lifecycle
# Swap launch_module to point at a student implementation:
#   launch_module="rollouts.inference.skeleton_server"
# or keep "rollouts.inference.gold_server" for the naive HF baseline.
# ---------------------------------------------------------------------------

endpoint = OwnedEndpoint(
    spec="custom-http",
    model=MODEL,
    cuda_device_ids=(0, 1, 2, 3),
    port=PORT,
    capabilities=EndpointCapabilities(weight_sync=None),
    launch_module="rollouts.inference.gold_server",  # swap to skeleton_server to test students
    startup_timeout=600.0,  # GLM-4.7-Flash is 60GB, takes 5-10min to load on 4x A100
    max_tokens=128,  # short to avoid 120s request timeout on naive sequential baseline
)

# ---------------------------------------------------------------------------
# Tasks - reverse text (simple, deterministic, easy to verify by hand)
# ---------------------------------------------------------------------------

tasks = [
    {"text": "the quick brown fox"},
    {"text": "attention is all you need"},
    {"text": "mixture of experts"},
    {"text": "sparse activation"},
    {"text": "expert parallelism"},
    {"text": "key value cache"},
    {"text": "continuous batching"},
    {"text": "speculative decoding"},
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
                f"Output only the reversed text, nothing else.\n\nText: {text}"
            ),
        )
    ]


def score_fn(sample: AttemptResult, _context: object) -> Score:
    expected = sample.input["text"][::-1]
    response = ""
    if sample.trajectory:
        for msg in reversed(sample.trajectory.messages):
            if msg.role == "assistant":
                content = msg.content
                response = content if isinstance(content, str) else str(content)
                break

    exact = response.strip() == expected

    return Score(metrics=(Metric("exact_match", 1.0 if exact else 0.0, weight=1.0),))


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
        max_concurrent=1,  # serialize - naive HF baseline has no batching
        max_samples=len(tasks),
        max_turns=1,
        verbose=True,
    ),
    output=EvalOutputConfig(
        experiment_name="glm_47_flash_eval",
    ),
    hardware=hardware,
)
