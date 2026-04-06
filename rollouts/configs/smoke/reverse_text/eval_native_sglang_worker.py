"""Smoke eval using a named actor inference worker with the native agent loop.

This is the first eval-side witness for the worker-topology model:
- allocate hardware once
- define an actor inference worker on that allocation
- lower the actor worker into today's eval runner surface

TODO(judge-worker): add a second judge inference worker and use it in the
scoring path once eval-side multi-role inference is wired through natively.
"""

from __future__ import annotations

from examples.rl.reverse_text.base_config import SYSTEM_PROMPT, parse_reversed_text
from rollouts.config_status import draft
from rollouts.core import Message, Metric, Score
from rollouts.eval import (
    AgentRunSpec,
    EndpointCapabilities,
    EvalOutputConfig,
    EvalRunConfig,
    EvalTaskSpec,
    OwnedEndpoint,
)
from rollouts.training.configs import (
    DepsConfig,
    HardwareConfig,
    InferenceConfig,
    InferenceRoleBinding,
    InferenceWorkerConfig,
    WorkerTopologyConfig,
)
from rollouts.training.scoring import FunctionScorer
from rollouts.training.types import AttemptResult


def prepare_messages(sample: dict[str, str]) -> list[Message]:
    text = sample["text"]
    return [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(
            role="user",
            content=(
                "Reverse the following text character-by-character. "
                "Put your answer in <reversed_text> tags.\n\n"
                f"Text to reverse: {text}"
            ),
        ),
    ]


def reverse_text_eval_score_fn(sample: AttemptResult, _context: object) -> Score:
    expected = sample.input["text"][::-1]
    response = sample.response
    parsed = parse_reversed_text(response)
    normalized = parsed if parsed is not None else response.strip().strip("\"'")
    exact_match = normalized == expected
    similarity = 1.0 if exact_match else 0.0
    return Score(
        metrics=(
            Metric("similarity", similarity, weight=1.0),
            Metric("exact_match", 1.0 if exact_match else 0.0, weight=0.0),
        )
    )


# Keep this smoke on-distribution so it remains a lifecycle witness for
# OwnedEndpoint rather than pretending to be a strong model-quality check.
# In local manual probes, this SFT served requests correctly but missed exact
# reversal on some simple handcrafted strings like "hello world" and
# "prime intellect". Use representative examples from Prime's dataset here.
SMOKE_TASKS = [
    {"text": "The community in Bruck was merged into it"},
    {"text": "In 1891 the community inaugurated its own cemetery"},
]


config_status = draft(
    "Smoke eval for the worker-topology actor path. Intended as the first native "
    "inference-worker eval witness before adding judge-worker bindings."
)

worker_topology = WorkerTopologyConfig(
    hardware=HardwareConfig(
        provider="modal",
        gpu_type="A100",
        gpu_count=1,
        deps=DepsConfig(
            bootstrap_commands=(
                "mkdir -p /tmp/rollouts-sglang-deps",
                """python3 -c "from pathlib import Path; Path('/tmp/rollouts-sglang-deps/pyproject.toml').write_text('[project]\\nname = \\"rollouts-eval-sglang-worker\\"\\nversion = \\"0.0.1\\"\\nrequires-python = \\"==3.12.*\\"\\ndependencies = [\\n  \\"sglang[all] @ git+https://github.com/sgl-project/sglang.git@main#subdirectory=python\\",\\n]\\n\\n[tool.uv.sources]\\ntorch = { index = \\"pytorch-cu124\\" }\\n\\n[[tool.uv.index]]\\nname = \\"pytorch-cu124\\"\\nurl = \\"https://download.pytorch.org/whl/cu124\\"\\nexplicit = true\\n')" """,
                "~/.local/bin/uv pip compile /tmp/rollouts-sglang-deps/pyproject.toml --output-file /tmp/rollouts-sglang-deps/requirements.txt",
                "~/.local/bin/uv pip install --compile-bytecode --python /opt/venvs/rollouts/bin/python -r /tmp/rollouts-sglang-deps/requirements.txt",
            ),
        ),
    ),
    inference_workers=(
        InferenceWorkerConfig(
            worker_id="actor",
            model="PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT",
            inference=InferenceConfig(
                cuda_device_ids=(0,),
                port=30000,
                mem_fraction=0.6,
            ),
        ),
    ),
    role_bindings=(InferenceRoleBinding(role="actor", worker_id="actor"),),
)

actor_worker = worker_topology.get_worker_for_role("actor")
endpoint = OwnedEndpoint(
    spec=actor_worker.inference.spec,
    model=actor_worker.model,
    cuda_device_ids=actor_worker.inference.cuda_device_ids,
    port=actor_worker.inference.port,
    mem_fraction=actor_worker.inference.mem_fraction,
    startup_timeout=actor_worker.inference.startup_timeout,
    capabilities=EndpointCapabilities(weight_sync=None),
)

eval_task = EvalTaskSpec(
    tasks=SMOKE_TASKS,
    run_spec=AgentRunSpec(
        endpoint=endpoint,
        prepare_messages=prepare_messages,
    ),
    scorer=FunctionScorer(reverse_text_eval_score_fn),
    run=EvalRunConfig(
        max_concurrent=1,
        max_samples=2,
        max_turns=1,
        verbose=True,
        show_progress=True,
    ),
    output=EvalOutputConfig(
        experiment_name="smoke_reverse_text_native_sglang_worker",
    ),
    hardware=worker_topology.hardware,
)
