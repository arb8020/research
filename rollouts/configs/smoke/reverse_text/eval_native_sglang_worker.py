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
    EvalOutputConfig,
    EvalRunConfig,
    EvalTaskSpec,
    endpoint_and_server_for_role,
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
            model="Qwen/Qwen2.5-0.5B-Instruct",
            inference=InferenceConfig(
                cuda_device_ids=(0,),
                port=30000,
                mem_fraction=0.6,
            ),
        ),
    ),
    role_bindings=(InferenceRoleBinding(role="actor", worker_id="actor"),),
)

endpoint, server = endpoint_and_server_for_role(worker_topology, "actor")

eval_task = EvalTaskSpec(
    tasks=[
        {"text": "hello world"},
        {"text": "prime intellect"},
    ],
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
    server=server,
)
