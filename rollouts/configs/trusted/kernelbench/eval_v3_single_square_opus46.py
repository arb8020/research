"""Single-problem KernelBench-v3 CUDA eval using Claude Opus 4.6."""

from __future__ import annotations

import os

from examples.rl.kernelbench.resources import (
    KernelBenchRolloutResources,
    KernelBenchScoringResources,
)
from examples.rl.kernelbench.scoring import KEVIN_MULTI_TURN_REWARD_WEIGHTS
from examples.rl.kernelbench.subsets import select_problem_names
from examples.rl.kernelbench.dataset import load_kernelbench_v3_prompts
from rollouts.config_status import draft
from rollouts.core import Message
from rollouts.environments.modal_sandbox_resource import ModalSandboxResourceConfig
from rollouts.eval import EndpointConfig, EvalOutputConfig, EvalRunConfig

config_status = draft(
    "Single-problem end-to-end KernelBench-v3 CUDA run on Square_matrix_multiplication_. "
    "Requires KERNELBENCH_V3_ROOT and is intended for real run validation."
)

_kernelbench_v3_root = os.environ.get("KERNELBENCH_V3_ROOT")
if not _kernelbench_v3_root:
    raise ValueError(
        "KERNELBENCH_V3_ROOT must point at a local KernelBench-v3 checkout for this config."
    )

endpoint = EndpointConfig(
    provider="anthropic",
    model="claude-opus-4-6",
    temperature=0.7,
    max_tokens=4096,
)

run = EvalRunConfig(
    max_concurrent=1,
    max_samples=1,
    max_turns=10,
    verbose=True,
    show_progress=True,
)

output = EvalOutputConfig(
    experiment_name="kernelbench_v3_single_square_opus46",
)

tasks = select_problem_names(
    load_kernelbench_v3_prompts(
        root_path=_kernelbench_v3_root,
        backend="cuda",
        levels=[1],
    ),
    ("Square_matrix_multiplication_",),
)

rollout_resources = KernelBenchRolloutResources.with_modal_sandbox(
    ModalSandboxResourceConfig(
        app_name="rollouts-kernelbench-v3-square-opus46",
        gpu="A100",
    ),
    backend="cuda",
    max_turns=run.max_turns,
)
scoring_resources = KernelBenchScoringResources.metadata_only(
    reward_weights=KEVIN_MULTI_TURN_REWARD_WEIGHTS,
)
sample_scorer = scoring_resources.scorer
per_sample_environment = True
make_environment = rollout_resources


def prepare_messages(sample_data: dict[str, object]) -> list[Message]:
    return [Message(role=msg["role"], content=msg["content"]) for msg in sample_data["messages"]]
