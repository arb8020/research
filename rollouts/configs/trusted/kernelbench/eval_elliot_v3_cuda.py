"""Trusted KernelBench-v3 eval config using the curated Elliot subset."""

from __future__ import annotations

import os

from examples.rl.kernelbench.resources import (
    KernelBenchRolloutResources,
    KernelBenchScoringResources,
)
from examples.rl.kernelbench.scoring import KEVIN_MULTI_TURN_REWARD_WEIGHTS
from examples.rl.kernelbench.subsets import load_kernelbench_v3_elliot_prompts
from rollouts.config_status import draft
from rollouts.core import Message
from rollouts.environments.modal_sandbox_resource import ModalSandboxResourceConfig
from rollouts.eval import EndpointConfig, EvalOutputConfig, EvalRunConfig

config_status = draft(
    "Curated KernelBench-v3 Elliot subset for CUDA eval. Requires KERNELBENCH_V3_ROOT "
    "to point at a local KernelBench-v3 checkout and has not yet been stamped known-good.",
)

_kernelbench_v3_root = os.environ.get("KERNELBENCH_V3_ROOT")
if not _kernelbench_v3_root:
    raise ValueError(
        "KERNELBENCH_V3_ROOT must point at a local KernelBench-v3 checkout for this config."
    )

endpoint = EndpointConfig(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    temperature=0.7,
    max_tokens=4096,
)

run = EvalRunConfig(
    max_concurrent=4,
    max_samples=41,
    max_turns=10,
    verbose=True,
    show_progress=True,
)

output = EvalOutputConfig(
    experiment_name="kernelbench_v3_elliot_cuda",
)

tasks = load_kernelbench_v3_elliot_prompts(
    root_path=_kernelbench_v3_root,
    backend="cuda",
)
rollout_resources = KernelBenchRolloutResources.with_modal_sandbox(
    ModalSandboxResourceConfig(
        app_name="rollouts-kernelbench-v3",
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
