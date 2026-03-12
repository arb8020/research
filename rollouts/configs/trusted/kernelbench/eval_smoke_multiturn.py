"""Trusted KernelBench multi-turn eval smoke config."""

from __future__ import annotations

from examples.rl.kernelbench.resources import (
    KernelBenchRolloutResources,
    KernelBenchScoringResources,
)
from examples.rl.kernelbench.scoring import KEVIN_MULTI_TURN_REWARD_WEIGHTS
from examples.rl.kernelbench.subsets import load_kernelbench_smoke_prompts
from rollouts.config_status import draft
from rollouts.core import Message
from rollouts.environments.modal_sandbox_resource import ModalSandboxResourceConfig
from rollouts.eval import EndpointConfig, EvalOutputConfig, EvalRunConfig

config_status = draft(
    "Canonical KernelBench smoke subset (ReLU + Softmax). Uses explicit Modal sandbox resources for agent execution and remote scoring workers, but is not yet stamped known-good through this config entrypoint.",
)

endpoint = EndpointConfig(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    temperature=0.7,
    max_tokens=4096,
)

run = EvalRunConfig(
    max_concurrent=1,
    max_samples=2,
    max_turns=5,
    verbose=True,
    show_progress=True,
)

output = EvalOutputConfig(
    experiment_name="kernelbench_smoke_multiturn",
)

tasks = load_kernelbench_smoke_prompts(backend="cuda")
rollout_resources = KernelBenchRolloutResources.with_modal_sandbox(
    ModalSandboxResourceConfig(
        app_name="rollouts-kernelbench",
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
    return [Message(role=msg["role"], content=msg["content"]) for msg in sample_data["messages"]]  # type: ignore[index]
