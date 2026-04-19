"""Harbor v0 eval: run native rollouts agents against Terminal-Bench 2.0 tasks.

Pipeline per sample:
  1. Load the task's task.toml via Harbor's parser (handles memory="2G" etc).
  2. Read instruction.md as the user message.
  3. Start a HarborEnvironment (Docker container from task's environment/).
  4. Let the rollouts agent drive it with standard coding tools.
  5. After the agent stops, invoke Harbor's Verifier against the running
     container — uploads tests/, runs bash /tests/test.sh, reads
     /logs/verifier/reward.txt (or .json) back.
  6. Score comes from the Verifier's reward. Agent's self-verification is
     just a soft signal; Harbor's verifier is authoritative.

Tasks are JIT-cloned from github.com/harbor-framework/terminal-bench-2 at a
pinned commit. See prepare.py.

This eval requires harbor installed out-of-band:
    uv pip install 'harbor @ git+https://github.com/laude-institute/harbor.git@e0fcdc2'
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from rollouts.core import Message
from rollouts.core.eval import Metric, Score
from rollouts.environments.harbor_environment import (
    HarborEnvironment,
    _aio_in_thread,
    parse_harbor_host_config,
)
from rollouts.eval_runner import EvalSpec
from rollouts.training.scoring import FunctionScorer

from .prepare import TB2TaskRef, get_task

logger = logging.getLogger(__name__)

_HARBOR_HOST_KEY = "harbor_host"


def _resolve_scoring_harbor_environment(sample: Any, context: Any) -> HarborEnvironment | None:
    """Resolve the live HarborEnvironment for scoring.

    The eval runtime threads the final live environment through
    ScoringContext.environment. Older sample shapes may also carry an
    `environment` attribute, so keep that as a compatibility fallback.
    """
    context_env = getattr(context, "environment", None)
    if isinstance(context_env, HarborEnvironment):
        return context_env

    sample_env = getattr(sample, "environment", None)
    if isinstance(sample_env, HarborEnvironment):
        return sample_env

    return None


async def _run_harbor_verifier(verifier: Any) -> Any:
    """Run Harbor's asyncio-native verifier from the Trio scoring loop.

    Harbor's environment backends use asyncio internally. Our eval runtime is
    Trio-based. Tool execution already crosses that boundary through
    `_aio_in_thread`; scoring must do the same or Harbor's verifier ends up
    awaiting backend calls on the wrong event loop.
    """

    return await _aio_in_thread(lambda: verifier.verify())


# ── Per-sample environment construction ─────────────────────────────────────


async def make_environment(sample_data: dict[str, Any]) -> HarborEnvironment:
    """Construct a HarborEnvironment for one TB2 task.

    sample_data carries `task_id`. We clone (or use cached) TB2 tasks,
    parse the task.toml via Harbor's own validator (so memory="2G" etc.
    translate correctly), then hand the resolved fields to
    HarborEnvironment.create.
    """
    # Lazy import — keeps the module loadable without Harbor for test/CI.
    from harbor.models.task.config import TaskConfig

    task_id = sample_data["task_id"]
    task: TB2TaskRef = get_task(task_id)

    task_toml_text = task.task_toml.read_text()
    task_config = TaskConfig.model_validate_toml(task_toml_text)
    env_cfg = task_config.environment

    # Per-sample session id so Harbor's resource naming is unique.
    session_id = f"{task_id}__{uuid.uuid4().hex[:8]}"
    host = parse_harbor_host_config(sample_data.get(_HARBOR_HOST_KEY))

    return await HarborEnvironment.create(
        task_dir=task.environment_dir,
        environment_name=task_id,
        session_id=session_id,
        working_dir=sample_data.get("working_dir", "/app"),
        cpus=env_cfg.cpus,
        memory_mb=env_cfg.memory_mb,
        storage_mb=env_cfg.storage_mb,
        docker_image=env_cfg.docker_image,
        host=host,
        tools="full",
    )


# ── Prompt ──────────────────────────────────────────────────────────────────


def prepare_messages(sample_data: dict[str, Any]) -> list[Message]:
    """Read instruction.md as the user message; add a small system prompt."""
    task: TB2TaskRef = get_task(sample_data["task_id"])
    instruction = task.instruction_md.read_text()

    system = (
        "You are a terminal coding agent working inside a Linux container. "
        "Your tools are read, write, edit, bash, glob, grep. Use them to "
        "solve the task. When you believe the task is complete, stop — "
        "an external verifier will run the task's tests and report success "
        "or failure."
    )
    return [
        Message(role="system", content=system),
        Message(role="user", content=instruction),
    ]


# ── Scoring ─────────────────────────────────────────────────────────────────


async def score_sample(sample: Any, _context: Any) -> Score:
    """Run Harbor's Verifier against the final environment state.

    The verifier uploads tests/test.sh into the container, executes it,
    downloads /logs/verifier/ back, parses reward.txt or reward.json. We
    take the top-level "reward" as the score.

    This happens at score time (post-agent). The environment passed in is
    the agent's final state; the Verifier operates on the live container.

    TODO(parity-review): scoring is now aligned to Harbor's verifier
    contract, but trajectory/runtime parity is still only partially proven.
    We should later run the same short witness task through native Harbor,
    Prime HarborEnv, and this eval, then compare prompts, tool transcripts,
    stop reasons, and final reward to isolate any model-behavior drift.
    """
    from harbor.models.trial.paths import TrialPaths
    from harbor.verifier.verifier import Verifier

    env = _resolve_scoring_harbor_environment(sample, _context)
    if env is None:
        return Score(
            metrics=(
                Metric(
                    "passed",
                    0.0,
                    weight=1.0,
                    metadata={"error": "no live HarborEnvironment in scoring context"},
                ),
            )
        )

    sample_data = sample.problem.payload if hasattr(sample, "problem") else {}
    task_id = sample_data.get("task_id", "unknown")
    task: TB2TaskRef = get_task(task_id)

    from harbor.models.task.task import Task

    harbor_task = Task(task.task_dir)

    # Verifier wants its own trial_paths for downloading verifier output.
    with tempfile.TemporaryDirectory(prefix="rollouts-harbor-verify-") as td:
        trial_paths = TrialPaths(trial_dir=Path(td))
        if hasattr(trial_paths, "mkdir"):
            trial_paths.mkdir()

        verifier = Verifier(
            task=harbor_task,
            trial_paths=trial_paths,
            environment=env.harbor_env,
        )
        try:
            result = await _run_harbor_verifier(verifier)
        except Exception as exc:
            logger.exception("Harbor Verifier failed for %s", task_id)
            return Score(
                metrics=(
                    Metric(
                        "passed",
                        0.0,
                        weight=1.0,
                        metadata={"error": f"verifier raised: {exc}"},
                    ),
                )
            )

    rewards = getattr(result, "rewards", None) or {}
    # Harbor's convention: top-level "reward" is a float.
    reward_value = float(rewards.get("reward", 0.0))
    passed = 1.0 if reward_value >= 1.0 else 0.0

    return Score(
        metrics=(
            Metric("passed", passed, weight=1.0),
            Metric("reward", reward_value, weight=0.0, metadata={"rewards": rewards}),
        )
    )


# ── EvalSpec ────────────────────────────────────────────────────────────────


spec = EvalSpec(
    name="harbor_v0",
    prepare_messages=prepare_messages,
    scorer=FunctionScorer(score_sample),
    make_environment=make_environment,
    per_sample_environment=True,
)
