"""Tau2-bench v0 eval: run rollouts agents against tau2 customer-service tasks.

Pipeline per sample:
  1. `make_environment(row)` builds a fresh Tau2Environment (fresh domain
     database per task). Reads `user_endpoint` from the row.
  2. `prepare_messages(row)` (async) builds the initial trajectory with
     tau2-canonical shape: [system AGENT_INSTRUCTION, assistant
     DEFAULT_FIRST_AGENT_MESSAGE, user (sim's first reply)]. This requires
     I/O — calling the user-sim endpoint once — which is why it's async.
  3. The runtime drives the agent loop. Tau2Environment.on_assistant_message
     keeps tau2's user_state in sync, drives subsequent user-sim turns, and
     applies tau2's stop-token classifiers.
  4. After the agent stops, `Tau2Environment.score(trajectory)` delegates
     to tau2's official `evaluate_simulation`.

Requires `tau2` installed out-of-band:
    uv pip install 'tau2 @ git+https://github.com/sierra-research/tau2-bench'

Row schema produced by tau2_v0.prepare.build_sample_rows:
    {
      "task_id": str,
      "domain": "airline" | "retail" | "telecom",
      "task_json": str,
      "user_endpoint": {
          "model": str,                # e.g. "openai/gpt-4o-mini"
          "base_url": str,             # e.g. "https://api.openai.com/v1"
          "api_format": str,           # e.g. "openai-completions"
          "api_key_env": str,          # name of env var holding the key
      },
      "max_steps": int (optional),
      "max_errors": int (optional),
    }

`user_endpoint` lives on the row (not a closure on the eval module) so that
the row is the single source of truth: every per-sample input — including
which model plays the user — is reproducible from the dataset.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from rollouts.core import Endpoint, Message
from rollouts.core.eval import Metric, Score
from rollouts.environments.tau2_environment import Tau2Environment
from rollouts.eval_runner import EvalSpec
from rollouts.training.scoring import FunctionScorer

logger = logging.getLogger(__name__)


def _user_endpoint_from_row(sample_data: dict[str, Any]) -> Endpoint:
    """Build the user-simulator Endpoint from the row's `user_endpoint`
    config block. The API key is resolved from an env var named in the row.
    """
    cfg = sample_data["user_endpoint"]
    api_key_env = cfg["api_key_env"]
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        logger.warning(
            "User-sim env var %s is not set; user simulator calls will fail",
            api_key_env,
        )
    return Endpoint(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_format=cfg["api_format"],
        api_key=api_key,
    )


def _persona_config_from_row(sample_data: dict[str, Any]) -> Any:
    """Decode `persona_config` from the row (or None for tau2 defaults)."""
    raw = sample_data.get("persona_config")
    if raw is None:
        return None
    from tau2.data_model.persona import PersonaConfig

    return PersonaConfig.model_validate(raw)


async def make_environment(sample_data: dict[str, Any]) -> Tau2Environment:
    """Construct a Tau2Environment for one task.

    The same `user_endpoint` and `persona_config` blocks from the row drive
    both this construction AND `prepare_messages` — single source of truth.
    """
    from tau2.data_model.tasks import Task as Tau2Task

    domain = sample_data["domain"]
    task = Tau2Task.model_validate_json(sample_data["task_json"])

    return await Tau2Environment.create(
        domain=domain,
        task=task,
        user_endpoint=_user_endpoint_from_row(sample_data),
        max_steps=sample_data.get("max_steps", 200),
        max_errors=sample_data.get("max_errors", 10),
        persona_config=_persona_config_from_row(sample_data),
    )


async def prepare_messages(sample_data: dict[str, Any]) -> list[Message]:
    """Async bootstrap of the initial trajectory in tau2's canonical shape.

    tau2's official Orchestrator starts each task with a hardcoded
    assistant greeting (`DEFAULT_FIRST_AGENT_MESSAGE`), then asks the user
    simulator to reply. The agent's first LLM call happens in response to
    that user-sim reply.

    rollouts' runtime always starts with an agent LLM call, so to match
    tau2's trajectory shape we pre-seed the trajectory here:
        [0] system: AGENT_INSTRUCTION (+ domain policy via env)
        [1] assistant: DEFAULT_FIRST_AGENT_MESSAGE
        [2] user: user-sim's reply to the greeting

    This requires one I/O call (the user-sim rollout) — that's why
    prepare_messages is async. The user-sim endpoint comes from the same
    `user_endpoint` config in the row that make_environment uses.

    The env reconstructs `_user_state` from this seeded trajectory on its
    first `on_assistant_message` call (so we don't duplicate the bootstrap
    conversation in two places).
    """
    from tau2.agent.llm_agent import AGENT_INSTRUCTION
    from tau2.data_model.message import (
        AssistantMessage as Tau2AssistantMessage,
    )
    from tau2.data_model.tasks import Task as Tau2Task
    from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
    from tau2.user.user_simulator import UserSimulator

    from rollouts.agents import Actor, rollout
    from rollouts.core import Trajectory

    task = Tau2Task.model_validate_json(sample_data["task_json"])
    persona_config = _persona_config_from_row(sample_data)

    # Throwaway UserSimulator just to build the canonical system prompt and
    # role-flipped message list. Same persona_config as the live env, so
    # the bootstrap user-sim call uses an identical system prompt to the
    # ones the env will issue on subsequent turns.
    sim = UserSimulator(
        llm="dummy",
        instructions=str(task.user_scenario),
        tools=None,
        persona_config=persona_config,
    )
    user_state = sim.get_init_state()

    # Seed user_state with the canonical assistant greeting so flip_roles
    # produces the right context for the user-sim's first reply.
    greeting = Tau2AssistantMessage(
        role="assistant",
        content=DEFAULT_FIRST_AGENT_MESSAGE.content,
    )
    user_state.messages.append(greeting)

    # Translate tau2 messages → rollouts and call rollout(). We import the
    # translator from the env module to keep the conversion logic in one
    # place (tau2 messages → rollouts text-only messages).
    from rollouts.environments.tau2_environment import _tau2_messages_to_rollouts

    rollouts_msgs = _tau2_messages_to_rollouts(
        user_state.flip_roles(),
        user_state.system_messages,
    )
    actor = Actor(
        trajectory=Trajectory(messages=rollouts_msgs),
        endpoint=_user_endpoint_from_row(sample_data),
        tools=[],
    )

    async def _silent(_event: Any) -> None:
        return None

    actor = await rollout(actor, on_chunk=_silent)
    last = actor.trajectory.messages[-1] if actor.trajectory.messages else None
    # rollouts replies may come back as plain str or as a list of
    # ContentBlock; the env's _message_text helper handles both.
    from rollouts.environments.tau2_environment import _message_text

    sim_reply_text = _message_text(last) if last is not None else ""
    if not sim_reply_text:
        # Defensive: if the user sim returned nothing usable, downstream
        # provider calls will reject the empty user message. Surface this
        # as a hard failure here rather than a confusing 400 later.
        raise RuntimeError(
            "tau2 user simulator returned empty content during prepare_messages "
            "bootstrap; check user_endpoint config in the dataset row."
        )

    return [
        Message(role="system", content=AGENT_INSTRUCTION),
        Message(role="assistant", content=DEFAULT_FIRST_AGENT_MESSAGE.content),
        Message(role="user", content=sim_reply_text),
    ]


async def score_sample(sample: Any, _context: Any) -> Score:
    """Delegate to Tau2Environment.score.

    The environment holds the parallel tau2 trajectory needed by tau2's
    evaluator; the rollouts Trajectory is its own representation.
    """
    env = getattr(sample, "environment", None)
    if env is None or not isinstance(env, Tau2Environment):
        return Score(
            metrics=(
                Metric(
                    "passed",
                    0.0,
                    weight=1.0,
                    metadata={"error": "no Tau2Environment on sample"},
                ),
            )
        )
    trajectory = getattr(sample, "trajectory", None)
    return await env.score(trajectory)


spec = EvalSpec(
    name="tau2_v0",
    prepare_messages=prepare_messages,
    scorer=FunctionScorer(score_sample),
    make_environment=make_environment,
    per_sample_environment=True,
    # tau2 IS a tool environment, but turns where the agent sends plain text
    # (no tool call) are normal — Tau2Environment.on_assistant_message has
    # already injected the user-simulator's reply into the trajectory, so the
    # loop should continue. has_tools=False routes the no-tool path to a
    # noop instead of TASK_COMPLETED. Stops come from Tau2Environment itself
    # (agent ###STOP###, user-sim stop tokens, step/error caps).
    has_tools=False,
)
