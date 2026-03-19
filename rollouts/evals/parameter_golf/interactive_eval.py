"""Interactive Parameter Golf scaffold for Claude Code / Codex eval launch."""

from __future__ import annotations

from rollouts.eval import AgentRunSpec, EvalOutputConfig, EvalRunConfig, MaxTurnsStop

from evals.parameter_golf.common import default_sample, make_environment, prepare_messages

tasks = [default_sample()]

run = EvalRunConfig(
    max_concurrent=1,
    max_samples=1,
    stop_handler=MaxTurnsStop(90),
    verbose=True,
    show_progress=True,
)

run_spec = AgentRunSpec(
    prepare_messages=prepare_messages,
    environment_factory=make_environment,
)

output = EvalOutputConfig(experiment_name="parameter_golf_interactive")
