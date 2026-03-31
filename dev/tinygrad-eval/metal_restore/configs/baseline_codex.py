"""Baseline config - Codex without special tooling."""

from rollouts.config.tiers import EndpointConfig, OutputConfig, RunConfig

endpoint = EndpointConfig(
    model="gpt-5.1-codex",
    provider="openai",
)

run = RunConfig(
    max_turns=100,
    max_concurrent=1,
    limit=1,
    verbose=True,
)

output = OutputConfig(
    experiment_name="metal_restore_baseline_codex",
)
