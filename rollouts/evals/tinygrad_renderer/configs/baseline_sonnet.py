"""Baseline config - Sonnet without special tooling."""

from rollouts.config.tiers import EndpointConfig, OutputConfig, RunConfig

endpoint = EndpointConfig(
    model="claude-sonnet-4-20250514",
    provider="anthropic",
    temperature=0.0,
)

run = RunConfig(
    max_turns=100,
    max_concurrent=1,
    limit=1,
    verbose=True,
)

output = OutputConfig(
    experiment_name="metal_restore_baseline_sonnet",
)
