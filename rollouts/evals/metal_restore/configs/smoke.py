"""Smoke test config - quick check that eval runs."""

from rollouts.config.tiers import EndpointConfig, OutputConfig, RunConfig

endpoint = EndpointConfig(
    model="claude-sonnet-4-20250514",
    provider="anthropic",
)

run = RunConfig(
    max_turns=10,  # Very short for smoke test
    max_concurrent=1,
    limit=1,
    verbose=True,
)

output = OutputConfig(
    experiment_name="metal_restore_smoke",
)
