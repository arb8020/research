"""Full eval config - run all models."""

from rollouts.config.tiers import EndpointConfig, OutputConfig, RunConfig

endpoint = EndpointConfig(
    model="claude-sonnet-4-20250514",
    provider="anthropic",
)

run = RunConfig(
    max_turns=100,
    max_concurrent=4,
    limit=None,  # All tasks
    verbose=True,
    show_progress=True,
)

output = OutputConfig(
    experiment_name="functional_extractor_full",
)
