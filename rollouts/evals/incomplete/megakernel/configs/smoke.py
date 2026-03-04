from rollouts.config.tiers import EndpointConfig, OutputConfig, RunConfig

endpoint = EndpointConfig(
    model="claude-sonnet-4-20250514",
    provider="anthropic",
)

run = RunConfig(
    max_turns=30,
    max_concurrent=1,
    limit=1,
    verbose=True,
    show_progress=True,
)

output = OutputConfig(
    experiment_name="megakernel_smoke",
)
