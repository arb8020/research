"""Smoke test config - run 1 small model to verify setup works."""

from rollouts.config.tiers import EndpointConfig, OutputConfig, RunConfig

endpoint = EndpointConfig(
    model="claude-sonnet-4-20250514",
    provider="anthropic",
)

run = RunConfig(
    max_turns=50,
    max_concurrent=1,
    limit=1,
    verbose=True,
    show_progress=True,
)

output = OutputConfig(
    experiment_name="functional_extractor_smoke",
)

# Use SmolLM2-135M for smoke test (smallest, fastest)
tasks_override = [
    {
        "task_id": "smollm2-135m",
        "model_name": "HuggingFaceTB/SmolLM2-135M",
        "test_inputs": [[1, 2, 3, 4]],
        "expected_loc": 300,
        "gpu_type": "T4",
        "timeout_seconds": 900,
    }
]
