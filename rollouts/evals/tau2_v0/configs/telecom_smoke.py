"""Telecom smoke: 3 tasks to exercise the user_tools path."""

from rollouts.config.tiers import EndpointConfig, OutputConfig, RunConfig
from tau2_v0.prepare import build_sample_rows

endpoint = EndpointConfig(model="claude-sonnet-4-5-20250929", provider="anthropic")
run = RunConfig(max_turns=40, max_concurrent=1, verbose=False, show_progress=True)
output = OutputConfig(experiment_name="tau2_v0_telecom_smoke")
tasks_override = build_sample_rows(domain="telecom", split="base", limit=3)
