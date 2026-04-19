"""Tau2 v0 full: all 3 domains, base split.

Airline (~50 tasks) + Retail (~115 tasks) + Telecom base (114 tasks).
For benchmarking tool-calling on the mi355x deepseek serving endpoint:
configure the agent endpoint to point at our serving URL and leave the
user-sim endpoint on an API model (the user-sim quality is a confound we
don't want to introduce).
"""

from rollouts.config.tiers import EndpointConfig, OutputConfig, RunConfig
from tau2_v0.prepare import build_sample_rows

endpoint = EndpointConfig(
    model="claude-sonnet-4-5-20250929",
    provider="anthropic",
)

run = RunConfig(
    max_turns=60,
    max_concurrent=8,
    verbose=False,
    show_progress=True,
)

output = OutputConfig(
    experiment_name="tau2_v0_full",
)

tasks_override = (
    build_sample_rows(domain="airline")
    + build_sample_rows(domain="retail")
    + build_sample_rows(domain="telecom", split="base")
)
