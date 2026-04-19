"""Telecom solo_mode smoke: 1 task, no user simulator.

Verifies the solo_mode wiring: agent runs against domain tools alone,
text-only non-stop turns terminate as AGENT_ERROR, ###STOP### terminates
as AGENT_STOP.
"""

from rollouts.config.tiers import EndpointConfig, OutputConfig, RunConfig
from tau2_v0.prepare import build_sample_rows

endpoint = EndpointConfig(model="claude-sonnet-4-5-20250929", provider="anthropic")
run = RunConfig(max_turns=40, max_concurrent=1, verbose=False, show_progress=True)
output = OutputConfig(experiment_name="tau2_v0_solo_smoke")

# solo_mode telecom — no user_endpoint needed.
tasks_override = build_sample_rows(
    domain="telecom",
    split="base",
    limit=1,
    solo_mode=True,
)
