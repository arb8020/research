"""Tau2 v0 smoke: 3 retail tasks against an API model.

Retail is the smallest, fastest domain. 3 tasks is enough to exercise:
  - agent tool call path (exec_tool → tau2 env)
  - user simulator path (on_assistant_message → rollout)
  - scoring path (evaluate_simulation → Score)

Requires:
  - tau2 installed: `uv pip install 'tau2-bench @ git+https://github.com/sierra-research/tau2-bench'`
  - OPENAI_API_KEY set (default user-sim model is gpt-4.1-mini on OpenAI)
  - Whatever credentials the agent endpoint needs (ANTHROPIC_API_KEY for default)
"""

from rollouts.config.tiers import EndpointConfig, OutputConfig, RunConfig
from tau2_v0.prepare import build_sample_rows

endpoint = EndpointConfig(
    model="claude-sonnet-4-5-20250929",
    provider="anthropic",
)

run = RunConfig(
    max_turns=40,
    max_concurrent=1,
    verbose=True,
    show_progress=True,
)

output = OutputConfig(
    experiment_name="tau2_v0_smoke",
)

# Build 3 retail sample rows at config-eval time — tau2 package is imported
# here, which means running this config requires tau2 installed.
tasks_override = build_sample_rows(domain="retail", limit=3)
