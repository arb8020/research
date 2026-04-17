"""Harbor v0 smoke: run 2 TB2 tasks end-to-end.

Tasks: cancel-async-tasks, crack-7z-hash. Both have prebuilt docker images
(no build time) and small tractable success criteria. If this works,
widen the task set in mid.py / full.py.

Requires:
  - Harbor installed out-of-band:
      uv pip install 'harbor @ git+https://github.com/laude-institute/harbor.git@e0fcdc2'
  - Docker daemon running (HarborEnvironment uses docker compose).
  - ANTHROPIC_API_KEY set.
"""

from rollouts.config.tiers import EndpointConfig, OutputConfig, RunConfig

endpoint = EndpointConfig(
    model="claude-sonnet-4-5-20250929",
    provider="anthropic",
)

run = RunConfig(
    max_turns=30,
    max_concurrent=1,
    verbose=True,
    show_progress=True,
)

output = OutputConfig(
    experiment_name="harbor_v0_smoke",
)

# Two tight tasks with prebuilt images.
tasks_override = [
    {"task_id": "cancel-async-tasks"},
    {"task_id": "crack-7z-hash"},
]
