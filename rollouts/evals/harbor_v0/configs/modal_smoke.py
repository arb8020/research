"""Harbor v0 smoke on Modal: run 2 TB2 tasks end-to-end.

Tasks: cancel-async-tasks, crack-7z-hash. Both have prebuilt docker images
and small tractable success criteria. This is the live witness for Harbor's
Modal-backed task container path.

Requires:
  - Harbor installed with Modal support:
      uv pip install 'harbor[modal] @ git+https://github.com/laude-institute/harbor.git@e0fcdc2'
  - Modal auth configured (`modal token new` or env vars).
  - ANTHROPIC_API_KEY set.
"""

from harbor_v0.config_types import (
    HarborTaskEnvironmentConfig,
    ModalHarborHost,
)
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
    experiment_name="harbor_v0_modal_smoke",
)

environment = HarborTaskEnvironmentConfig(
    host=ModalHarborHost(
        app_name="rollouts-harbor",
    ),
)

tasks_override = [
    {"task_id": "cancel-async-tasks"},
    {"task_id": "crack-7z-hash"},
]
