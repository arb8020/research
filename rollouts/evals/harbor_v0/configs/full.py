"""Harbor v0 full: run a broader TB2 task set.

For v0 this is a hand-picked 8-task subset to stress the harness without
running all 89 tasks (several of which need heavy builds or GPU). Once
the smoke is stable and we've run the 8, we extend further.

Same requirements as smoke.py.
"""

from harbor_v0.config_types import (
    HarborTaskEnvironmentConfig,
    LocalHarborHost,
)
from rollouts.config.tiers import EndpointConfig, OutputConfig, RunConfig

endpoint = EndpointConfig(
    model="claude-sonnet-4-5-20250929",
    provider="anthropic",
)

run = RunConfig(
    max_turns=60,
    max_concurrent=2,
    verbose=True,
    show_progress=True,
)

output = OutputConfig(
    experiment_name="harbor_v0_full",
)

environment = HarborTaskEnvironmentConfig(
    host=LocalHarborHost(),
)

# Eight hand-picked TB2 tasks — all with prebuilt images, varied
# categories (async, crypto, text processing, builds, data manipulation).
tasks_override = [
    {"task_id": "cancel-async-tasks"},
    {"task_id": "crack-7z-hash"},
    {"task_id": "count-dataset-tokens"},
    {"task_id": "chess-best-move"},
    {"task_id": "break-filter-js-from-html"},
    {"task_id": "build-cython-ext"},
    {"task_id": "configure-git-webserver"},
    {"task_id": "cobol-modernization"},
]
