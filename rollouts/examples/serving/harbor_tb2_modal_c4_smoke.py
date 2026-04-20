"""Small Harbor/TB2 serving scenario on Modal.

Runs 4 short TB2 tasks at concurrency 4 to exercise:
- Harbor-backed tool execution
- live verifier scoring
- post-run operational metrics aggregation
"""

from __future__ import annotations

import sys
from pathlib import Path

from rollouts.eval import (
    AgentRunSpec,
    EvalOutputConfig,
    EvalRunConfig,
    EvalTaskSpec,
    ExternalEndpoint,
)
from rollouts.serving.configs import EvalServingWorkload, ServingOutputConfig, ServingScenario
from rollouts.training.scoring import FunctionScorer

_EVALS_ROOT = Path(__file__).resolve().parents[2] / "evals"
if str(_EVALS_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVALS_ROOT))

from harbor_v0.config_types import ModalHarborHost
from harbor_v0.eval import make_environment, prepare_messages, score_sample

from rollouts.environments.harbor_environment import attach_harbor_host_to_tasks

TASKS = attach_harbor_host_to_tasks(
    [
        {"task_id": "cancel-async-tasks"},
        {"task_id": "crack-7z-hash"},
        {"task_id": "count-dataset-tokens"},
        {"task_id": "chess-best-move"},
    ],
    ModalHarborHost(app_name="rollouts-harbor"),
)

endpoint = ExternalEndpoint(
    url="https://api.anthropic.com/v1",
    model="claude-sonnet-4-5-20250929",
    provider="anthropic",
)

harbor_eval = EvalTaskSpec(
    tasks=TASKS,
    run_spec=AgentRunSpec(
        endpoint=endpoint,
        prepare_messages=prepare_messages,
        environment_factory=make_environment,
    ),
    scorer=FunctionScorer(score_sample),
    run=EvalRunConfig(
        max_concurrent=4,
        max_samples=4,
        max_turns=30,
        verbose=False,
        show_progress=True,
    ),
    output=EvalOutputConfig(experiment_name="harbor_tb2_modal_c4_smoke"),
)

serving_scenario = ServingScenario(
    endpoint=endpoint,
    workloads=[
        EvalServingWorkload(
            name="harbor_tb2_modal_c4",
            eval_task=harbor_eval,
            concurrency=4,
            max_samples=4,
        ),
    ],
    output=ServingOutputConfig(experiment_name="harbor_tb2_modal_c4_smoke"),
)
