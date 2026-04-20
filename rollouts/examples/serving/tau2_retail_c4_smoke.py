"""Small tau2 retail serving scenario.

Runs 4 retail tau2 tasks at concurrency 4 against an API endpoint to exercise:
- tau2 tool-environment execution
- user-simulator turns
- post-run operational metrics aggregation
"""

from __future__ import annotations

import os
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


def _ensure_tau2_data_dir() -> None:
    if os.environ.get("TAU2_DATA_DIR"):
        return

    candidates = [
        *Path.home().glob(".cache/uv/git-v0/checkouts/*/*/data/tau2/domains/retail/tasks.json"),
        *Path.home().glob(
            ".cache/uv/archive-v0/*/inspect_evals/tau2/data/domains/retail/tasks.json"
        ),
    ]
    if not candidates:
        raise FileNotFoundError(
            "Could not locate tau2 retail task data. Set TAU2_DATA_DIR explicitly."
        )
    os.environ["TAU2_DATA_DIR"] = str(candidates[0].parents[3])


_ensure_tau2_data_dir()

from tau2_v0.eval import make_environment, prepare_messages, score_sample
from tau2_v0.prepare import DEFAULT_USER_ENDPOINT, build_sample_rows

TASKS = build_sample_rows(
    domain="retail",
    limit=4,
    user_endpoint=DEFAULT_USER_ENDPOINT,
)

endpoint = ExternalEndpoint(
    url="https://api.anthropic.com/v1",
    model="claude-sonnet-4-5-20250929",
    provider="anthropic",
)

tau2_eval = EvalTaskSpec(
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
        max_turns=40,
        verbose=False,
        show_progress=True,
    ),
    output=EvalOutputConfig(experiment_name="tau2_retail_c4_smoke"),
)

serving_scenario = ServingScenario(
    endpoint=endpoint,
    workloads=[
        EvalServingWorkload(
            name="tau2_retail_c4",
            eval_task=tau2_eval,
            concurrency=4,
            max_samples=4,
        ),
    ],
    output=ServingOutputConfig(experiment_name="tau2_retail_c4_smoke"),
)
