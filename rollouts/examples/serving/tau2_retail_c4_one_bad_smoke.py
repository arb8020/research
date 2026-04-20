"""Tau2 retail serving scenario with one intentionally bad sample.

This is a failure-recovery witness:
- 3 valid tau2 tasks should run normally
- 1 malformed sample should fail at the sample boundary
- the scenario should still complete and emit operational reports
"""

from __future__ import annotations

from tau2_v0.eval import make_environment, prepare_messages, score_sample

from examples.serving.tau2_retail_c4_smoke import (
    TASKS as VALID_TASKS,
)
from examples.serving.tau2_retail_c4_smoke import (
    endpoint,
)
from rollouts.eval import AgentRunSpec, EvalOutputConfig, EvalRunConfig, EvalTaskSpec
from rollouts.serving.configs import EvalServingWorkload, ServingOutputConfig, ServingScenario
from rollouts.training.scoring import FunctionScorer

TASKS = [
    *VALID_TASKS[:3],
    {
        "task_id": "bad_sample",
        "domain": "retail",
        "task_json": "{}",
        "user_endpoint": VALID_TASKS[0]["user_endpoint"],
    },
]

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
    output=EvalOutputConfig(experiment_name="tau2_retail_c4_one_bad_smoke"),
)

serving_scenario = ServingScenario(
    endpoint=endpoint,
    workloads=[
        EvalServingWorkload(
            name="tau2_retail_c4_one_bad",
            eval_task=tau2_eval,
            concurrency=4,
            max_samples=4,
        ),
    ],
    output=ServingOutputConfig(experiment_name="tau2_retail_c4_one_bad_smoke"),
)
