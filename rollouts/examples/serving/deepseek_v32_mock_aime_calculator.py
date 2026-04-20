"""Mixed serving scenario for DeepSeek V3.2 on the MI355X node.

This first scenario stays intentionally small:
- one single-turn exact-answer math workload
- one calculator tool-loop workload over the same answer style

The tasks are hand-authored mock AIME-style exact-integer prompts, not the
official AIME dataset. The point is serving-shape coverage, not benchmark
authority.

Usage:
    python -m argus run --config examples/serving/deepseek_v32_mock_aime_calculator.py --force-deploy-committed
"""

from __future__ import annotations

from examples.inference.evals.configs.bench.bench_deepseek_v3_2_amd_mi355x import (
    endpoint,
    hardware,
)
from examples.serving.math_serving_lib import (
    CALCULATOR_INTEGER_SYSTEM_PROMPT,
    SINGLE_TURN_INTEGER_SYSTEM_PROMPT,
    calculator_integer_score_fn,
    prepare_calculator_messages,
    prepare_single_turn_messages,
    single_turn_integer_score_fn,
)
from rollouts.environments.calculator import CalculatorEnvironment
from rollouts.eval import AgentRunSpec, EvalOutputConfig, EvalRunConfig, EvalTaskSpec
from rollouts.serving.configs import EvalServingWorkload, ServingOutputConfig, ServingScenario
from rollouts.training.scoring import FunctionScorer

TASKS = [
    {
        "id": "contest_1",
        "prompt": "Compute (17 * 19) + (23 * 21). Give only the final integer answer.",
        "answer": 806,
    },
    {
        "id": "contest_2",
        "prompt": "What is 1 + 2 + 3 + ... + 40? Give only the final integer answer.",
        "answer": 820,
    },
    {
        "id": "contest_3",
        "prompt": "A rectangle has side lengths 17 and 29. What is its perimeter?",
        "answer": 92,
    },
    {
        "id": "contest_4",
        "prompt": "Compute 7^2 + 8^2 + 9^2. Give only the final integer answer.",
        "answer": 194,
    },
    {
        "id": "contest_5",
        "prompt": "Compute (144 / 12) * (35 - 18). Give only the final integer answer.",
        "answer": 204,
    },
]


async def _calculator_environment_factory(_: dict[str, object]) -> CalculatorEnvironment:
    return CalculatorEnvironment()


single_turn_eval = EvalTaskSpec(
    tasks=TASKS,
    run_spec=AgentRunSpec(
        endpoint=endpoint,
        prepare_messages=lambda sample: prepare_single_turn_messages(
            sample,
            system_prompt=SINGLE_TURN_INTEGER_SYSTEM_PROMPT,
        ),
    ),
    scorer=FunctionScorer(single_turn_integer_score_fn),
    run=EvalRunConfig(
        max_concurrent=8,
        max_samples=len(TASKS),
        max_turns=1,
        verbose=False,
        show_progress=True,
    ),
    output=EvalOutputConfig(experiment_name="mock_aime_single_turn"),
    hardware=hardware,
)

calculator_eval = EvalTaskSpec(
    tasks=TASKS,
    run_spec=AgentRunSpec(
        endpoint=endpoint,
        prepare_messages=lambda sample: prepare_calculator_messages(
            sample,
            system_prompt=CALCULATOR_INTEGER_SYSTEM_PROMPT,
        ),
        environment_factory=_calculator_environment_factory,
    ),
    scorer=FunctionScorer(calculator_integer_score_fn),
    run=EvalRunConfig(
        max_concurrent=4,
        max_samples=len(TASKS),
        max_turns=8,
        verbose=False,
        show_progress=True,
    ),
    output=EvalOutputConfig(experiment_name="mock_aime_calculator"),
    hardware=hardware,
)

serving_scenario = ServingScenario(
    endpoint=endpoint,
    workloads=[
        EvalServingWorkload(
            name="mock_aime_single_turn",
            eval_task=single_turn_eval,
            concurrency=single_turn_eval.run.max_concurrent,
        ),
        EvalServingWorkload(
            name="mock_aime_calculator",
            eval_task=calculator_eval,
            concurrency=calculator_eval.run.max_concurrent,
        ),
    ],
    output=ServingOutputConfig(experiment_name="deepseek_v32_mock_aime_calculator"),
    hardware=hardware,
)
