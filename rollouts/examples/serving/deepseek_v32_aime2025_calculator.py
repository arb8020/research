"""Real AIME 2025 mixed serving scenario for DeepSeek V3.2 on the MI355X node.

This uses the official `opencompass/AIME2025` dataset:
- AIME2025-I test split
- AIME2025-II test split

Usage:
    python -m argus run --config examples/serving/deepseek_v32_aime2025_calculator.py --force-deploy-committed
"""

from __future__ import annotations

from dataclasses import replace

from examples.inference.evals.configs.bench.bench_deepseek_v3_2_amd_mi355x import (
    endpoint as _bench_endpoint,
)
from examples.inference.evals.configs.bench.bench_deepseek_v3_2_amd_mi355x import (
    hardware,
)

# AIME problems need room to reason; the bench endpoint's 256-token cap
# truncates every response to finish_reason="length" before the model
# can ever reach a tool call. 8192 is a comfortable ceiling for a
# per-turn generation; the calculator workload is multi-turn so total
# generation can exceed this across turns.
endpoint = replace(_bench_endpoint, max_tokens=8192)
from examples.serving.math_serving_lib import (
    AIME2025_CALCULATOR_SYSTEM_PROMPT,
    AIME2025_SINGLE_TURN_SYSTEM_PROMPT,
    calculator_integer_score_fn,
    load_aime2025_tasks,
    prepare_calculator_messages,
    prepare_single_turn_messages,
    single_turn_integer_score_fn,
)
from rollouts.environments.calculator import CalculatorEnvironment
from rollouts.eval import AgentRunSpec, EvalOutputConfig, EvalRunConfig, EvalTaskSpec
from rollouts.serving.configs import EvalServingWorkload, ServingOutputConfig, ServingScenario
from rollouts.training.scoring import FunctionScorer

TASKS = load_aime2025_tasks()


async def _calculator_environment_factory(_: dict[str, object]) -> CalculatorEnvironment:
    return CalculatorEnvironment()


single_turn_eval = EvalTaskSpec(
    tasks=TASKS,
    run_spec=AgentRunSpec(
        endpoint=endpoint,
        prepare_messages=lambda sample: prepare_single_turn_messages(
            sample,
            system_prompt=AIME2025_SINGLE_TURN_SYSTEM_PROMPT,
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
    output=EvalOutputConfig(experiment_name="aime2025_single_turn"),
    hardware=hardware,
)

calculator_eval = EvalTaskSpec(
    tasks=TASKS,
    run_spec=AgentRunSpec(
        endpoint=endpoint,
        prepare_messages=lambda sample: prepare_calculator_messages(
            sample,
            system_prompt=AIME2025_CALCULATOR_SYSTEM_PROMPT,
        ),
        environment_factory=_calculator_environment_factory,
    ),
    # TODO: Keep this workload only as an aspirational tool-loop witness until
    # we verify DeepSeek is actually producing tool calls here under serving load.
    scorer=FunctionScorer(calculator_integer_score_fn),
    run=EvalRunConfig(
        max_concurrent=4,
        max_samples=len(TASKS),
        max_turns=12,
        verbose=False,
        show_progress=True,
    ),
    output=EvalOutputConfig(experiment_name="aime2025_calculator"),
    hardware=hardware,
)

serving_scenario = ServingScenario(
    endpoint=endpoint,
    workloads=[
        EvalServingWorkload(
            name="aime2025_single_turn",
            eval_task=single_turn_eval,
            concurrency=single_turn_eval.run.max_concurrent,
        ),
        EvalServingWorkload(
            name="aime2025_calculator",
            eval_task=calculator_eval,
            concurrency=calculator_eval.run.max_concurrent,
        ),
    ],
    output=ServingOutputConfig(experiment_name="deepseek_v32_aime2025_calculator"),
    hardware=hardware,
)
