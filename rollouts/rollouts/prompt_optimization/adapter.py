"""GEPA adapter types.

Type aliases for adapter functions - no Protocol needed.
Following: pure functions, explicit data flow.
"""

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from .types import Candidate, EvaluationBatch

# ─── Type Aliases ─────────────────────────────────────────────────────────────

# Evaluate function: runs candidate on batch, returns scores and optional traces
EvaluateFn = Callable[
    [Sequence[dict[str, Any]], Candidate, bool],  # (batch, candidate, capture_traces)
    Awaitable[EvaluationBatch],
]

# Make reflective dataset: extracts feedback from traces for LLM reflection
MakeReflectiveFn = Callable[
    [Candidate, EvaluationBatch, list[str]],  # (candidate, eval_batch, components_to_update)
    dict[str, list[dict[str, Any]]],
]


# ─── Documentation ────────────────────────────────────────────────────────────

"""
GEPA uses two functions as its integration point:

1. evaluate_fn(batch, candidate, capture_traces) -> EvaluationBatch
   - Runs the candidate on a batch of samples
   - Returns scores and optionally execution traces
   - capture_traces=True needed for reflective mutation

2. make_reflective_fn(candidate, eval_batch, components_to_update) -> dict
   - Extracts per-component feedback from execution traces
   - Returns dict mapping component name to list of feedback items
   - Each item should have: Inputs, Generated Outputs, Feedback

Example (simple single-prompt adapter):
    
    async def evaluate(
        batch: Sequence[dict],
        candidate: Candidate,
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        outputs = []
        scores = []
        for sample in batch:
            output = await run_llm(candidate["system"], sample["query"])
            score = 1.0 if output == sample["answer"] else 0.0
            outputs.append(output)
            scores.append(score)
        return EvaluationBatch(outputs=tuple(outputs), scores=tuple(scores))
    
    def make_reflective(
        candidate: Candidate,
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict]]:
        if "system" not in components_to_update:
            return {}
        items = [
            {"Inputs": o, "Feedback": "Correct" if s == 1.0 else "Wrong"}
            for o, s in zip(eval_batch.outputs, eval_batch.scores)
        ]
        return {"system": items}
    
    # Use with run_gepa
    result = await run_gepa(
        seed_candidate={"system": "You are a classifier."},
        dataset=my_dataset,
        evaluate_fn=evaluate,
        make_reflective_fn=make_reflective,
        config=GEPAConfig(max_evaluations=100),
        reflection_endpoint=endpoint,
    )

Example (terminal-bench adapter with config):

    from functools import partial
    
    config = TerminalBenchConfig(endpoint=endpoint, max_turns=30)
    
    result = await run_gepa(
        seed_candidate={"instruction_prompt": "You are a terminal agent..."},
        dataset=[{"task_id": "fix-permissions"}],
        evaluate_fn=partial(evaluate_terminal_bench, config),
        make_reflective_fn=make_terminal_bench_reflective,
        config=GEPAConfig(max_evaluations=100),
        reflection_endpoint=endpoint,
    )
"""
