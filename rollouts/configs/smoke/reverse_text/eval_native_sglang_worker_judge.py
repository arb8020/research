"""Smoke eval using a native actor worker plus an LLM judge scorer.

This is the first scorer-owned judge witness for the worker-topology model:
- the actor and judge are semantic roles
- both roles are currently bound to the same realized worker
- the judge call goes through the shared eval endpoint contract

TODO(split-judge-worker): once eval can launch more than one inference worker
per run, bind `judge` to its own worker instead of reusing the actor worker.
"""

from __future__ import annotations

import re
from dataclasses import replace

from examples.rl.reverse_text.base_config import SYSTEM_PROMPT, parse_reversed_text
from rollouts.config_status import draft
from rollouts.core import Message, Metric, Score, Trajectory
from rollouts.eval import (
    AgentRunSpec,
    EndpointConfig,
    EvalOutputConfig,
    EvalRunConfig,
    EvalTaskSpec,
    endpoint_and_server_for_role,
    endpoint_config_from_inference_worker,
    materialize_endpoint,
)
from rollouts.training.configs import (
    HardwareConfig,
    InferenceConfig,
    InferenceRoleBinding,
    InferenceWorkerConfig,
    WorkerTopologyConfig,
)
from rollouts.training.scoring import FunctionScorer
from rollouts.training.types import AttemptResult


def prepare_messages(sample: dict[str, str]) -> list[Message]:
    text = sample["text"]
    return [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(
            role="user",
            content=(
                "Reverse the following text character-by-character. "
                "Put your answer in <reversed_text> tags.\n\n"
                f"Text to reverse: {text}"
            ),
        ),
    ]


async def grade_reversed_text_with_llm(
    *,
    candidate: str,
    expected: str,
    judge_endpoint_config: EndpointConfig,
) -> bool:
    from rollouts.agents import Actor
    from rollouts.providers import get_provider_function_by_format

    judge_endpoint = materialize_endpoint(judge_endpoint_config)
    judge_prompt = (
        "You are grading whether a reverse-text answer is correct.\n\n"
        f"Expected reversed text: {expected}\n"
        f"Candidate answer: {candidate}\n\n"
        "Reply exactly with:\n"
        "correct: yes\n"
        "or\n"
        "correct: no"
    )

    actor = Actor(
        trajectory=Trajectory(messages=[Message(role="user", content=judge_prompt)]),
        endpoint=judge_endpoint,
        tools=[],
    )
    provider_fn = get_provider_function_by_format(judge_endpoint.api_format)

    async def noop_callback(_chunk: object) -> None:
        return None

    judged_actor = await provider_fn(actor, noop_callback)
    response_text = ""
    for msg in reversed(judged_actor.trajectory.messages):
        if msg.role == "assistant" and msg.content:
            response_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            break
    match = re.search(r"correct:\s*(yes|no)", response_text, re.IGNORECASE)
    return match is not None and match.group(1).lower() == "yes"


config_status = draft(
    "Smoke eval for the scorer-owned LLM judge path over a named actor worker. "
    "Today the judge reuses the actor worker; splitting the judge onto its own "
    "worker needs multi-worker eval launch support."
)

worker_topology = WorkerTopologyConfig(
    hardware=HardwareConfig(
        provider="local",
        gpu_type="A100",
        gpu_count=1,
    ),
    inference_workers=(
        InferenceWorkerConfig(
            worker_id="actor",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            inference=InferenceConfig(
                cuda_device_ids=(0,),
                port=30000,
                mem_fraction=0.6,
            ),
        ),
    ),
    role_bindings=(
        InferenceRoleBinding(role="actor", worker_id="actor"),
        InferenceRoleBinding(role="judge", worker_id="actor"),
    ),
)

endpoint, server = endpoint_and_server_for_role(worker_topology, "actor")
judge_worker = worker_topology.get_worker_for_role("judge")
judge_endpoint = replace(
    endpoint_config_from_inference_worker(judge_worker),
    temperature=0.0,
    max_tokens=16,
)


async def reverse_text_llm_judge_score_fn(sample: AttemptResult, _context: object) -> Score:
    expected = sample.input["text"][::-1]
    response = sample.response
    parsed = parse_reversed_text(response)
    normalized = parsed if parsed is not None else response.strip().strip("\"'")
    llm_correct = await grade_reversed_text_with_llm(
        candidate=normalized,
        expected=expected,
        judge_endpoint_config=judge_endpoint,
    )
    exact_match = normalized == expected
    return Score(
        metrics=(
            Metric("judge_correct", 1.0 if llm_correct else 0.0, weight=1.0),
            Metric("exact_match", 1.0 if exact_match else 0.0, weight=0.0),
        )
    )


eval_task = EvalTaskSpec(
    tasks=[
        {"text": "hello world"},
        {"text": "prime intellect"},
    ],
    run_spec=AgentRunSpec(
        endpoint=endpoint,
        prepare_messages=prepare_messages,
    ),
    scorer=FunctionScorer(reverse_text_llm_judge_score_fn),
    run=EvalRunConfig(
        max_concurrent=1,
        max_samples=2,
        max_turns=1,
        verbose=True,
        show_progress=True,
    ),
    output=EvalOutputConfig(
        experiment_name="smoke_reverse_text_native_sglang_worker_judge",
    ),
    hardware=worker_topology.hardware,
    server=server,
)
