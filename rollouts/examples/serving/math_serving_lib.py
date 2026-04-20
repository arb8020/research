from __future__ import annotations

import json
import re

from rollouts.core import Message, Metric, Score
from rollouts.training.types import RowAttempt

SINGLE_TURN_INTEGER_SYSTEM_PROMPT = (
    "You are solving exact-answer contest math problems. "
    "Return the final answer as a bare integer or in \\boxed{n}."
)

CALCULATOR_INTEGER_SYSTEM_PROMPT = (
    "You are solving exact-answer contest math problems. "
    "Use the calculator tools for arithmetic. "
    "When you are done, call complete_task with the final_result."
)

AIME2025_SINGLE_TURN_SYSTEM_PROMPT = (
    "Solve the following AIME 2025 problem step by step. "
    "The final answer must be an integer from 000 to 999 inclusive. "
    "Put the final answer on its own line as `Answer: NNN`, or in \\boxed{NNN}."
)

AIME2025_CALCULATOR_SYSTEM_PROMPT = (
    "Solve the following AIME 2025 problem step by step. "
    "Use the calculator tools for arithmetic. "
    "AIME answers are integers from 000 to 999 inclusive. "
    "When you are done, call complete_task with the final_result."
)


def normalize_integer_answer(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(int(float(str(value).strip())))
    except (TypeError, ValueError):
        return None


def extract_integer_answer(text: str) -> float | None:
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        parsed = normalize_integer_answer(boxed[-1])
        if parsed is not None:
            return parsed

    answer_lines = re.findall(r"Answer:\s*([0-9]+)", text, flags=re.IGNORECASE)
    if answer_lines:
        parsed = normalize_integer_answer(answer_lines[-1])
        if parsed is not None:
            return parsed

    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    return normalize_integer_answer(matches[-1])


def single_turn_integer_score_fn(sample: RowAttempt, _context: object) -> Score:
    trajectory = sample.trajectory
    if trajectory is None:
        return Score(metrics=(Metric("correct", 0.0, weight=1.0),))

    assistant_text = ""
    for msg in reversed(trajectory.messages):
        # TODO: Handle structured assistant content blocks, not just raw strings.
        # OpenAI-compatible providers often return `[{"type": "text", ...}]` here,
        # and this scorer currently misclassifies valid answers as parse failures.
        if msg.role == "assistant" and isinstance(msg.content, str):
            assistant_text = msg.content
            break

    predicted = extract_integer_answer(assistant_text)
    ground_truth = normalize_integer_answer(sample.ground_truth or sample.input["answer"])
    if predicted is None or ground_truth is None:
        return Score(
            metrics=(
                Metric("correct", 0.0, weight=1.0),
                Metric("parse_failed", 1.0, weight=0.0),
            )
        )
    return Score(
        metrics=(
            Metric("correct", 1.0 if abs(predicted - ground_truth) < 0.01 else 0.0, weight=1.0),
            Metric("predicted", predicted, weight=0.0),
            Metric("ground_truth", ground_truth, weight=0.0),
        )
    )


def calculator_integer_score_fn(sample: RowAttempt, _context: object) -> Score:
    trajectory = sample.trajectory
    if trajectory is None:
        return Score(metrics=(Metric("correct", 0.0, weight=1.0),))

    final_answer = None
    # TODO: Add a stronger witness for "tool workload actually used tools".
    # Right now this scorer treats "no tool calls, no complete_task" as a parse
    # failure, but the serving report does not surface that the calculator path
    # never exercised tool use at all.
    for msg in trajectory.messages:
        if msg.role != "assistant":
            continue
        tool_calls = msg.get_tool_calls() if hasattr(msg, "get_tool_calls") else []
        for tool_call in tool_calls:
            if tool_call.name != "complete_task":
                continue
            args = tool_call.args
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    continue
            final_answer = normalize_integer_answer(args.get("final_result"))
            if final_answer is not None:
                break
        if final_answer is not None:
            break

    if final_answer is None:
        for msg in trajectory.messages:
            if msg.role == "tool" and isinstance(msg.content, str):
                final_answer = extract_integer_answer(msg.content)
                if final_answer is not None:
                    break

    ground_truth = normalize_integer_answer(sample.ground_truth or sample.input["answer"])
    if final_answer is None or ground_truth is None:
        return Score(
            metrics=(
                Metric("correct", 0.0, weight=1.0),
                Metric("parse_failed", 1.0, weight=0.0),
            )
        )
    return Score(
        metrics=(
            Metric("correct", 1.0 if abs(final_answer - ground_truth) < 0.01 else 0.0, weight=1.0),
            Metric("predicted", final_answer, weight=0.0),
            Metric("ground_truth", ground_truth, weight=0.0),
        )
    )


def prepare_single_turn_messages(
    sample: dict[str, object],
    *,
    system_prompt: str,
) -> list[Message]:
    return [
        Message(role="system", content=system_prompt),
        Message(role="user", content=str(sample["prompt"])),
    ]


def prepare_calculator_messages(
    sample: dict[str, object],
    *,
    system_prompt: str,
) -> list[Message]:
    # TODO: Tighten this prompt or tool contract for DeepSeek V3.2 on SGLang.
    # The current calculator serving run completed with empty assistant turns,
    # so this workload is not yet a reliable witness for multi-turn tool calling.
    return [
        Message(role="system", content=system_prompt),
        Message(role="user", content=str(sample["prompt"])),
    ]


def load_aime2025_tasks(max_samples: int | None = None) -> list[dict[str, object]]:
    from datasets import load_dataset

    tasks: list[dict[str, object]] = []
    for subset in ("AIME2025-I", "AIME2025-II"):
        dataset = load_dataset("opencompass/AIME2025", subset, split="test")
        for idx, row in enumerate(dataset):
            tasks.append({
                "id": f"{subset.lower()}_{idx:02d}",
                "subset": subset,
                "prompt": row["question"],
                "answer": row["answer"],
            })

    if max_samples is not None:
        return tasks[:max_samples]
    return tasks
