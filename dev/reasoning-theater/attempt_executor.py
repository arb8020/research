"""Reasoning-theater attempt executor built around rollouts datatypes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
import sys
from typing import Any, Protocol, cast

from datasets import parse_multiple_choice_answer
from prefixes import build_prefix_slices
from records import MultipleChoiceTask, PrefixPrediction, PrefixSlice

ROLLOUTS_PROJECT_ROOT = Path(__file__).resolve().parents[2] / "rollouts"
if str(ROLLOUTS_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(ROLLOUTS_PROJECT_ROOT))

from rollouts.core import ThinkingContent
from rollouts.training.types import AttemptRow


class ReasoningTheaterBackend(Protocol):
    """Backend contract for one reasoning-theater sample.

    The backend owns model-specific execution. This module owns the sample
    denotation and metadata normalization.
    """

    async def run_primary_attempt(
        self,
        sample_data: dict[str, Any],
        sample_id: str,
        environment: Any | None,
        run_config: Any,
    ) -> AttemptRow: ...


def extract_last_assistant_content(sample: AttemptRow) -> tuple[str, str]:
    """Return `(thinking_text, visible_text)` from the last assistant message."""
    if sample.trajectory is None:
        return "", ""

    for message in reversed(sample.trajectory.messages):
        if getattr(message, "role", None) != "assistant":
            continue
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return "", content
        if not isinstance(content, list):
            return "", ""

        thinking_parts: list[str] = []
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, ThinkingContent):
                thinking_parts.append(block.thinking)
                continue
            block_type = getattr(block, "type", None)
            if block_type == "thinking":
                thinking_parts.append(getattr(block, "thinking", ""))
            elif block_type == "text":
                text_parts.append(getattr(block, "text", ""))
            elif isinstance(block, dict):
                if block.get("type") == "thinking":
                    thinking_parts.append(str(block.get("thinking", "")))
                elif block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
        return "".join(thinking_parts), "".join(text_parts)
    return "", ""


def extract_last_completion_token_ids(sample: AttemptRow) -> tuple[list[int] | None, list[int] | None]:
    """Return `(prompt_token_ids, completion_token_ids)` from the last completion."""
    if sample.trajectory is None or not sample.trajectory.completions:
        return None, None

    last_completion = sample.trajectory.completions[-1]
    prompt_ids = (
        list(last_completion.prompt_token_ids) if last_completion.prompt_token_ids is not None else None
    )
    completion_ids: list[int] | None = None
    if last_completion.choices and last_completion.choices[0].token_ids is not None:
        completion_ids = list(last_completion.choices[0].token_ids)
    return prompt_ids, completion_ids


async def _maybe_call_prediction_hook(
    hook: Callable[..., Any] | None,
    *,
    sample_data: dict[str, Any],
    sample_id: str,
    prefix: PrefixSlice,
    primary_sample: AttemptRow,
    environment: Any | None,
    run_config: Any,
) -> PrefixPrediction | None:
    if hook is None:
        return None
    value = hook(
        sample_data=sample_data,
        sample_id=sample_id,
        prefix=prefix,
        primary_sample=primary_sample,
        environment=environment,
        run_config=run_config,
    )
    if isinstance(value, Awaitable):
        value = await cast(Awaitable[Any], value)
    if value is None:
        return None
    if isinstance(value, PrefixPrediction):
        return value
    if isinstance(value, dict):
        return PrefixPrediction(**value)
    if isinstance(value, str):
        return PrefixPrediction(prefix_label=prefix.label, predicted_answer=value, raw_response=value)
    raise TypeError(f"Unsupported prefix prediction type: {type(value).__name__}")


def build_attempt_executor(
    *,
    backend: ReasoningTheaterBackend,
    prefix_fractions: Sequence[float],
    min_prefix_chars: int = 32,
    forced_answer_hook: Callable[..., Any] | None = None,
    monitor_hook: Callable[..., Any] | None = None,
) -> Callable[[dict[str, Any], str, Any | None, Any], Awaitable[AttemptRow]]:
    """Build a `rollouts`-compatible attempt executor."""

    async def attempt_executor(
        sample_data: dict[str, Any],
        sample_id: str,
        environment: Any | None,
        run_config: Any,
    ) -> AttemptRow:
        task = MultipleChoiceTask.from_payload(sample_data)
        sample = await backend.run_primary_attempt(sample_data, sample_id, environment, run_config)
        if sample.trajectory is None:
            raise ValueError("Primary attempt must populate AttemptRow.trajectory")

        thinking_text, visible_text = extract_last_assistant_content(sample)
        prompt_token_ids, completion_token_ids = extract_last_completion_token_ids(sample)
        answer_source = visible_text or thinking_text
        parsed_final_answer = parse_multiple_choice_answer(answer_source, task.choice_labels)
        prefix_slices = build_prefix_slices(
            reasoning_text=thinking_text or visible_text,
            prefix_fractions=prefix_fractions,
            completion_token_ids=completion_token_ids,
            min_chars=min_prefix_chars,
        )

        forced_predictions: list[PrefixPrediction] = []
        monitor_predictions: list[PrefixPrediction] = []
        for prefix in prefix_slices:
            forced = await _maybe_call_prediction_hook(
                forced_answer_hook,
                sample_data=sample_data,
                sample_id=sample_id,
                prefix=prefix,
                primary_sample=sample,
                environment=environment,
                run_config=run_config,
            )
            if forced is not None:
                forced_predictions.append(forced)

            monitor = await _maybe_call_prediction_hook(
                monitor_hook,
                sample_data=sample_data,
                sample_id=sample_id,
                prefix=prefix,
                primary_sample=sample,
                environment=environment,
                run_config=run_config,
            )
            if monitor is not None:
                monitor_predictions.append(monitor)

        sample.metadata = {
            **sample.metadata,
            "reasoning_theater": {
                "task_id": task.task_id,
                "parsed_final_answer": parsed_final_answer,
                "ground_truth_answer": task.correct_answer,
                "thinking_text": thinking_text,
                "visible_text": visible_text,
                "has_prompt_token_ids": prompt_token_ids is not None,
                "has_completion_token_ids": completion_token_ids is not None,
                "prompt_token_ids": prompt_token_ids,
                "completion_token_ids": completion_token_ids,
                "prefixes": [prefix.to_dict() for prefix in prefix_slices],
                "forced_answers": [prediction.to_dict() for prediction in forced_predictions],
                "monitor_predictions": [prediction.to_dict() for prediction in monitor_predictions],
            },
        }
        return sample

    return attempt_executor
