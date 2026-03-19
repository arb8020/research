from __future__ import annotations

from typing import Any


def problem_prompt(problem: dict[str, Any] | None) -> str:
    if not isinstance(problem, dict):
        return ""
    payload = problem.get("payload")
    if not isinstance(payload, dict):
        return ""
    prompt = payload.get("prompt")
    if isinstance(prompt, str):
        return prompt
    messages = payload.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    return content
    return ""


def normalize_sample_payload(sample: dict[str, Any]) -> dict[str, Any]:
    """Flatten canonical attempt artifacts into the current sample read model."""
    sample = dict(sample)
    sample.setdefault("id", sample.get("attempt_id"))
    problem = sample.get("problem")
    if isinstance(problem, dict):
        payload = problem.get("payload")
        if isinstance(payload, dict):
            sample.setdefault("input", payload)
        sample.setdefault("ground_truth", problem.get("ground_truth"))
        sample.setdefault("prompt", problem_prompt(problem))

    evaluation = sample.get("evaluation")
    if isinstance(evaluation, dict):
        sample.setdefault("reward", evaluation.get("reward"))
        sample.setdefault("score", evaluation.get("score"))

    sample.setdefault("group_index", None)
    sample.setdefault("index", None)
    return sample
