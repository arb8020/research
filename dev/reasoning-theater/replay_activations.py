"""Extract replay manifests from saved AttemptRow JSON/JSONL files.

This is the bridge from `rollouts` outputs to later activation replay.
It does not run a model. It only normalizes the token-ID artifacts we need
to replay the exact generated sequence when TI/TO data is available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="AttemptRow JSON or JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Replay manifest JSONL path")
    return parser.parse_args()


def iter_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            return [payload]
        raise TypeError(f"Unsupported JSON payload type: {type(payload).__name__}")

    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"Expected JSON object per line, got {type(payload).__name__}")
            records.append(payload)
    return records


def _extract_text_parts(attempt: dict[str, Any]) -> tuple[str, str]:
    trajectory = attempt.get("trajectory", {})
    messages = trajectory.get("messages", [])
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return "", content
        if not isinstance(content, list):
            return "", ""
        thinking_parts: list[str] = []
        visible_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "thinking":
                thinking_parts.append(str(block.get("thinking", "")))
            elif block.get("type") == "text":
                visible_parts.append(str(block.get("text", "")))
        return "".join(thinking_parts), "".join(visible_parts)
    return "", ""


def extract_replay_record(attempt: dict[str, Any]) -> dict[str, Any] | None:
    trajectory = attempt.get("trajectory", {})
    completions = trajectory.get("completions", [])
    if not completions:
        return None

    last_completion = completions[-1]
    prompt_token_ids = last_completion.get("prompt_token_ids")
    choices = last_completion.get("choices", [])
    choice = choices[0] if choices else {}
    completion_token_ids = choice.get("token_ids")
    full_token_ids = None
    if prompt_token_ids is not None and completion_token_ids is not None:
        full_token_ids = list(prompt_token_ids) + list(completion_token_ids)

    thinking_text, visible_text = _extract_text_parts(attempt)
    return {
        "attempt_id": attempt.get("attempt_id"),
        "problem_id": attempt.get("problem", {}).get("problem_id"),
        "prompt_token_ids": prompt_token_ids,
        "completion_token_ids": completion_token_ids,
        "full_token_ids": full_token_ids,
        "thinking_text": thinking_text,
        "visible_text": visible_text,
        "has_tito_prompt_ids": prompt_token_ids is not None,
        "has_tito_completion_ids": completion_token_ids is not None,
    }


def main() -> None:
    args = parse_args()
    records = iter_json_records(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.output.open("w") as handle:
        for attempt in records:
            replay_record = extract_replay_record(attempt)
            if replay_record is None:
                continue
            handle.write(json.dumps(replay_record))
            handle.write("\n")
            written += 1

    print(f"Wrote {written} replay records to {args.output}")


if __name__ == "__main__":
    main()
