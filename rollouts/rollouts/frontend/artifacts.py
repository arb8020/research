from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
    """Flatten the canonical attempt result into the current frontend DTO shape."""
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


def find_trace_dir(results_dir: Path, known_results_dirs: list[Path], trace_id: str) -> Path | None:
    for search_dir in [results_dir, *known_results_dirs]:
        candidate = search_dir / trace_id
        if candidate.is_dir():
            return candidate
    return None


def list_trace_samples(trace_dir: Path) -> list[dict[str, Any]]:
    samples_dir = trace_dir / "samples"
    samples = []
    if not samples_dir.exists():
        return samples

    for sample_file in sorted(samples_dir.glob("*.json")):
        try:
            sample = normalize_sample_payload(json.loads(sample_file.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
        trajectory = sample.get("trajectory", {})
        samples.append({
            "name": sample_file.stem,
            "messages": trajectory.get("messages", []),
            "rewards": sample.get("reward", 0),
            "metadata": sample.get("metadata", {}),
        })
    return samples


def load_trace_payload(trace_dir: Path, trace_id: str) -> dict[str, Any]:
    report_path = trace_dir / "report.json"
    report = json.loads(report_path.read_text())
    samples = list_trace_samples(trace_dir)
    return {
        "id": trace_id,
        "name": report.get("config_name", trace_id),
        "total_samples": report.get("total_samples", len(samples)),
        "mean_reward": report.get("summary_metrics", {}).get("mean_reward", 0),
        "samples": samples,
        "sample_ids": report.get("sample_ids", []),
        "report": report,
    }


def _rehydrate_external_session_trajectory(
    sample: dict[str, Any], metadata: dict[str, Any]
) -> None:
    runtime = metadata.get("runtime")
    if runtime == "claude_code" and sample.get("status") == "completed":
        session_id = metadata.get("session_id")
        if isinstance(session_id, str) and session_id:
            try:
                from ..drivers.session_adapter import (
                    claude_session_to_messages,
                    find_claude_session,
                )

                session_path = find_claude_session(session_id)
                if session_path is not None and session_path.exists():
                    sample["trajectory"] = {
                        "completions": [],
                        "messages": [asdict(m) for m in claude_session_to_messages(session_path)],
                    }
            except Exception:
                logger.exception("Failed to rehydrate Claude session %s", session_id)
    elif runtime == "codex" and sample.get("status") == "completed":
        session_id = metadata.get("session_id")
        if isinstance(session_id, str) and session_id:
            try:
                from ..drivers.session_adapter import (
                    codex_session_to_messages,
                    find_codex_session,
                )

                session_path = find_codex_session(session_id)
                if session_path is not None and session_path.exists():
                    sample["trajectory"] = {
                        "completions": [],
                        "messages": [asdict(m) for m in codex_session_to_messages(session_path)],
                    }
            except Exception:
                logger.exception("Failed to rehydrate Codex session %s", session_id)


def _reconstruct_external_messages_from_events(
    events_path: Path,
    *,
    runtime: str,
    sample_id: str,
) -> list[dict[str, Any]]:
    from ..drivers.claude import _ClaudeEventParser
    from ..drivers.codex import _CodexEventParser
    from ..drivers.runner import _EventAccumulator

    parser = _ClaudeEventParser() if runtime == "claude_code" else _CodexEventParser()
    accumulator = _EventAccumulator()

    for line in events_path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("sample_id") != sample_id:
            continue
        raw_line = event.get("raw_line")
        if not isinstance(raw_line, str) or not raw_line:
            continue
        try:
            raw_msg = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        for parsed_event in parser.parse(raw_msg):
            accumulator.handle(parsed_event)

    return [asdict(message) for message in accumulator.finalize()]


def load_sample_payload(results_dir: Path, trace_id: str, sample_id: str) -> dict[str, Any]:
    return load_sample_payload_from_trace_dir(results_dir / trace_id, sample_id)


def load_sample_payload_from_trace_dir(trace_dir: Path, sample_id: str) -> dict[str, Any]:
    sample_path = trace_dir / "samples" / f"{sample_id}.json"
    sample = json.loads(sample_path.read_text())
    metadata = sample.get("metadata", {})

    _rehydrate_external_session_trajectory(sample, metadata)

    traj = sample.get("trajectory", {})
    if "completions" not in traj:
        traj["completions"] = []
    if "messages" not in traj:
        traj["messages"] = []

    if not traj["messages"] and metadata.get("runtime") in {"claude_code", "codex"}:
        events_path = trace_dir / "events.jsonl"
        if events_path.exists():
            try:
                reconstructed = _reconstruct_external_messages_from_events(
                    events_path,
                    runtime=metadata["runtime"],
                    sample_id=sample_id,
                )
                if reconstructed:
                    traj["messages"] = reconstructed
            except Exception:
                logger.exception(
                    "Failed to reconstruct trajectory from raw driver lines for %s/%s",
                    trace_dir.name,
                    sample_id,
                )

    sample["trajectory"] = traj
    return normalize_sample_payload(sample)
