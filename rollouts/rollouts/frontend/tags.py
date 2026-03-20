from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def tags_path(run_dir: Path) -> Path:
    return run_dir / "tags.json"


def load_user_tags(run_dir: Path) -> dict[str, str]:
    path = tags_path(run_dir)
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"tags.json must contain a JSON object: {path}")
    tags: dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError(f"tags.json keys must be strings: {path}")
        if not isinstance(value, str):
            raise ValueError(f"tags.json values must be strings: {path}")
        tags[key] = value
    return tags


def write_user_tags(run_dir: Path, tags: dict[str, str]) -> None:
    normalized: dict[str, str] = {}
    for key, value in tags.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("tag keys must be non-empty strings")
        if not isinstance(value, str):
            raise ValueError(f"tag value for {key!r} must be a string")
        normalized[key] = value
    path = tags_path(run_dir)
    path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n")


def update_user_tags(run_dir: Path, updates: dict[str, str | None]) -> dict[str, str]:
    tags = load_user_tags(run_dir)
    for key, value in updates.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("tag keys must be non-empty strings")
        if value is None:
            tags.pop(key, None)
            continue
        if not isinstance(value, str):
            raise ValueError(f"tag value for {key!r} must be a string or null")
        tags[key] = value
    write_user_tags(run_dir, tags)
    return tags


def derive_run_tags(report: dict[str, Any]) -> dict[str, str]:
    metrics = report.get("summary_metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    config = report.get("config", {})
    if not isinstance(config, dict):
        config = {}
    endpoint = config.get("endpoint", {})
    if not isinstance(endpoint, dict):
        endpoint = {}

    interrupted = bool(config.get("interrupted", False))
    total_samples = _as_int(report.get("total_samples", metrics.get("total_samples"))) or 0
    completion_rate = _as_float(metrics.get("completion_rate"))
    success_rate = _as_float(metrics.get("success_rate"))
    provider_errors = _as_int(metrics.get("provider_errors")) or 0
    failed_samples = _as_int(metrics.get("failed_samples")) or 0
    aborted_samples = _as_int(metrics.get("aborted_samples")) or 0

    completed = False
    if not interrupted:
        if completion_rate is not None:
            completed = completion_rate >= 1.0
        elif total_samples > 0:
            completed = provider_errors == 0 and failed_samples == 0 and aborted_samples == 0

    successful = False
    if completed:
        if success_rate is not None:
            successful = success_rate >= 1.0
        elif total_samples > 0:
            successful = provider_errors == 0 and failed_samples == 0 and aborted_samples == 0

    status = "aborted" if interrupted else "completed"
    if not completed and not interrupted:
        status = "incomplete"
    if failed_samples > 0 or provider_errors > 0:
        status = "failed"

    tags = {
        "status": status,
        "interrupted": _bool_tag(interrupted),
        "completed": _bool_tag(completed),
        "successful": _bool_tag(successful),
        "provider_errors": str(provider_errors),
        "failed_samples": str(failed_samples),
        "aborted_samples": str(aborted_samples),
        "total_samples": str(total_samples),
    }

    provider = endpoint.get("provider")
    if isinstance(provider, str) and provider:
        tags["provider"] = provider
    model = endpoint.get("model")
    if isinstance(model, str) and model:
        tags["model"] = model
    eval_name = report.get("eval_name")
    if isinstance(eval_name, str) and eval_name:
        tags["eval_name"] = eval_name
    return tags


def build_run_tags(
    run_dir: Path, report: dict[str, Any] | None = None
) -> dict[str, dict[str, str]]:
    return {
        "user": load_user_tags(run_dir),
        "derived": derive_run_tags(report or {}),
    }


def build_live_run_tags(run: dict[str, Any]) -> dict[str, dict[str, str]]:
    status = str(run.get("status", "running"))
    return {
        "user": {},
        "derived": {
            "status": status,
            "live": "true",
            "completed": _bool_tag(status == "completed"),
            "successful": "false",
        },
    }


def _bool_tag(value: bool) -> str:
    return "true" if value else "false"


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None
