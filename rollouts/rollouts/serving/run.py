#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, replace
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import trio
import trio_asyncio

from ..core import EvalConfig
from ..eval import evaluate
from ..eval.configs import (
    ExternalEndpoint,
    OwnedEndpoint,
    materialize_endpoint,
)
from ..eval.run import (
    _apply_endpoint_env_overrides,
    _build_stop_handler,
    _find_config_project_root,
    _lower_eval_stop_handler,
    _resolve_config_path,
    load_config_module,
)
from .configs import (
    EvalServingWorkload,
    ServingScenario,
    ServingWorkload,
    ToolCallVerifierWorkload,
    resolve_serving_scenario,
)
from .tool_call_verifier import run_tool_call_verifier_workload

logger = logging.getLogger(__name__)
_ENGINE_EVENT_NAMES = {"llm_call", "llm_first_token"}


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(payload)


def _percentile(values: list[float], pct: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _workload_span_paths(workload_dir: Path) -> list[Path]:
    sessions_dir = workload_dir / "sessions"
    if not sessions_dir.exists():
        return []
    return sorted(sessions_dir.glob("*/spans.jsonl"))


def _collect_workload_spans(workload_dir: Path) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for spans_path in _workload_span_paths(workload_dir):
        spans.extend(_iter_jsonl(spans_path))
    return spans


def _collect_engine_records(workload_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spans_path in _workload_span_paths(workload_dir):
        session_id = spans_path.parent.name
        for row in _iter_jsonl(spans_path):
            records.append({
                "type": "engine_result",
                "session_id": session_id,
                **row,
            })
    return records


def _collect_eval_records(workload_dir: Path) -> list[dict[str, Any]]:
    events_path = workload_dir / "events.jsonl"
    if not events_path.exists():
        return []
    rows = []
    for row in _iter_jsonl(events_path):
        event_name = row.get("event") or row.get("message")
        if event_name in _ENGINE_EVENT_NAMES:
            continue
        rows.append({
            "type": "eval_event",
            **row,
        })
    return rows


def _iter_sample_artifacts(workload_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    samples_dir = workload_dir / "samples"
    if not samples_dir.exists():
        return []

    artifacts: list[tuple[Path, dict[str, Any]]] = []
    for sample_path in sorted(samples_dir.glob("*.json")):
        try:
            sample = json.loads(sample_path.read_text())
        except Exception:
            logger.exception("Failed to read sample artifact: %s", sample_path)
            continue
        artifacts.append((sample_path, sample))
    return artifacts


def _summarize_spans(spans: list[dict[str, Any]]) -> dict[str, Any]:
    total_requests = len(spans)
    error_count = sum(1 for span in spans if span.get("error"))
    timeout_count_estimated = sum(
        1
        for span in spans
        if isinstance(span.get("error"), str)
        and ("timeout" in span["error"].lower() or "timed out" in span["error"].lower())
    )
    ttfts = [float(span["ttft_ms"]) for span in spans if span.get("ttft_ms") is not None]
    durations = [
        float(span["duration_ms"]) for span in spans if span.get("duration_ms") is not None
    ]
    output_tok_per_sec = [
        float(span["output_tokens"]) / (float(span["duration_ms"]) / 1000.0)
        for span in spans
        if span.get("output_tokens") not in (None, 0) and span.get("duration_ms") not in (None, 0)
    ]
    return {
        "llm_request_count_total": total_requests,
        "llm_request_error_count": error_count,
        "llm_request_success_rate": _safe_rate(total_requests - error_count, total_requests),
        "llm_timeout_count_estimated": timeout_count_estimated,
        "llm_timeout_rate_estimated": _safe_rate(timeout_count_estimated, total_requests),
        "llm_input_tokens_total": sum(int(span.get("input_tokens", 0) or 0) for span in spans),
        "llm_output_tokens_total": sum(int(span.get("output_tokens", 0) or 0) for span in spans),
        "llm_reasoning_tokens_total": sum(
            int(span.get("reasoning_tokens", 0) or 0) for span in spans
        ),
        "llm_cache_read_tokens_total": sum(
            int(span.get("cache_read_tokens", 0) or 0) for span in spans
        ),
        "llm_cache_write_tokens_total": sum(
            int(span.get("cache_write_tokens", 0) or 0) for span in spans
        ),
        "llm_duration_ms_mean": (sum(durations) / len(durations)) if durations else None,
        "llm_duration_ms_p50": _percentile(durations, 50),
        "llm_duration_ms_p95": _percentile(durations, 95),
        "llm_ttft_ms_mean": (sum(ttfts) / len(ttfts)) if ttfts else None,
        "llm_ttft_ms_p50": _percentile(ttfts, 50),
        "llm_ttft_ms_p95": _percentile(ttfts, 95),
        "llm_output_tokens_per_sec_mean": (
            sum(output_tok_per_sec) / len(output_tok_per_sec) if output_tok_per_sec else None
        ),
        "llm_output_tokens_per_sec_p50": _percentile(output_tok_per_sec, 50),
        "llm_output_tokens_per_sec_p95": _percentile(output_tok_per_sec, 95),
    }


def _summarize_completed_samples(
    sample_artifacts: list[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    llm_metrics: list[dict[str, Any]] = []
    tool_metrics: list[dict[str, Any]] = []
    llm_usage_rows: list[dict[str, Any]] = []

    for _, sample in sample_artifacts:
        metadata = dict(sample.get("metadata") or {})
        llm_metrics.extend(list(metadata.get("llm_call_metrics") or []))
        tool_metrics.extend(list(metadata.get("tool_execution_metrics") or []))
        semantic_trace = dict(metadata.get("semantic_trace") or {})
        llm_usage_rows.extend(
            dict((row or {}).get("usage") or {})
            for row in list(semantic_trace.get("llm_calls") or [])
        )

    llm_request_count_total = len(llm_metrics)
    llm_request_error_count = sum(
        1 for row in llm_metrics if row.get("status") not in (None, "success") or row.get("error")
    )
    llm_timeout_count_estimated = sum(
        1
        for row in llm_metrics
        if isinstance(row.get("error"), str)
        and ("timeout" in row["error"].lower() or "timed out" in row["error"].lower())
    )
    llm_durations = [
        float(row["duration_ms"]) for row in llm_metrics if row.get("duration_ms") is not None
    ]
    llm_ttfts = [float(row["ttft_ms"]) for row in llm_metrics if row.get("ttft_ms") is not None]
    llm_output_tok_per_sec = [
        float(row["tokens_out"]) / (float(row["duration_ms"]) / 1000.0)
        for row in llm_metrics
        if row.get("tokens_out") not in (None, 0) and row.get("duration_ms") not in (None, 0)
    ]

    tool_execution_count_total = len(tool_metrics)
    tool_execution_error_count = sum(
        1 for row in tool_metrics if bool(row.get("is_error")) or row.get("status") == "error"
    )
    tool_durations = [
        float(row["duration_ms"]) for row in tool_metrics if row.get("duration_ms") is not None
    ]

    return {
        "llm_requests": {
            "llm_request_count_total": llm_request_count_total,
            "llm_request_error_count": llm_request_error_count,
            "llm_request_success_rate": _safe_rate(
                llm_request_count_total - llm_request_error_count,
                llm_request_count_total,
            ),
            "llm_timeout_count_estimated": llm_timeout_count_estimated,
            "llm_timeout_rate_estimated": _safe_rate(
                llm_timeout_count_estimated,
                llm_request_count_total,
            ),
            "llm_input_tokens_total": sum(
                int(row.get("input_tokens", 0) or 0) for row in llm_usage_rows
            ),
            "llm_output_tokens_total": sum(
                int(row.get("output_tokens", 0) or 0) for row in llm_usage_rows
            ),
            "llm_reasoning_tokens_total": sum(
                int(row.get("reasoning_tokens", 0) or 0) for row in llm_usage_rows
            ),
            "llm_cache_read_tokens_total": sum(
                int(row.get("cache_read_tokens", 0) or 0) for row in llm_usage_rows
            ),
            "llm_cache_write_tokens_total": None,
            "llm_duration_ms_mean": (sum(llm_durations) / len(llm_durations))
            if llm_durations
            else None,
            "llm_duration_ms_p50": _percentile(llm_durations, 50),
            "llm_duration_ms_p95": _percentile(llm_durations, 95),
            "llm_ttft_ms_mean": (sum(llm_ttfts) / len(llm_ttfts)) if llm_ttfts else None,
            "llm_ttft_ms_p50": _percentile(llm_ttfts, 50),
            "llm_ttft_ms_p95": _percentile(llm_ttfts, 95),
            "llm_output_tokens_per_sec_mean": (
                sum(llm_output_tok_per_sec) / len(llm_output_tok_per_sec)
                if llm_output_tok_per_sec
                else None
            ),
            "llm_output_tokens_per_sec_p50": _percentile(llm_output_tok_per_sec, 50),
            "llm_output_tokens_per_sec_p95": _percentile(llm_output_tok_per_sec, 95),
        },
        "tools": {
            "tool_execution_count_total": tool_execution_count_total,
            "tool_execution_error_count": tool_execution_error_count,
            "tool_execution_success_rate": _safe_rate(
                tool_execution_count_total - tool_execution_error_count,
                tool_execution_count_total,
            )
            if tool_execution_count_total > 0
            else None,
            "tool_execution_duration_ms_mean": (
                sum(tool_durations) / len(tool_durations) if tool_durations else None
            ),
            "tool_execution_duration_ms_p50": _percentile(tool_durations, 50),
            "tool_execution_duration_ms_p95": _percentile(tool_durations, 95),
        },
    }


def _summarize_eval_records(eval_records: list[dict[str, Any]]) -> dict[str, Any]:
    tool_dispatch_rows = [
        row
        for row in eval_records
        if (row.get("event") or row.get("message")) == "tool_call_dispatch"
    ]
    tool_detect_rows = [
        row
        for row in eval_records
        if (row.get("event") or row.get("message")) == "tool_calls_detected"
    ]
    return {
        "tool_calls_detected_total": sum(int(row.get("count", 0) or 0) for row in tool_detect_rows),
        "tool_call_dispatch_total": len(tool_dispatch_rows),
        "tool_call_execute_total": sum(
            1 for row in tool_dispatch_rows if row.get("action") == "execute"
        ),
        "tool_call_parse_error_total": sum(
            1 for row in tool_dispatch_rows if row.get("action") == "parse_error"
        ),
        "tool_call_schema_error_total": sum(
            1 for row in tool_dispatch_rows if row.get("action") == "schema_error"
        ),
        "tool_call_missing_environment_total": sum(
            1 for row in tool_dispatch_rows if row.get("action") == "missing_environment"
        ),
    }


def _build_operational_metrics(
    *,
    workload_result: dict[str, Any],
    workload_dir: Path,
) -> dict[str, Any]:
    summary = dict(workload_result.get("summary_metrics") or {})
    total_samples = int(workload_result.get("total_samples", 0) or 0)
    successful_samples = int(summary.get("successful_samples", 0) or 0)
    provider_errors = int(summary.get("provider_errors", 0) or 0)
    failed_samples = int(summary.get("failed_samples", 0) or 0)
    aborted_samples = int(summary.get("aborted_samples", 0) or 0)
    sample_artifacts = _iter_sample_artifacts(workload_dir)
    completed_metrics = _summarize_completed_samples(sample_artifacts)
    spans = _collect_workload_spans(workload_dir)
    observed_span_metrics = _summarize_spans(spans)

    if workload_result.get("status") != "failed":
        completed_metrics["llm_requests"]["llm_cache_write_tokens_total"] = observed_span_metrics[
            "llm_cache_write_tokens_total"
        ]

    return {
        "samples": {
            "total": total_samples,
            "successful": successful_samples,
            "provider_errors": provider_errors,
            "failed": failed_samples,
            "aborted": aborted_samples,
            "request_success_rate": _safe_rate(successful_samples, total_samples),
            "provider_error_rate": _safe_rate(provider_errors, total_samples),
            "failed_rate": _safe_rate(failed_samples, total_samples),
            "aborted_rate": _safe_rate(aborted_samples, total_samples),
        },
        "llm_requests": completed_metrics["llm_requests"],
        "throughput": {
            "wall_time_seconds": summary.get("wall_time_seconds"),
            "requests_per_sec": summary.get("requests_per_sec"),
            "total_output_tokens_per_sec": summary.get("total_output_tokens_per_sec"),
        },
        "tools": completed_metrics["tools"],
        "debug": {
            "observed_llm_requests": observed_span_metrics,
            "sample_artifact_count": len(sample_artifacts),
            "workload_status": workload_result.get("status", "completed"),
        },
    }


def _build_workload_engine_report(
    *,
    workload_name: str,
    workload_result: dict[str, Any],
    workload_dir: Path,
) -> dict[str, Any]:
    engine_records = _collect_engine_records(workload_dir)
    return {
        "workload": workload_name,
        "status": workload_result.get("status", "completed"),
        "session_count": len({row["session_id"] for row in engine_records}),
        "requests": _summarize_spans(engine_records),
        "throughput": {
            "wall_time_seconds": (workload_result.get("summary_metrics") or {}).get(
                "wall_time_seconds"
            ),
            "requests_per_sec": (workload_result.get("summary_metrics") or {}).get(
                "requests_per_sec"
            ),
            "total_output_tokens_per_sec": (
                (workload_result.get("summary_metrics") or {}).get("total_output_tokens_per_sec")
            ),
        },
    }


def _build_workload_eval_report(
    *,
    workload_name: str,
    workload_result: dict[str, Any],
    workload_dir: Path,
) -> dict[str, Any]:
    summary = dict(workload_result.get("summary_metrics") or {})
    sample_artifacts = _iter_sample_artifacts(workload_dir)
    completed_metrics = _summarize_completed_samples(sample_artifacts)
    eval_records = _collect_eval_records(workload_dir)
    stop_reasons: dict[str, int] = {}
    for _, sample in sample_artifacts:
        stop_reason = (sample.get("metadata") or {}).get("stop_reason")
        if stop_reason:
            stop_reasons[str(stop_reason)] = stop_reasons.get(str(stop_reason), 0) + 1
    return {
        "workload": workload_name,
        "status": workload_result.get("status", "completed"),
        "samples": {
            "total": int(workload_result.get("total_samples", 0) or 0),
            "successful": int(summary.get("successful_samples", 0) or 0),
            "provider_errors": int(summary.get("provider_errors", 0) or 0),
            "failed": int(summary.get("failed_samples", 0) or 0),
            "aborted": int(summary.get("aborted_samples", 0) or 0),
            "request_success_rate": _safe_rate(
                int(summary.get("successful_samples", 0) or 0),
                int(workload_result.get("total_samples", 0) or 0),
            ),
        },
        "outcomes": {
            "mean_passed": summary.get("mean_passed"),
            "mean_reward": summary.get("mean_reward"),
            "avg_turns": summary.get("avg_turns"),
            "avg_tokens": summary.get("avg_tokens"),
            "stop_reasons": stop_reasons,
        },
        "tool_calls": _summarize_eval_records(eval_records),
        "tool_executions": completed_metrics["tools"],
    }


def _aggregate_engine_reports(workloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    request_metrics = {name: dict(report["requests"]) for name, report in workloads.items()}
    return {
        "session_count": sum(
            int(report.get("session_count", 0) or 0) for report in workloads.values()
        ),
        "requests": _aggregate_operational_metrics({
            name: {
                "samples": {
                    "total": 0,
                    "successful": 0,
                    "provider_errors": 0,
                    "failed": 0,
                    "aborted": 0,
                },
                "llm_requests": metrics,
                "throughput": {
                    "wall_time_seconds": None,
                    "requests_per_sec": None,
                    "total_output_tokens_per_sec": None,
                },
                "tools": {"tool_execution_count_total": 0, "tool_execution_error_count": 0},
            }
            for name, metrics in request_metrics.items()
        })["llm_requests"],
        "throughput": {
            "wall_time_seconds_sum": sum(
                float(report["throughput"]["wall_time_seconds"] or 0.0)
                for report in workloads.values()
                if report["throughput"]["wall_time_seconds"] is not None
            )
            if any(
                report["throughput"]["wall_time_seconds"] is not None
                for report in workloads.values()
            )
            else None,
            "requests_per_sec_sum": sum(
                float(report["throughput"]["requests_per_sec"] or 0.0)
                for report in workloads.values()
                if report["throughput"]["requests_per_sec"] is not None
            )
            if any(
                report["throughput"]["requests_per_sec"] is not None
                for report in workloads.values()
            )
            else None,
            "total_output_tokens_per_sec_sum": sum(
                float(report["throughput"]["total_output_tokens_per_sec"] or 0.0)
                for report in workloads.values()
                if report["throughput"]["total_output_tokens_per_sec"] is not None
            )
            if any(
                report["throughput"]["total_output_tokens_per_sec"] is not None
                for report in workloads.values()
            )
            else None,
        },
    }


def _aggregate_eval_reports(workloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": {
            "total": sum(int(report["samples"]["total"] or 0) for report in workloads.values()),
            "successful": sum(
                int(report["samples"]["successful"] or 0) for report in workloads.values()
            ),
            "provider_errors": sum(
                int(report["samples"]["provider_errors"] or 0) for report in workloads.values()
            ),
            "failed": sum(int(report["samples"]["failed"] or 0) for report in workloads.values()),
            "aborted": sum(int(report["samples"]["aborted"] or 0) for report in workloads.values()),
        },
        "tool_calls": {
            key: sum(int(report["tool_calls"][key] or 0) for report in workloads.values())
            for key in (
                "tool_calls_detected_total",
                "tool_call_dispatch_total",
                "tool_call_execute_total",
                "tool_call_parse_error_total",
                "tool_call_schema_error_total",
                "tool_call_missing_environment_total",
            )
        },
        "tool_executions": {
            "tool_execution_count_total": sum(
                int(report["tool_executions"]["tool_execution_count_total"] or 0)
                for report in workloads.values()
            ),
            "tool_execution_error_count": sum(
                int(report["tool_executions"]["tool_execution_error_count"] or 0)
                for report in workloads.values()
            ),
        },
    }


def _write_split_workload_artifacts(
    *,
    scenario_report: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    workloads_root = output_dir / "workloads"
    engine_reports: dict[str, dict[str, Any]] = {}
    eval_reports: dict[str, dict[str, Any]] = {}
    for workload_name, workload_result in scenario_report["workloads"].items():
        workload_dir = workloads_root / workload_name
        kind = workload_result.get("kind", "eval")

        if kind == "tool_call_verifier":
            # The conformance runner already wrote engine.jsonl + engine_report.json
            # directly; don't overwrite with eval-shaped aggregation. There is no
            # per-sample eval truth for this workload kind, so eval_report is a stub.
            engine_report_path = workload_dir / "engine_report.json"
            if engine_report_path.exists():
                engine_reports[workload_name] = json.loads(engine_report_path.read_text())
            else:
                engine_reports[workload_name] = {"status": "missing"}
            eval_report_stub = {
                "workload": workload_name,
                "kind": "tool_call_verifier",
                "note": "no eval artifacts for tool_call_verifier workload",
            }
            (workload_dir / "eval_report.json").write_text(json.dumps(eval_report_stub, indent=2))
            eval_reports[workload_name] = eval_report_stub
            continue

        engine_records = _collect_engine_records(workload_dir)
        _write_jsonl(workload_dir / "engine.jsonl", engine_records)
        eval_records = _collect_eval_records(workload_dir)
        _write_jsonl(workload_dir / "eval.jsonl", eval_records)

        engine_report = _build_workload_engine_report(
            workload_name=workload_name,
            workload_result=workload_result,
            workload_dir=workload_dir,
        )
        eval_report = _build_workload_eval_report(
            workload_name=workload_name,
            workload_result=workload_result,
            workload_dir=workload_dir,
        )
        (workload_dir / "engine_report.json").write_text(json.dumps(engine_report, indent=2))
        (workload_dir / "eval_report.json").write_text(json.dumps(eval_report, indent=2))
        engine_reports[workload_name] = engine_report
        eval_reports[workload_name] = eval_report
    return engine_reports, eval_reports


def _load_workload_report_summary(workload_dir: Path) -> tuple[dict[str, Any], int]:
    report_path = workload_dir / "report.json"
    if not report_path.exists():
        return {}, 0
    try:
        report = json.loads(report_path.read_text())
    except Exception:
        logger.exception("Failed to read partial workload report: %s", report_path)
        return {}, 0
    return dict(report.get("summary_metrics") or {}), int(report.get("total_samples", 0) or 0)


def _aggregate_operational_metrics(
    workloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    def _sum_known(values: list[float | int | None]) -> float | int | None:
        known = [value for value in values if value is not None]
        if not known:
            return None
        return sum(known)

    total_samples = sum(int(w["samples"]["total"] or 0) for w in workloads.values())
    successful_samples = sum(int(w["samples"]["successful"] or 0) for w in workloads.values())
    provider_errors = sum(int(w["samples"]["provider_errors"] or 0) for w in workloads.values())
    failed_samples = sum(int(w["samples"]["failed"] or 0) for w in workloads.values())
    aborted_samples = sum(int(w["samples"]["aborted"] or 0) for w in workloads.values())
    llm_request_count_total = sum(
        int(w["llm_requests"]["llm_request_count_total"] or 0) for w in workloads.values()
    )
    llm_request_error_count = sum(
        int(w["llm_requests"]["llm_request_error_count"] or 0) for w in workloads.values()
    )
    llm_timeout_count_estimated = sum(
        int(w["llm_requests"]["llm_timeout_count_estimated"] or 0) for w in workloads.values()
    )
    tool_execution_count_total = sum(
        int(w["tools"]["tool_execution_count_total"] or 0) for w in workloads.values()
    )
    tool_execution_error_count = sum(
        int(w["tools"]["tool_execution_error_count"] or 0) for w in workloads.values()
    )
    cache_write_values = [
        w["llm_requests"].get("llm_cache_write_tokens_total") for w in workloads.values()
    ]
    aggregate = {
        "samples": {
            "total": total_samples,
            "successful": successful_samples,
            "provider_errors": provider_errors,
            "failed": failed_samples,
            "aborted": aborted_samples,
            "request_success_rate": _safe_rate(successful_samples, total_samples),
            "provider_error_rate": _safe_rate(provider_errors, total_samples),
            "failed_rate": _safe_rate(failed_samples, total_samples),
            "aborted_rate": _safe_rate(aborted_samples, total_samples),
        },
        "llm_requests": {
            "llm_request_count_total": llm_request_count_total,
            "llm_request_error_count": llm_request_error_count,
            "llm_request_success_rate": _safe_rate(
                llm_request_count_total - llm_request_error_count,
                llm_request_count_total,
            ),
            "llm_timeout_count_estimated": llm_timeout_count_estimated,
            "llm_timeout_rate_estimated": _safe_rate(
                llm_timeout_count_estimated,
                llm_request_count_total,
            ),
            "llm_input_tokens_total": sum(
                int(w["llm_requests"]["llm_input_tokens_total"] or 0) for w in workloads.values()
            ),
            "llm_output_tokens_total": sum(
                int(w["llm_requests"]["llm_output_tokens_total"] or 0) for w in workloads.values()
            ),
            "llm_reasoning_tokens_total": sum(
                int(w["llm_requests"]["llm_reasoning_tokens_total"] or 0)
                for w in workloads.values()
            ),
            "llm_cache_read_tokens_total": sum(
                int(w["llm_requests"]["llm_cache_read_tokens_total"] or 0)
                for w in workloads.values()
            ),
            "llm_cache_write_tokens_total": (
                None
                if any(value is None for value in cache_write_values)
                else sum(int(value or 0) for value in cache_write_values)
            ),
        },
        "throughput": {
            "wall_time_seconds_sum": _sum_known([
                w["throughput"].get("wall_time_seconds") for w in workloads.values()
            ]),
            "requests_per_sec_sum": _sum_known([
                w["throughput"].get("requests_per_sec") for w in workloads.values()
            ]),
            "total_output_tokens_per_sec_sum": _sum_known([
                w["throughput"].get("total_output_tokens_per_sec") for w in workloads.values()
            ]),
        },
        "tools": {
            "tool_execution_count_total": tool_execution_count_total,
            "tool_execution_error_count": tool_execution_error_count,
            "tool_execution_success_rate": (
                _safe_rate(
                    tool_execution_count_total - tool_execution_error_count,
                    tool_execution_count_total,
                )
                if tool_execution_count_total > 0
                else None
            ),
        },
    }
    return aggregate


def _collect_sample_failures(
    *,
    workload_name: str,
    workload_dir: Path,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for sample_path, sample in _iter_sample_artifacts(workload_dir):
        metadata = dict(sample.get("metadata") or {})
        status = metadata.get("status", "success")
        if status == "success":
            continue

        problem = dict(sample.get("problem") or {})
        payload = dict(problem.get("payload") or {})
        failures.append({
            "workload": workload_name,
            "sample_id": sample.get("attempt_id"),
            "problem_id": problem.get("problem_id"),
            "task_id": payload.get("task_id"),
            "status": status,
            "error": metadata.get("error"),
            "error_type": metadata.get("error_type"),
            "path": str(sample_path),
        })

    return failures


def _collect_interrupted_samples(
    *,
    workload_name: str,
    workload_dir: Path,
) -> list[dict[str, Any]]:
    events_path = workload_dir / "events.jsonl"
    if not events_path.exists():
        return []

    finalized_ids = {
        str(sample.get("attempt_id"))
        for _, sample in _iter_sample_artifacts(workload_dir)
        if sample.get("attempt_id") is not None
    }
    observed: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(events_path):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in finalized_ids:
            continue
        snapshot = observed.setdefault(
            sample_id,
            {
                "workload": workload_name,
                "sample_id": sample_id,
                "event_count": 0,
                "last_event": None,
                "last_status": None,
                "last_timestamp": None,
            },
        )
        snapshot["event_count"] += 1
        snapshot["last_event"] = row.get("event")
        snapshot["last_status"] = row.get("status")
        snapshot["last_timestamp"] = row.get("timestamp")

    return list(observed.values())


def _build_failure_diagnostics(
    *,
    scenario_report: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    workloads_root = output_dir / "workloads"
    workload_failures = [
        {
            "workload": workload_name,
            "error": workload_result.get("error"),
            "report_path": workload_result.get("report_path"),
            "events_path": workload_result.get("events_path"),
        }
        for workload_name, workload_result in scenario_report["workloads"].items()
        if workload_result.get("status") == "failed"
    ]
    sample_failures = [
        failure
        for workload_name in scenario_report["workloads"]
        for failure in _collect_sample_failures(
            workload_name=workload_name,
            workload_dir=workloads_root / workload_name,
        )
    ]
    interrupted_samples = [
        interrupted
        for workload_name in scenario_report["workloads"]
        for interrupted in _collect_interrupted_samples(
            workload_name=workload_name,
            workload_dir=workloads_root / workload_name,
        )
    ]
    return {
        "experiment_name": scenario_report["experiment_name"],
        "output_dir": scenario_report["output_dir"],
        "config_path": scenario_report["config_path"],
        "generated_at": datetime.now().isoformat(),
        "aggregate": {
            "workload_failure_count": len(workload_failures),
            "sample_failure_count": len(sample_failures),
            "interrupted_sample_count": len(interrupted_samples),
        },
        "workload_failures": workload_failures,
        "sample_failures": sample_failures,
        "interrupted_samples": interrupted_samples,
    }


def _resolve_output_dir(
    *,
    config_path: Path,
    scenario: ServingScenario,
    cli_output_dir: Path | None = None,
) -> Path:
    explicit_output_dir = os.environ.get("ROLLOUTS_OUTPUT_DIR")
    if explicit_output_dir:
        return Path(explicit_output_dir)

    project_root = _find_config_project_root(config_path)
    if cli_output_dir is not None:
        return cli_output_dir if cli_output_dir.is_absolute() else (project_root / cli_output_dir)

    configured_output_dir = scenario.output.output_dir
    if configured_output_dir is not None:
        return (
            configured_output_dir
            if configured_output_dir.is_absolute()
            else (project_root / configured_output_dir)
        )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return project_root / "results" / "serving" / f"{scenario.output.experiment_name}_{timestamp}"


def _sanitize_endpoint(endpoint_config: Any) -> dict[str, Any] | None:
    if endpoint_config is None:
        return None
    payload = asdict(endpoint_config)
    if "api_key" in payload and payload["api_key"]:
        payload["api_key"] = "***REDACTED***"
    if "output_dir" in payload and payload["output_dir"] is not None:
        payload["output_dir"] = str(payload["output_dir"])
    if isinstance(endpoint_config, OwnedEndpoint):
        payload["provider"] = endpoint_config.provider
        payload["base_url"] = endpoint_config.base_url
    elif isinstance(endpoint_config, ExternalEndpoint):
        payload["base_url"] = endpoint_config.base_url
    return payload


def _load_tasks(eval_task: Any) -> list[dict[str, Any]]:
    if eval_task.tasks is not None:
        tasks = eval_task.tasks
    elif eval_task.tasks_path is not None:
        data = json.loads(eval_task.tasks_path.read_text())
        tasks = data if isinstance(data, list) else data.get("tasks", [])
    else:
        raise ValueError("EvalTaskSpec must define tasks or tasks_path")

    if not isinstance(tasks, list):
        raise ValueError("EvalTaskSpec tasks must resolve to a list")
    return tasks


def _rewrite_workload_eval_task(
    *,
    workload: EvalServingWorkload,
    endpoint_config: Any,
    workload_output_dir: Path,
) -> Any:
    eval_task = workload.eval_task
    rewritten_run = replace(
        eval_task.run,
        max_concurrent=workload.concurrency,
        max_samples=(
            workload.max_samples if workload.max_samples is not None else eval_task.run.max_samples
        ),
    )
    rewritten_output = replace(
        eval_task.output,
        experiment_name=workload.name,
        output_dir=workload_output_dir,
    )
    rewritten_run_spec = replace(eval_task.run_spec, endpoint=endpoint_config)
    return replace(
        eval_task,
        run=rewritten_run,
        output=rewritten_output,
        run_spec=rewritten_run_spec,
    )


async def _run_workload(
    *,
    workload: EvalServingWorkload,
    eval_task: Any,
) -> dict[str, Any]:
    tasks = _load_tasks(eval_task)
    if eval_task.run.max_samples is not None:
        tasks = tasks[: eval_task.run.max_samples]

    run_spec = eval_task.run_spec
    endpoint = materialize_endpoint(run_spec.endpoint) if run_spec.endpoint is not None else None

    async def silent_on_chunk(_: object) -> None:
        pass

    from ..agents import AgentState, StopReason
    from ..agents import RunConfig as AgentRunConfig

    async def stop_on_no_tool(state: AgentState, _run_config: AgentRunConfig) -> AgentState:
        return replace(state, stop=StopReason.TASK_COMPLETED)

    handle_stop = _build_stop_handler(eval_task.run)
    if run_spec.stop_handler is not None:
        handle_stop = _lower_eval_stop_handler(run_spec.stop_handler)

    handle_no_tool = run_spec.handle_no_tool or stop_on_no_tool
    agent_run_config = AgentRunConfig(
        on_chunk=silent_on_chunk,
        handle_stop=handle_stop,
        handle_no_tool=handle_no_tool,
    )

    report = await evaluate(
        iter(tasks),
        EvalConfig(
            endpoint=endpoint,
            scorer=eval_task.scorer,
            prepare_messages=run_spec.prepare_messages,
            environment=run_spec.environment,
            environment_factory=run_spec.environment_factory,
            attempt_executor=run_spec.attempt_executor,
            run_config=agent_run_config,
            max_samples=len(tasks),
            max_concurrent=eval_task.run.max_concurrent,
            max_api_concurrent=eval_task.run.max_api_concurrent,
            max_tool_concurrent=eval_task.run.max_tool_concurrent,
            verbose=eval_task.run.verbose,
            output_dir=eval_task.output.output_dir,
            eval_name=eval_task.output.experiment_name,
            show_progress=eval_task.run.show_progress,
            stream_tokens=eval_task.run.stream_tokens,
            max_sample_retries=eval_task.run.max_sample_retries,
            summary_distribution_percentiles=eval_task.summary_distribution_percentiles,
        ),
    )
    return {
        "workload": workload.name,
        "report_path": str(eval_task.output.output_dir / "report.json"),
        "events_path": str(eval_task.output.output_dir / "events.jsonl"),
        "samples_dir": str(eval_task.output.output_dir / "samples"),
        "summary_metrics": report.summary_metrics,
        "total_samples": report.total_samples,
    }


def _manifest_entry_for_workload(
    workload: ServingWorkload,
    workload_dir: Path,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": workload.name,
        "concurrency": workload.concurrency,
        "max_samples": workload.max_samples,
        "output_dir": str(workload_dir),
    }
    if isinstance(workload, EvalServingWorkload):
        base["kind"] = "eval"
        base["source_eval_name"] = workload.eval_task.output.experiment_name
    elif isinstance(workload, ToolCallVerifierWorkload):
        base["kind"] = "tool_call_verifier"
        base["corpus_path"] = (
            str(workload.corpus_path) if workload.corpus_path is not None else None
        )
    else:
        raise AssertionError(f"Unhandled workload variant: {type(workload)!r}")
    return base


def _scenario_manifest(
    *,
    scenario: ServingScenario,
    endpoint_config: Any,
    output_dir: Path,
    workload_dirs: dict[str, Path],
) -> dict[str, Any]:
    return {
        "experiment_name": scenario.output.experiment_name,
        "output_dir": str(output_dir),
        "started_at": datetime.now().isoformat(),
        "endpoint": _sanitize_endpoint(endpoint_config),
        "workloads": [
            _manifest_entry_for_workload(workload, workload_dirs[workload.name])
            for workload in scenario.workloads
        ],
    }


def _engine_metrics_interval_s(scenario: ServingScenario) -> float:
    """Choose a scrape cadence at the serving-run boundary.

    Short finite runs benefit from denser scrapes for trend inspection.
    Long soaks keep the poller's 5s default so telemetry volume does not
    silently triple.
    """
    if scenario.duration is None:
        return 2.0
    return 5.0 if scenario.duration.total_seconds() >= 3600.0 else 2.0


async def _run_scenario(
    config_path: Path,
    scenario: ServingScenario,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    workloads_dir = output_dir / "workloads"
    workloads_dir.mkdir(parents=True, exist_ok=True)

    endpoint_config = _apply_endpoint_env_overrides(scenario.endpoint)
    workload_dirs = {
        workload.name: workloads_dir / workload.name for workload in scenario.workloads
    }
    manifest = _scenario_manifest(
        scenario=scenario,
        endpoint_config=endpoint_config,
        output_dir=output_dir,
        workload_dirs=workload_dirs,
    )
    if scenario.output.save_manifest:
        (output_dir / "scenario_manifest.json").write_text(json.dumps(manifest, indent=2))

    results: dict[str, dict[str, Any]] = {}
    scenario_base_url = getattr(endpoint_config, "base_url", None)
    if scenario_base_url is None:
        scenario_base_url = endpoint_config.get_base_url()
    assert isinstance(scenario_base_url, str) and scenario_base_url

    async def _run_eval_servng_workload(workload: EvalServingWorkload) -> None:
        workload_output_dir = workload_dirs[workload.name]
        workload_output_dir.mkdir(parents=True, exist_ok=True)
        eval_task = _rewrite_workload_eval_task(
            workload=workload,
            endpoint_config=endpoint_config,
            workload_output_dir=workload_output_dir,
        )
        try:
            result = await _run_workload(workload=workload, eval_task=eval_task)
            result["kind"] = "eval"
            results[workload.name] = result
        except Exception as exc:
            logger.exception("Workload %s failed", workload.name)
            summary_metrics, total_samples = _load_workload_report_summary(workload_output_dir)
            results[workload.name] = {
                "workload": workload.name,
                "kind": "eval",
                "report_path": str(workload_output_dir / "report.json"),
                "events_path": str(workload_output_dir / "events.jsonl"),
                "samples_dir": str(workload_output_dir / "samples"),
                "summary_metrics": summary_metrics,
                "total_samples": total_samples,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

    async def _run_tool_call_verifier_serving_workload(
        workload: ToolCallVerifierWorkload,
    ) -> None:
        workload_output_dir = workload_dirs[workload.name]
        workload_output_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = await run_tool_call_verifier_workload(
                base_url=scenario_base_url,
                api_key=getattr(endpoint_config, "api_key", None) or None,
                model=endpoint_config.model,
                corpus_path=workload.corpus_path,
                max_samples=workload.max_samples,
                concurrency=workload.concurrency,
                extra_body=workload.extra_body,
                output_dir=workload_output_dir,
                request_timeout_s=workload.request_timeout_s,
            )
            result["workload"] = workload.name
            result["kind"] = "tool_call_verifier"
            result["total_samples"] = result.get("row_count", 0)
            result["summary_metrics"] = {}
            results[workload.name] = result
        except Exception as exc:
            logger.exception("Workload %s failed", workload.name)
            results[workload.name] = {
                "workload": workload.name,
                "kind": "tool_call_verifier",
                "engine_report_path": str(workload_output_dir / "engine_report.json"),
                "engine_jsonl_path": str(workload_output_dir / "engine.jsonl"),
                "summary_metrics": {},
                "total_samples": 0,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

    async def _run_named_workload(workload: ServingWorkload) -> None:
        # TODO(workload-shape): ToolCallVerifierWorkload dispatches into its
        # own runner (serving/tool_call_verifier/runner.py) that bypasses
        # the agent loop entirely to get raw-response access for K2VV
        # fidelity. That's why its span tree under the workload row looks
        # different from EvalServingWorkload's (no sample_attempt /
        # agent_step layers — spans go straight from workload -> llm_call).
        # This is a smell: the "workload" concept forks into two code paths
        # that agree only at the outer boundary. Before adding a third
        # conformance-style workload (schema probes, structured-output
        # probes, etc), factor out the shared "iterate rows, send, classify,
        # emit" shape so all workloads land in the same span hierarchy and
        # reports have the same structure. See the matching TODO in
        # serving/tool_call_verifier/runner.py docstring.
        from .._observability import start_span

        async with start_span(
            "workload",
            attributes={
                "workload.name": workload.name,
                "workload.kind": "eval"
                if isinstance(workload, EvalServingWorkload)
                else "tool_call_verifier",
                "workload.concurrency": workload.concurrency,
                "workload.max_samples": workload.max_samples,
            },
        ):
            if isinstance(workload, EvalServingWorkload):
                await _run_eval_servng_workload(workload)
            elif isinstance(workload, ToolCallVerifierWorkload):
                await _run_tool_call_verifier_serving_workload(workload)
            else:
                raise AssertionError(f"Unhandled workload variant: {type(workload)!r}")

    async def _run_all_workloads(parent_nursery: trio.Nursery) -> None:
        try:
            async with trio.open_nursery() as workload_nursery:
                for workload in scenario.workloads:
                    workload_nursery.start_soon(_run_named_workload, workload)
        finally:
            parent_nursery.cancel_scope.cancel()

    # Drain-aware nursery: if the supervisor sends SIGINT/SIGTERM to this
    # child process (duration elapsed, operator stop, etc), Python raises
    # KeyboardInterrupt which trio repackages as BaseExceptionGroup. We
    # catch that and treat it as "the scenario was asked to stop" rather
    # than "the scenario crashed" — per-workload reports that finished
    # still count, scenario_report is still written, exit code is 0.
    scenario_drained = False
    from .._observability import new_trace_id, start_span

    # serving_run span: one root per ServingRun. Trace id is minted here and
    # inherited by every workload span. Each sample_attempt nested inside a
    # workload will still get its own trace_id (samples are independent units
    # of work); serving_run and workload spans are cross-trace anchors.
    async with start_span(
        "serving_run",
        trace_id=new_trace_id(),
        attributes={
            "serving_run.experiment_name": scenario.output.experiment_name,
            "serving_run.workload_count": len(scenario.workloads),
        },
    ):
        try:
            from .._observability.engine_metrics import poll_engine_metrics

            async with trio.open_nursery() as nursery:
                if endpoint_config.provider == "sglang":
                    nursery.start_soon(
                        partial(
                            poll_engine_metrics,
                            base_url=scenario_base_url,
                            output_path=output_dir / "engine_metrics.jsonl",
                            interval_s=_engine_metrics_interval_s(scenario),
                            cancel_scope=nursery.cancel_scope,
                        )
                    )
                nursery.start_soon(_run_all_workloads, nursery)
        except BaseExceptionGroup as eg:  # noqa: F821,UP041 — stdlib class, present 3.11+
            # Walk the nested ExceptionGroup leaves. If every leaf is a
            # KeyboardInterrupt or trio.Cancelled, this was a shutdown request
            # (duration watchdog → supervisor → SIGINT → Python KeyboardInterrupt
            # in child → trio Cancel). Otherwise it's a real failure.
            def _leaves(exc: BaseException) -> list[BaseException]:
                if isinstance(exc, BaseExceptionGroup):  # noqa: F821 — stdlib 3.11+
                    out: list[BaseException] = []
                    for sub in exc.exceptions:
                        out.extend(_leaves(sub))
                    return out
                return [exc]

            stop_types = (KeyboardInterrupt, trio.Cancelled)
            all_leaves = _leaves(eg)
            if all_leaves and all(isinstance(leaf, stop_types) for leaf in all_leaves):
                scenario_drained = True
            else:
                raise

    failed_workloads = {
        name: result for name, result in results.items() if result.get("status") == "failed"
    }
    scenario_report = {
        "experiment_name": scenario.output.experiment_name,
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "endpoint": _sanitize_endpoint(endpoint_config),
        "workloads": results,
        "total_samples": sum(int(result["total_samples"]) for result in results.values()),
        "completed_workloads": len(results),
        "failed_workloads": len(failed_workloads),
        "status": "drained" if scenario_drained else "completed",
    }
    if scenario.output.save_report:
        (output_dir / "scenario_report.json").write_text(json.dumps(scenario_report, indent=2))
        engine_reports, eval_reports = _write_split_workload_artifacts(
            scenario_report=scenario_report,
            output_dir=output_dir,
        )
        # Scenario-level aggregators assume eval-shaped per-workload reports.
        # Conformance (tool_call_verifier) workloads have a K2VV-shape report
        # instead; partition them so each aggregator only sees compatible input.
        eval_engine_reports = {
            name: report
            for name, report in engine_reports.items()
            if scenario_report["workloads"][name].get("kind", "eval") == "eval"
        }
        conformance_engine_reports = {
            name: report
            for name, report in engine_reports.items()
            if scenario_report["workloads"][name].get("kind") == "tool_call_verifier"
        }
        (output_dir / "engine_report.json").write_text(
            json.dumps(
                {
                    "experiment_name": scenario_report["experiment_name"],
                    "output_dir": scenario_report["output_dir"],
                    "config_path": scenario_report["config_path"],
                    "generated_at": datetime.now().isoformat(),
                    "aggregate": _aggregate_engine_reports(eval_engine_reports),
                    "workloads": engine_reports,
                    "conformance_workloads": list(conformance_engine_reports.keys()),
                },
                indent=2,
            )
        )
        eval_only_reports = {
            name: report
            for name, report in eval_reports.items()
            if scenario_report["workloads"][name].get("kind", "eval") == "eval"
        }
        (output_dir / "eval_report.json").write_text(
            json.dumps(
                {
                    "experiment_name": scenario_report["experiment_name"],
                    "output_dir": scenario_report["output_dir"],
                    "config_path": scenario_report["config_path"],
                    "generated_at": datetime.now().isoformat(),
                    "aggregate": _aggregate_eval_reports(eval_only_reports),
                    "workloads": eval_reports,
                    "failures": _build_failure_diagnostics(
                        scenario_report=scenario_report,
                        output_dir=output_dir,
                    ),
                },
                indent=2,
            )
        )
    return scenario_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a serving scenario")
    parser.add_argument("--config", type=Path, required=True, help="Config file path")
    parser.add_argument("--output-dir", type=Path, help="Explicit output directory")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config_path = _resolve_config_path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    try:
        config_module = load_config_module(config_path)
        scenario = resolve_serving_scenario(config_module)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    output_dir = _resolve_output_dir(
        config_path=config_path,
        scenario=scenario,
        cli_output_dir=args.output_dir,
    ).resolve()

    print(f"Config: {config_path}")
    print(f"Scenario: {scenario.output.experiment_name}")
    print(f"Workloads: {', '.join(workload.name for workload in scenario.workloads)}")
    print(f"Output dir: {output_dir}")

    # Bind a span sink for this serving-child process. spans.jsonl is shared
    # with the parent serving supervisor (see parent process binding). POSIX
    # atomic-append on writes under 16KB keeps the two writers from
    # interleaving.
    from .._observability import SpanSink, set_sink

    span_sink = SpanSink(output_dir / "spans.jsonl")
    set_sink(span_sink)
    try:
        scenario_report = trio_asyncio.run(_run_scenario, config_path, scenario, output_dir)
    except KeyboardInterrupt:
        # Supervisor asked us to stop before we even started the nursery
        # (or between workloads exiting and reporting). Treat as drained.
        # In practice _run_scenario now catches cancellation internally, so
        # this branch is a belt-and-suspenders guard.
        logger.info("Serving scenario interrupted before completion")
        return 0
    except Exception as exc:
        logger.exception("Serving scenario failed: %s", exc)
        return 1
    finally:
        span_sink.close()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  completed_workloads: {scenario_report['completed_workloads']}")
    print(f"  total_samples: {scenario_report['total_samples']}")
    for workload_name, workload_result in scenario_report["workloads"].items():
        print(f"  {workload_name}: {workload_result['total_samples']} samples")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
