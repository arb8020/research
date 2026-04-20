"""ToolCallVerifierWorkload runner: iterate K2VV-shape rows, classify, emit.

Public entrypoint `run_tool_call_verifier_workload` is called from
serving/run.py's sum-type dispatch. Produces engine.jsonl +
engine_report.json under the workload's output dir, matching the
artifact shape of the existing eval workload branch.

# TODO(workload-shape): ToolCallVerifierWorkload bypasses run_agent entirely
# because we need raw-response access for K2VV fidelity. If we ever add more
# request/response-conformance workloads (schema probes, structured-output
# probes, etc.) they'll want the same "send -> classify -> emit" shape.
# Extract that shape before adding a third such workload; don't let each
# new conformance variant invent its own branch in serving/run.py.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import trio

from .aggregator import send_row
from .classifier import compute_summary, validate_tool_call
from .corpus import ensure_k2vv_corpus

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_corpus(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if max_samples is not None and len(rows) >= max_samples:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping row %d: %s", line_num, exc)
                continue
            row.setdefault("_data_index", line_num)
            rows.append(row)
    return rows


def _classify_response(
    response: dict[str, Any],
    tools: list[dict[str, Any]],
) -> tuple[str | None, bool | None]:
    choices = response.get("choices") or []
    if not choices:
        return None, None
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    if finish_reason != "tool_calls":
        return finish_reason, None

    tool_calls = (choice.get("message") or {}).get("tool_calls") or []
    if not tool_calls:
        # Server said tool_calls but gave us none — count as schema failure.
        return finish_reason, False
    all_valid = all(validate_tool_call(tc, tools) for tc in tool_calls)
    return finish_reason, all_valid


async def _process_one_row(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str | None,
    row: dict[str, Any],
    model: str,
    extra_body: dict[str, Any] | None,
    limiter: trio.CapacityLimiter,
) -> dict[str, Any]:
    async with limiter:
        data_index = row.pop("_data_index", None)
        start = time.time()
        status, response = await send_row(
            client=client,
            base_url=base_url,
            api_key=api_key,
            row=row,
            model=model,
            extra_body=extra_body,
        )
        duration_ms = int((time.time() - start) * 1000)

        finish_reason: str | None = None
        tool_calls_valid: bool | None = None
        if status == "success":
            tools = row.get("tools") or []
            finish_reason, tool_calls_valid = _classify_response(response, tools)

        return {
            "data_index": data_index,
            "request": row,
            "extra_body": extra_body or {},
            "response": response,
            "status": status,
            "finish_reason": finish_reason,
            "tool_calls_valid": tool_calls_valid,
            "last_run_at": _now_iso(),
            "duration_ms": duration_ms,
        }


async def run_tool_call_verifier_workload(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    corpus_path: Path | None,
    max_samples: int | None,
    concurrency: int,
    extra_body: dict[str, Any] | None,
    output_dir: Path,
    request_timeout_s: float,
) -> dict[str, Any]:
    """Run the verifier workload end-to-end. Returns a workload result dict.

    Writes engine.jsonl (per-row) and engine_report.json (summary) into
    output_dir. The returned dict is the shape serving/run.py aggregates
    into scenario_report["workloads"][<name>].
    """
    assert concurrency > 0, "concurrency must be positive"

    output_dir.mkdir(parents=True, exist_ok=True)
    engine_jsonl = output_dir / "engine.jsonl"
    engine_report = output_dir / "engine_report.json"

    if corpus_path is None:
        corpus_path = ensure_k2vv_corpus()

    rows = _read_corpus(corpus_path, max_samples)
    logger.info(
        "tool_call_verifier: %d rows from %s (concurrency=%d)",
        len(rows),
        corpus_path,
        concurrency,
    )

    started_at = _now_iso()
    started_ts = time.time()

    limiter = trio.CapacityLimiter(concurrency)
    results: list[dict[str, Any]] = [None] * len(rows)  # type: ignore[list-item]

    timeout = httpx.Timeout(timeout=None, connect=60.0, read=request_timeout_s)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with trio.open_nursery() as nursery:

            async def _worker(i: int, row: dict[str, Any]) -> None:
                result = await _process_one_row(
                    client=client,
                    base_url=base_url,
                    api_key=api_key,
                    row=row,
                    model=model,
                    extra_body=extra_body,
                    limiter=limiter,
                )
                results[i] = result

            for i, row in enumerate(rows):
                nursery.start_soon(_worker, i, row)

    finished_at = _now_iso()
    duration_ms = int((time.time() - started_ts) * 1000)

    with engine_jsonl.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary = compute_summary(
        model=model,
        results=results,
        eval_started_at=started_at,
        eval_finished_at=finished_at,
        eval_duration_ms=duration_ms,
    )
    summary["base_url"] = base_url
    summary["corpus_path"] = str(corpus_path)
    summary["row_count"] = len(rows)

    with engine_report.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "status": "ok",
        "engine_report_path": str(engine_report),
        "engine_jsonl_path": str(engine_jsonl),
        "row_count": len(rows),
        "summary": summary,
    }
