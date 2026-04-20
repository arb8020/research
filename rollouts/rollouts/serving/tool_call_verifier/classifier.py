"""K2-Vendor-Verifier scoring logic, vendored verbatim for fidelity.

Source:
  https://github.com/MoonshotAI/K2-Vendor-Verifier
  tool_calls_eval.py @ main (2026-04 clone)

We copy two small functions from KVV — validate_tool_call and the
compute_summary body — rather than vendor the whole 1018-line script.
Our dispatcher uses rollouts' streaming layer; KVV's dispatcher is not
reused. What we reuse is the measurement: per-tool-call jsonschema
validation and the K2VV-shape summary fields.

If KVV's upstream changes its metrics shape, re-port from upstream
deliberately rather than drifting.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from jsonschema import ValidationError, validate

logger = logging.getLogger(__name__)


def validate_tool_call(
    tool_call: dict[str, Any],
    tools: list[dict[str, Any]],
) -> bool:
    """Validate tool call arguments against JSON Schema.

    Returns True iff the declared tool exists, arguments parse as JSON,
    and arguments validate against the tool's parameters schema.

    Vendored from KVV tool_calls_eval.py ToolCallsValidator.validate_tool_call.
    """
    try:
        tool_name = tool_call["function"]["name"]

        schema = next(
            (t["function"]["parameters"] for t in tools if t["function"]["name"] == tool_name),
            None,
        )

        if not schema:
            logger.warning("No schema found for tool %r", tool_name)
            return False

        args = tool_call["function"]["arguments"]
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as e:
                logger.warning("JSON parse failed for tool %r arguments: %s", tool_name, e)
                return False

        validate(instance=args, schema=schema)
        return True

    except ValidationError as e:
        logger.warning("Schema validation failed for tool %r: %s", tool_name, e.message)
        return False
    except KeyError as e:
        logger.warning("Tool call format error, missing field: %s", e)
        return False
    except Exception as e:
        logger.warning("Unexpected error during validation: %s", e)
        return False


def compute_summary(
    *,
    model: str,
    results: list[dict[str, Any]],
    eval_started_at: str | None = None,
    eval_finished_at: str | None = None,
    eval_duration_ms: int | None = None,
) -> dict[str, Any]:
    """Aggregate per-row results into K2VV-shape summary.

    Ported from KVV tool_calls_eval.py ToolCallsValidator.compute_summary.
    We keep the field names identical so published K2VV summaries and our
    engine_report.json are directly comparable.
    """
    summary: dict[str, Any] = {
        "model": model,
        "success_count": 0,
        "failure_count": 0,
        "finish_stop": 0,
        "finish_tool_calls": 0,
        "finish_others": 0,
        "finish_others_detail": {},
        "schema_validation_error_count": 0,
        "successful_tool_call_count": 0,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "eval_started_at": eval_started_at,
        "eval_finished_at": eval_finished_at,
        "eval_duration_ms": eval_duration_ms,
    }

    for r in results:
        status = r.get("status")
        finish_reason = r.get("finish_reason")
        tool_calls_valid = r.get("tool_calls_valid")

        usage = (r.get("response") or {}).get("usage")
        if isinstance(usage, dict):
            pt = usage.get("prompt_tokens")
            ct = usage.get("completion_tokens")
            tt = usage.get("total_tokens")
            if isinstance(pt, int):
                summary["usage"]["prompt_tokens"] += pt
            if isinstance(ct, int):
                summary["usage"]["completion_tokens"] += ct
            if isinstance(tt, int):
                summary["usage"]["total_tokens"] += tt

        if status == "success":
            summary["success_count"] += 1
        else:
            summary["failure_count"] += 1

        if finish_reason == "stop":
            summary["finish_stop"] += 1
        elif finish_reason == "tool_calls":
            summary["finish_tool_calls"] += 1
            if tool_calls_valid:
                summary["successful_tool_call_count"] += 1
            else:
                summary["schema_validation_error_count"] += 1
        elif finish_reason:
            summary["finish_others"] += 1
            summary["finish_others_detail"].setdefault(finish_reason, 0)
            summary["finish_others_detail"][finish_reason] += 1

    return summary
