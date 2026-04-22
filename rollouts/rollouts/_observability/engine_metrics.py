from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from urllib.parse import SplitResult, urlsplit, urlunsplit

import httpx
import trio

logger = logging.getLogger(__name__)
_HTTP_TIMEOUT = httpx.Timeout(10.0)
_SUFFIXED_KIND_MARKERS = ("_bucket", "_count", "_sum", "_created")


def _metrics_url_from_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    base_path = parsed.path.rstrip("/")
    metrics_path = f"{base_path[:-3]}/metrics" or "/metrics"
    return urlunsplit(
        SplitResult(
            scheme=parsed.scheme,
            netloc=parsed.netloc,
            path=metrics_path,
            query="",
            fragment="",
        )
    )


def _parse_labels(label_block: str) -> dict[str, str]:
    if not label_block:
        return {}

    labels: dict[str, str] = {}
    i = 0
    while i < len(label_block):
        while i < len(label_block) and label_block[i] in {" ", ","}:
            i += 1
        if i >= len(label_block):
            break

        key_end = label_block.find("=", i)
        if key_end <= i:
            raise ValueError(f"Invalid label block: {label_block!r}")
        key = label_block[i:key_end].strip()
        if not key:
            raise ValueError(f"Invalid label block: {label_block!r}")

        i = key_end + 1
        if i >= len(label_block) or label_block[i] != '"':
            raise ValueError(f"Invalid label block: {label_block!r}")
        i += 1

        value_chars: list[str] = []
        while i < len(label_block):
            char = label_block[i]
            if char == "\\":
                if i + 1 >= len(label_block):
                    raise ValueError(f"Invalid label block: {label_block!r}")
                value_chars.append(label_block[i + 1])
                i += 2
                continue
            if char == '"':
                i += 1
                break
            value_chars.append(char)
            i += 1
        else:
            raise ValueError(f"Invalid label block: {label_block!r}")

        labels[key] = "".join(value_chars)
    return labels


def _metric_kind(metric_name: str, kinds: dict[str, str]) -> str:
    if metric_name in kinds:
        return kinds[metric_name]
    for suffix in _SUFFIXED_KIND_MARKERS:
        if metric_name.endswith(suffix):
            base_name = metric_name[: -len(suffix)]
            if base_name in kinds:
                return kinds[base_name]
    return "untyped"


def _split_metric_line(raw_line: str, *, line_number: int) -> tuple[str, str]:
    in_braces = False
    in_quotes = False
    escaped = False

    for index, char in enumerate(raw_line):
        if in_quotes:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_quotes = False
            continue

        if char == "{":
            in_braces = True
            continue
        if char == "}":
            in_braces = False
            continue
        if in_braces and char == '"':
            in_quotes = True
            continue
        if char.isspace():
            metric_token = raw_line[:index]
            rest = raw_line[index:].strip()
            if not metric_token or not rest:
                break
            value_token = rest.split(None, 1)[0]
            return metric_token, value_token

    raise ValueError(f"Malformed metric line at {line_number}: {raw_line!r}")


def _parse_prometheus_metrics(
    payload: str,
    *,
    scrape_ts_unix_nano: int,
) -> list[dict[str, object]]:
    metric_kinds: dict[str, str] = {}
    rows: list[dict[str, object]] = []

    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# HELP "):
            continue
        if line.startswith("# TYPE "):
            parts = line.split(None, 3)
            if len(parts) != 4:
                raise ValueError(f"Malformed TYPE line at {line_number}: {raw_line!r}")
            metric_kinds[parts[2]] = parts[3]
            continue
        if line.startswith("#"):
            continue

        metric_token, value_token = _split_metric_line(line, line_number=line_number)
        if "{" in metric_token:
            if not metric_token.endswith("}"):
                raise ValueError(f"Malformed metric line at {line_number}: {raw_line!r}")
            metric_name, label_block = metric_token[:-1].split("{", 1)
            labels = _parse_labels(label_block)
        else:
            metric_name = metric_token
            labels = {}

        try:
            value = float(value_token)
        except ValueError as exc:
            raise ValueError(
                f"Invalid metric value at line {line_number}: {value_token!r}"
            ) from exc

        rows.append({
            "ts_unix_nano": scrape_ts_unix_nano,
            "name": metric_name,
            "labels": labels,
            "value": value,
            "kind": _metric_kind(metric_name, metric_kinds),
        })

    return rows


async def poll_engine_metrics(
    *,
    base_url: str,
    output_path: Path,
    interval_s: float = 5.0,
    cancel_scope: trio.CancelScope | None = None,
) -> None:
    parsed = urlsplit(base_url)
    normalized_path = parsed.path.rstrip("/")
    assert isinstance(base_url, str) and base_url
    assert parsed.scheme in {"http", "https"}
    assert parsed.netloc
    assert normalized_path.endswith("/v1"), f"base_url must end with /v1: {base_url!r}"
    assert isinstance(output_path, Path)
    assert interval_s > 0.0
    assert cancel_scope is None or isinstance(cancel_scope, trio.CancelScope)

    metrics_url = _metrics_url_from_base_url(base_url)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_file = await trio.open_file(output_path, "a")
    client = httpx.AsyncClient(timeout=_HTTP_TIMEOUT)
    try:
        while True:
            try:
                response = await client.get(metrics_url)
                response.raise_for_status()
                rows = _parse_prometheus_metrics(
                    response.text,
                    scrape_ts_unix_nano=time.time_ns(),
                )
            except trio.Cancelled:
                raise
            except (httpx.HTTPError, ValueError) as exc:
                logger.warning("Skipping engine metrics scrape from %s: %s", metrics_url, exc)
            else:
                payload = "".join(json.dumps(row) + "\n" for row in rows)
                if payload:
                    await output_file.write(payload)
                    await output_file.flush()
            await trio.sleep(interval_s)
    finally:
        with trio.CancelScope(shield=True):
            await output_file.aclose()
            await client.aclose()
