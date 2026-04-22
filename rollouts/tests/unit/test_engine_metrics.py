from __future__ import annotations

import pytest

from rollouts._observability.engine_metrics import (
    _metrics_url_from_base_url,
    _parse_prometheus_metrics,
)


def test_metrics_url_strips_v1_suffix() -> None:
    assert _metrics_url_from_base_url("http://localhost:30000/v1") == (
        "http://localhost:30000/metrics"
    )
    assert _metrics_url_from_base_url("http://localhost:30000/prefix/v1/") == (
        "http://localhost:30000/prefix/metrics"
    )


def test_parse_prometheus_metrics_preserves_kind_and_labels() -> None:
    payload = """
    # HELP sglang:num_running_reqs The number of running requests.
    # TYPE sglang:num_running_reqs gauge
    sglang:num_running_reqs{model_name="foo bar",dp_rank="0"} 8
    # TYPE sglang:queue_time_seconds histogram
    sglang:queue_time_seconds_bucket{le="0.1",model_name="foo\\\"bar"} 1
    sglang:queue_time_seconds_sum{model_name="foo\\\"bar"} 0.05
    sglang:queue_time_seconds_count{model_name="foo\\\"bar"} 1
    """

    rows = _parse_prometheus_metrics(payload, scrape_ts_unix_nano=123)

    assert rows == [
        {
            "ts_unix_nano": 123,
            "name": "sglang:num_running_reqs",
            "labels": {"model_name": "foo bar", "dp_rank": "0"},
            "value": 8.0,
            "kind": "gauge",
        },
        {
            "ts_unix_nano": 123,
            "name": "sglang:queue_time_seconds_bucket",
            "labels": {"le": "0.1", "model_name": 'foo"bar'},
            "value": 1.0,
            "kind": "histogram",
        },
        {
            "ts_unix_nano": 123,
            "name": "sglang:queue_time_seconds_sum",
            "labels": {"model_name": 'foo"bar'},
            "value": 0.05,
            "kind": "histogram",
        },
        {
            "ts_unix_nano": 123,
            "name": "sglang:queue_time_seconds_count",
            "labels": {"model_name": 'foo"bar'},
            "value": 1.0,
            "kind": "histogram",
        },
    ]


def test_parse_prometheus_metrics_rejects_malformed_lines() -> None:
    with pytest.raises(ValueError, match="Malformed metric line"):
        _parse_prometheus_metrics("sglang:num_running_reqs", scrape_ts_unix_nano=123)
