"""Profiling-oriented MI355X serving scenario for DeepSeek-V3.2.

This reuses the same mixed-workload shape as `mixture_mi355x_smoke.py`, but
enables server-side profiling hooks and disables warm-endpoint reuse so a
profiling run cannot silently attach to a non-profiled server.

The config only enables hooks:
- request-time stats in logs
- torch profiler hook (`--profiler-config.profiler=cuda`)
- layerwise NVTX ranges

Actual trace capture still requires an explicit POST `/start_profile` and/or
an external `nsys` attach.
"""

from __future__ import annotations

from dataclasses import replace

from examples.serving.mixture_mi355x_smoke import _build_docker_run
from examples.serving.mixture_mi355x_smoke import endpoint as _base_endpoint
from examples.serving.mixture_mi355x_smoke import serving_scenario as _base_serving_scenario
from rollouts.serving.configs import ServingOutputConfig

endpoint = replace(
    _base_endpoint,
    launch_cmd=_build_docker_run(
        enable_request_time_stats_logging=True,
        enable_cuda_profiler=True,
        enable_layerwise_nvtx_tracing=True,
    ),
)

serving_scenario = replace(
    _base_serving_scenario,
    endpoint=endpoint,
    output=ServingOutputConfig(experiment_name="mixture_mi355x_smoke_profiling"),
    reuse_running_endpoint=False,
    leave_endpoint_running=False,
)
