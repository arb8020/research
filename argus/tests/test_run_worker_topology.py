from types import SimpleNamespace

import pytest

from argus import run as argus_run
from rollouts.training.configs import (
    HardwareConfig,
    InferenceRoleBinding,
    InferenceWorkerConfig,
    TrainingWorkerConfig,
    WorkerTopologyConfig,
)
from rollouts.training.grpo import GRPOConfig


def test_normalize_worker_topology_prefers_module_export() -> None:
    topology = WorkerTopologyConfig(
        hardware=HardwareConfig(provider="local"),
        inference_workers=(InferenceWorkerConfig(worker_id="actor", model="Qwen/Qwen3-0.6B"),),
        training_workers=(TrainingWorkerConfig(worker_id="trainer"),),
        role_bindings=(InferenceRoleBinding(role="actor", worker_id="actor"),),
    )
    module = SimpleNamespace(worker_topology=topology, config=GRPOConfig())

    assert argus_run._normalize_worker_topology(module) is topology


def test_normalize_worker_topology_uses_grpo_topology() -> None:
    topology = WorkerTopologyConfig(
        hardware=HardwareConfig(provider="local"),
        service_runtime_layout="split_env",
        inference_workers=(InferenceWorkerConfig(worker_id="actor", model="Qwen/Qwen3-0.6B"),),
        training_workers=(TrainingWorkerConfig(worker_id="trainer"),),
        role_bindings=(InferenceRoleBinding(role="actor", worker_id="actor"),),
    )
    module = SimpleNamespace(config=GRPOConfig(topology=topology))

    assert argus_run._normalize_worker_topology(module) == topology


def test_normalize_worker_topology_rejects_invalid_export() -> None:
    module = SimpleNamespace(worker_topology="not-a-topology")

    with pytest.raises(ValueError, match="WorkerTopologyConfig"):
        argus_run._normalize_worker_topology(module)
