import pytest

from rollouts.training.configs import DistributedConfig, HardwareConfig
from rollouts.training.multi_node import MultiNodeConfig, compute_cluster_allocation


def test_ssh_provider_requires_explicit_deps() -> None:
    with pytest.raises(ValueError, match="requires explicit deps"):
        HardwareConfig(provider="runpod", deps=None)


def test_distributed_config_all_gpus_includes_workspace_gpus() -> None:
    config = DistributedConfig(
        inference_gpus=(0,),
        trainer_gpus=(1,),
        workspace_gpus=(2, 3),
    )

    assert config.all_gpus == (0, 1, 2, 3)


def test_multi_node_allocation_reserves_workspace_gpus() -> None:
    config = MultiNodeConfig(
        num_nodes=1,
        gpus_per_node=4,
        inference_gpus_per_node=1,
        workspace_gpus_per_node=2,
    )

    allocation = compute_cluster_allocation(config, ["1.2.3.4"])
    node = allocation.nodes[0]

    assert node.inference_gpus == (0,)
    assert node.trainer_gpus == (1,)
    assert node.workspace_gpus == (2, 3)
