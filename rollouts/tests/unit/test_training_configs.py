import pytest

from rollouts.training.backends.megatron.remote_backend import MegatronRemoteConfig
from rollouts.training.configs import DistributedConfig, HardwareConfig, MegatronOverrides
from rollouts.training.grpo import GRPOConfig
from rollouts.training.lowering import MegatronLowering, MegatronProvisioning, RealizationPlan
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


def test_grpo_config_from_dict_materializes_typed_megatron_overrides() -> None:
    config = GRPOConfig.from_dict({
        "trainer": {
            "backend": "megatron",
            "megatron_overrides": {
                "allocator_expandable_segments": True,
                "sequence_parallel": True,
            },
        },
        "checkpoint": {
            "weight_sync_mode": "nccl",
        },
    })

    assert config.trainer.megatron_overrides == MegatronOverrides(
        allocator_expandable_segments=True,
        sequence_parallel=True,
    )


def test_megatron_remote_config_rejects_unlowered_output_materialization() -> None:
    with pytest.raises(ValueError, match="output_materialization"):
        MegatronRemoteConfig(
            model_name="Qwen/Qwen3-0.6B",
            lowering=MegatronLowering(
                provisioning=MegatronProvisioning(tp=1, pp=1, ep=1),
                realization=RealizationPlan(),
            ),
            megatron_overrides=MegatronOverrides(output_materialization="chunked_logits"),
        )
