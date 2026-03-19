from __future__ import annotations

import inspect
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor


@dataclass(frozen=True)
class RuntimeParamInfo:
    """Backend-local Megatron runtime param metadata.

    This is the local product type we were previously smearing across raw
    tensors and ambient attribute reads. It is intentionally narrower than the
    miles/slime version for now: the current witness path is TP-only, so PP/EP
    source-rank exchange is still an explicit TODO rather than a dishonest
    implicit assumption.
    """

    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    attrs: dict[str, object]
    size_bytes: int
    src_rank: int


def named_params_and_buffers_global(model_chunks: Sequence[Any]) -> Iterator[tuple[str, Tensor]]:
    for vp_stage, model_chunk in enumerate(model_chunks):
        layer_offset = _transformer_layer_offset(model_chunk, vp_stage=vp_stage)

        for raw_name, param in model_chunk.named_parameters():
            yield _globalize_megatron_name(raw_name, layer_offset=layer_offset), param

        for raw_name, buffer in model_chunk.named_buffers():
            if "expert_bias" not in raw_name:
                continue
            yield _globalize_megatron_name(raw_name, layer_offset=layer_offset), buffer


def runtime_param_infos(model_chunks: Sequence[Any]) -> list[RuntimeParamInfo]:
    infos: list[RuntimeParamInfo] = []
    rank = dist.get_rank()
    for name, param in named_params_and_buffers_global(model_chunks):
        infos.append(
            RuntimeParamInfo(
                name=name,
                dtype=param.dtype,
                shape=tuple(param.shape),
                attrs={
                    "tensor_model_parallel": getattr(param, "tensor_model_parallel", False),
                    "partition_dim": getattr(param, "partition_dim", -1),
                    "partition_stride": getattr(param, "partition_stride", 1),
                    "parallel_mode": getattr(param, "parallel_mode", None),
                },
                size_bytes=param.numel() * param.element_size(),
                src_rank=rank,
            )
        )
    infos.sort(key=lambda info: info.name)
    return infos


def runtime_param_info_buckets(
    infos: Sequence[RuntimeParamInfo],
    *,
    bucket_size_bytes: int,
) -> list[list[RuntimeParamInfo]]:
    """Partition runtime params into buckets.

    This mirrors the miles/slime iterator shape. We still only support the TP
    witness path, but bucketization makes the export contract explicit and lets
    us validate it offline without waiting for NCCL send order to reveal bugs.
    """

    if bucket_size_bytes <= 0:
        return [list(infos)] if infos else []

    buckets: list[list[RuntimeParamInfo]] = [[]]
    current_size = 0
    for info in infos:
        full_param_size = info.size_bytes * _tp_size_for_name(info.name)
        if current_size + full_param_size > bucket_size_bytes and buckets[-1]:
            buckets.append([])
            current_size = 0
        buckets[-1].append(info)
        current_size += full_param_size
    return buckets


def all_gather_runtime_param(name: str, param: Tensor) -> Tensor:
    """Gather a TP-sharded Megatron parameter into its full tensor."""
    from megatron.core import mpu

    if "expert_bias" in name:
        return param

    if not getattr(param, "tensor_model_parallel", False):
        return param.data
    if getattr(param, "parallel_mode", None) == "duplicated":
        return param.data

    if ".experts." in name:
        tp_size = mpu.get_expert_tensor_parallel_world_size()
        tp_group = mpu.get_expert_tensor_parallel_group()
    else:
        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_group = mpu.get_tensor_model_parallel_group()

    param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
    dist.all_gather(param_partitions, param.data, group=tp_group)

    partition_dim = getattr(param, "partition_dim", -1)
    partition_stride = getattr(param, "partition_stride", 1)
    partition_stride, partition_dim = _check_and_fix_partition(
        name=name,
        partition_stride=partition_stride,
        partition_dim=partition_dim,
    )
    return _gather_with_stride(
        param_partitions=param_partitions,
        partition_dim=partition_dim,
        partition_stride=partition_stride,
    )


def _tp_size_for_name(name: str) -> int:
    from megatron.core import mpu

    if ".experts." in name:
        return int(mpu.get_expert_tensor_parallel_world_size())
    return int(mpu.get_tensor_model_parallel_world_size())


def _gather_with_stride(
    *,
    param_partitions: list[Tensor],
    partition_dim: int,
    partition_stride: int,
) -> Tensor:
    if partition_stride == 1:
        return torch.cat(param_partitions, dim=partition_dim)

    chunks_per_rank = [
        partition.chunk(partition_stride, dim=partition_dim) for partition in param_partitions
    ]
    interleaved = [
        chunks_per_rank[rank][stride]
        for stride in range(partition_stride)
        for rank in range(len(param_partitions))
    ]
    return torch.cat(interleaved, dim=partition_dim)


def _check_and_fix_partition(
    *, name: str, partition_stride: int, partition_dim: int
) -> tuple[int, int]:
    if "linear_fc1.weight" in name:
        if partition_stride not in (1, 2):
            raise RuntimeError(
                f"Expected partition_stride in (1, 2) for {name}, got {partition_stride}"
            )
        return partition_stride, partition_dim

    if "linear_fc2.weight" in name:
        if partition_stride != 1:
            raise RuntimeError(f"Expected partition_stride=1 for {name}, got {partition_stride}")
        if partition_dim == 0:
            partition_dim = 1
        return partition_stride, partition_dim

    if partition_stride != 1:
        raise RuntimeError(f"Expected partition_stride=1 for {name}, got {partition_stride}")
    return partition_stride, partition_dim


def _transformer_layer_offset(model_chunk: Any, *, vp_stage: int) -> int:
    try:
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
    except Exception:
        return 0

    config = getattr(model_chunk, "config", None)
    if config is None:
        module = getattr(model_chunk, "module", None)
        config = getattr(module, "config", None)
    if config is None:
        return 0

    try:
        sig = inspect.signature(get_transformer_layer_offset)
        if "vp_stage" in sig.parameters:
            return int(get_transformer_layer_offset(config, vp_stage))
        return int(get_transformer_layer_offset(config))
    except Exception:
        return 0


def _globalize_megatron_name(raw_name: str, *, layer_offset: int) -> str:
    name = raw_name
    while name.startswith("module."):
        name = name[len("module.") :]

    decoder_match = re.match(r"decoder\.layers\.(\d+)\.(.+)", name)
    if decoder_match:
        layer_idx, rest = decoder_match.groups()
        return f"decoder.layers.{int(layer_idx) + layer_offset}.{rest}"

    mtp_match = re.match(r"mtp\.layers\.(\d+)\.(.+)", name)
    if mtp_match:
        layer_idx, rest = mtp_match.groups()
        return f"mtp.layers.{int(layer_idx) + layer_offset}.{rest}"

    return name
