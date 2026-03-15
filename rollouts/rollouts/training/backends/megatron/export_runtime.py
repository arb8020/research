from __future__ import annotations

import inspect
import logging
import re
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from .weight_conversion import convert_megatron_to_hf, remove_padding

logger = logging.getLogger(__name__)


def build_megatron_hf_tensors_from_runtime(
    *,
    model_name: str,
    model_chunks: Sequence[Any],
    vocab_size: int,
    num_layers: int,
    num_attention_heads: int,
    hidden_size: int,
    num_query_groups: int | None = None,
    kv_channels: int | None = None,
    q_lora_rank: int | None = None,
) -> tuple[OrderedDict[str, Tensor], tuple[str, ...]]:
    """Materialize inference-exportable HF tensors from live Megatron runtime state.

    This is the missing backend-local denotation layer between Megatron-native
    parameter state and the tensor product type we can actually publish to an
    inference engine. It is intentionally backend-specific and may use TP
    collectives internally.
    """
    from megatron.core import mpu

    pp_size = mpu.get_pipeline_model_parallel_world_size()
    ep_size = mpu.get_expert_model_parallel_world_size()

    # TODO(chiraag): Support PP/EP export honestly by porting the full
    # slime/miles global-param-info + PP/EP exchange path. The current witness
    # path is TP-only, so reject the broader state space instead of pretending.
    if pp_size != 1:
        raise RuntimeError(
            "Megatron inference export from runtime currently only supports "
            f"pipeline_model_parallel_size=1, got {pp_size}."
        )
    if ep_size != 1:
        raise RuntimeError(
            "Megatron inference export from runtime currently only supports "
            f"expert_model_parallel_size=1, got {ep_size}."
        )

    tensors: OrderedDict[str, Tensor] = OrderedDict()
    dropped_unconverted_keys: list[str] = []

    for megatron_name, local_param in _named_params_and_buffers_global(model_chunks):
        full_param = _all_gather_param(megatron_name, local_param)
        try:
            converted_named_tensors = convert_megatron_to_hf(
                model_name=model_name,
                name=megatron_name,
                param=full_param,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_attention_heads=num_attention_heads,
                hidden_size=hidden_size,
                num_query_groups=num_query_groups,
                kv_channels=kv_channels,
                q_lora_rank=q_lora_rank,
            )
        except Exception:
            dropped_unconverted_keys.append(megatron_name)
            continue

        for hf_name, hf_param in converted_named_tensors:
            tensors[hf_name] = remove_padding(hf_name, hf_param, vocab_size)

    if dropped_unconverted_keys:
        logger.warning(
            "Megatron runtime export left %d tensors unconverted; first keys: %s",
            len(dropped_unconverted_keys),
            dropped_unconverted_keys[:8],
        )

    return tensors, tuple(dropped_unconverted_keys)


def _all_gather_param(name: str, param: Tensor) -> Tensor:
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
    # Megatron's fused GLU/SwiGLU fc1 is interleaved across TP ranks.
    if "linear_fc1.weight" in name:
        if partition_stride not in (1, 2):
            raise RuntimeError(
                f"Expected partition_stride in (1, 2) for {name}, got {partition_stride}"
            )
        return partition_stride, partition_dim

    # Grouped MoE fc2 uses a transposed shard dimension in Megatron.
    if "linear_fc2.weight" in name:
        if partition_stride != 1:
            raise RuntimeError(f"Expected partition_stride=1 for {name}, got {partition_stride}")
        if partition_dim == 0:
            partition_dim = 1
        return partition_stride, partition_dim

    if partition_stride != 1:
        raise RuntimeError(f"Expected partition_stride=1 for {name}, got {partition_stride}")
    return partition_stride, partition_dim


def _named_params_and_buffers_global(model_chunks: Sequence[Any]) -> Iterator[tuple[str, Tensor]]:
    for vp_stage, model_chunk in enumerate(model_chunks):
        layer_offset = _transformer_layer_offset(model_chunk, vp_stage=vp_stage)

        for raw_name, param in model_chunk.named_parameters():
            yield _globalize_megatron_name(raw_name, layer_offset=layer_offset), param

        for raw_name, buffer in model_chunk.named_buffers():
            if "expert_bias" not in raw_name:
                continue
            yield _globalize_megatron_name(raw_name, layer_offset=layer_offset), buffer


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
