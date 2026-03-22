from __future__ import annotations

import logging
import re
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from .export_common import named_params_and_buffers_global
from .export_iterator import build_runtime_hf_tensors
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
    """Materialize inference-exportable HF tensors from live Megatron state.

    This is the backend-local runtime export boundary. It now goes through an
    explicit global-name + param-info + iterator path instead of folding those
    concerns into one ad hoc function.
    """
    from megatron.core import mpu

    pp_size = mpu.get_pipeline_model_parallel_world_size()
    ep_size = mpu.get_expert_model_parallel_world_size()
    tp_size = mpu.get_tensor_model_parallel_world_size()

    if pp_size != 1:
        raise RuntimeError(
            "Megatron inference export from runtime currently only supports "
            f"pipeline_model_parallel_size=1, got {pp_size}."
        )

    # Narrow, explicit witness path for GLM/Qwen MoE on Modal:
    # PP=1, TP=1, EP>1. This is the smallest honest EP source-rank exchange
    # slice we need before porting the broader miles/slime export iterator.
    if ep_size != 1:
        if tp_size != 1:
            raise RuntimeError(
                "Megatron inference export from runtime currently only supports "
                "expert-model-parallel MoE export when tensor_model_parallel_size=1, "
                f"got tp={tp_size}, ep={ep_size}."
            )
        return _build_megatron_hf_tensors_from_runtime_ep_only(
            model_name=model_name,
            model_chunks=model_chunks,
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            num_query_groups=num_query_groups,
            kv_channels=kv_channels,
            q_lora_rank=q_lora_rank,
            ep_size=ep_size,
        )

    tensors, dropped_unconverted_keys = build_runtime_hf_tensors(
        model_chunks=model_chunks,
        model_name=model_name,
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        num_query_groups=num_query_groups,
        kv_channels=kv_channels,
        q_lora_rank=q_lora_rank,
    )

    if dropped_unconverted_keys:
        logger.warning(
            "Megatron runtime export left %d tensors unconverted; first keys: %s",
            len(dropped_unconverted_keys),
            list(dropped_unconverted_keys[:8]),
        )

    return tensors, dropped_unconverted_keys


def _build_megatron_hf_tensors_from_runtime_ep_only(
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
    ep_size: int,
) -> tuple[OrderedDict[str, Tensor], tuple[str, ...]]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == ep_size, (
        "EP runtime export currently assumes the trainer world is exactly one "
        f"expert-parallel group, got world_size={world_size}, ep_size={ep_size}."
    )

    local_named_params = dict(named_params_and_buffers_global(model_chunks))
    tensors: OrderedDict[str, Tensor] = OrderedDict()
    dropped_unconverted: list[str] = []

    local_expert_indices = sorted({
        int(match.group(2))
        for name in local_named_params
        if (match := _LOCAL_EXPERT_PARAM_PATTERN.match(name)) is not None
    })
    local_experts_per_rank = local_expert_indices[-1] + 1 if local_expert_indices else 0

    if local_expert_indices and local_expert_indices != list(range(local_experts_per_rank)):
        raise RuntimeError(
            "Megatron EP runtime export expected dense local expert indices, got "
            f"{local_expert_indices}."
        )

    local_expert_params: list[tuple[str, Tensor]] = []
    for name, param in local_named_params.items():
        local_expert_match = _LOCAL_EXPERT_PARAM_PATTERN.match(name)
        if local_expert_match is not None:
            local_expert_params.append((name, param))
            continue

        if rank != 0:
            continue

        full_param = _materialize_ep_local_param(param)
        _convert_runtime_param(
            tensors=tensors,
            dropped_unconverted=dropped_unconverted,
            model_name=model_name,
            name=name,
            param=full_param,
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            num_query_groups=num_query_groups,
            kv_channels=kv_channels,
            q_lora_rank=q_lora_rank,
        )

    _collect_ep_local_experts_to_rank0(
        tensors=tensors,
        dropped_unconverted=dropped_unconverted,
        model_name=model_name,
        local_expert_params=local_expert_params,
        rank=rank,
        world_size=world_size,
        local_experts_per_rank=local_experts_per_rank,
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        num_query_groups=num_query_groups,
        kv_channels=kv_channels,
        q_lora_rank=q_lora_rank,
    )

    if rank != 0:
        return OrderedDict(), ()

    return tensors, tuple(dropped_unconverted)


def _convert_runtime_param(
    *,
    tensors: OrderedDict[str, Tensor],
    dropped_unconverted: list[str],
    model_name: str,
    name: str,
    param: Tensor,
    vocab_size: int,
    num_layers: int,
    num_attention_heads: int,
    hidden_size: int,
    num_query_groups: int | None,
    kv_channels: int | None,
    q_lora_rank: int | None,
) -> None:
    try:
        converted_named_tensors = convert_megatron_to_hf(
            model_name=model_name,
            name=name,
            param=param,
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            num_query_groups=num_query_groups,
            kv_channels=kv_channels,
            q_lora_rank=q_lora_rank,
        )
    except Exception:
        dropped_unconverted.append(name)
        return

    for hf_name, hf_param in converted_named_tensors:
        tensors[hf_name] = remove_padding(hf_name, hf_param, vocab_size)


_LOCAL_EXPERT_PARAM_PATTERN = re.compile(
    r"^(decoder\.layers\.\d+\.mlp\.experts\.local_experts\.(\d+)\..+)$"
)


def _globalize_local_expert_name(name: str, *, ep_rank: int, local_experts_per_rank: int) -> str:
    match = _LOCAL_EXPERT_PARAM_PATTERN.match(name)
    assert match is not None, f"expected local expert param name, got {name!r}"
    local_idx = int(match.group(2))
    global_idx = ep_rank * local_experts_per_rank + local_idx
    return name.replace(
        f".mlp.experts.local_experts.{local_idx}.",
        f".mlp.experts.{global_idx}.",
        1,
    )


def _materialize_ep_local_param(param: Tensor) -> Tensor:
    # EP-only runtime export for the current witness path has TP=1, so every
    # local param is already complete. Materialize it on CPU directly instead of
    # paying an unnecessary GPU-side gather/all_gather buffer tax during
    # validation and sync.
    return param.detach().cpu()


def _collect_ep_local_experts_to_rank0(
    *,
    tensors: OrderedDict[str, Tensor],
    dropped_unconverted: list[str],
    model_name: str,
    local_expert_params: Sequence[tuple[str, Tensor]],
    rank: int,
    world_size: int,
    local_experts_per_rank: int,
    vocab_size: int,
    num_layers: int,
    num_attention_heads: int,
    hidden_size: int,
    num_query_groups: int | None,
    kv_channels: int | None,
    q_lora_rank: int | None,
) -> None:
    # TODO: Generalize this once EP runtime export stops assuming identical
    # dense local-expert layouts on every rank and we have a real shared tensor
    # transport contract for runtime export beyond this witness path.
    for src_rank in range(world_size):
        if src_rank == 0:
            if rank == 0:
                for local_name, local_param in local_expert_params:
                    _convert_runtime_param(
                        tensors=tensors,
                        dropped_unconverted=dropped_unconverted,
                        model_name=model_name,
                        name=_globalize_local_expert_name(
                            local_name,
                            ep_rank=0,
                            local_experts_per_rank=local_experts_per_rank,
                        ),
                        param=_materialize_ep_local_param(local_param),
                        vocab_size=vocab_size,
                        num_layers=num_layers,
                        num_attention_heads=num_attention_heads,
                        hidden_size=hidden_size,
                        num_query_groups=num_query_groups,
                        kv_channels=kv_channels,
                        q_lora_rank=q_lora_rank,
                    )
            dist.barrier()
            continue

        if rank == src_rank:
            for _local_name, local_param in local_expert_params:
                send_tensor = local_param.detach()
                if not send_tensor.is_contiguous():
                    send_tensor = send_tensor.contiguous()
                dist.send(send_tensor, dst=0)
        elif rank == 0:
            for local_name, local_param in local_expert_params:
                recv_tensor = torch.empty_like(local_param.detach())
                dist.recv(recv_tensor, src=src_rank)
                _convert_runtime_param(
                    tensors=tensors,
                    dropped_unconverted=dropped_unconverted,
                    model_name=model_name,
                    name=_globalize_local_expert_name(
                        local_name,
                        ep_rank=src_rank,
                        local_experts_per_rank=local_experts_per_rank,
                    ),
                    param=recv_tensor.cpu(),
                    vocab_size=vocab_size,
                    num_layers=num_layers,
                    num_attention_heads=num_attention_heads,
                    hidden_size=hidden_size,
                    num_query_groups=num_query_groups,
                    kv_channels=kv_channels,
                    q_lora_rank=q_lora_rank,
                )
                del recv_tensor
        dist.barrier()
