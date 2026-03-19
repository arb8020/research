from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

from torch import Tensor

from .export_iterator import build_runtime_hf_tensors

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

    # TODO(chiraag): Port the full miles/slime PP/EP source-rank exchange path.
    # The current witness path is TP-only, so reject the broader state space
    # rather than lying about export completeness.
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
