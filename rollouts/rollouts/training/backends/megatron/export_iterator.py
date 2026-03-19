from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Sequence
from typing import Any

from torch import Tensor

from .export_common import (
    all_gather_runtime_param,
    named_params_and_buffers_global,
    runtime_param_info_buckets,
    runtime_param_infos,
)
from .weight_conversion import convert_megatron_to_hf, remove_padding


class RuntimeHFWeightIteratorDirect:
    """Direct Megatron runtime -> HF tensor iterator.

    This mirrors the semantic role of miles/slime's
    `hf_weight_iterator_direct.py`, but keeps the first implementation narrow:
    global naming, TP reconstruction, bucketization, then Megatron->HF
    conversion. PP/EP source-rank exchange is still an explicit TODO.
    """

    def __init__(
        self,
        *,
        model_chunks: Sequence[Any],
        model_name: str,
        vocab_size: int,
        num_layers: int,
        num_attention_heads: int,
        hidden_size: int,
        num_query_groups: int | None = None,
        kv_channels: int | None = None,
        q_lora_rank: int | None = None,
        bucket_size_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self.model_chunks = model_chunks
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.num_query_groups = num_query_groups
        self.kv_channels = kv_channels
        self.q_lora_rank = q_lora_rank
        self.param_infos = runtime_param_infos(model_chunks)
        self.param_info_buckets = runtime_param_info_buckets(
            self.param_infos,
            bucket_size_bytes=bucket_size_bytes,
        )
        self._local_named_params = dict(named_params_and_buffers_global(model_chunks))

    def get_hf_weight_chunks(self) -> Iterator[tuple[list[tuple[str, Tensor]], tuple[str, ...]]]:
        dropped_unconverted: list[str] = []
        for bucket in self.param_info_buckets:
            hf_named_tensors: list[tuple[str, Tensor]] = []
            for info in bucket:
                local_param = self._local_named_params[info.name]
                full_param = all_gather_runtime_param(info.name, local_param)
                try:
                    converted_named_tensors = convert_megatron_to_hf(
                        model_name=self.model_name,
                        name=info.name,
                        param=full_param,
                        vocab_size=self.vocab_size,
                        num_layers=self.num_layers,
                        num_attention_heads=self.num_attention_heads,
                        hidden_size=self.hidden_size,
                        num_query_groups=self.num_query_groups,
                        kv_channels=self.kv_channels,
                        q_lora_rank=self.q_lora_rank,
                    )
                except Exception:
                    dropped_unconverted.append(info.name)
                    continue

                for hf_name, hf_param in converted_named_tensors:
                    hf_named_tensors.append((
                        hf_name,
                        remove_padding(hf_name, hf_param, self.vocab_size),
                    ))
            yield hf_named_tensors, tuple(dropped_unconverted)


def build_runtime_hf_tensors(
    *,
    model_chunks: Sequence[Any],
    model_name: str,
    vocab_size: int,
    num_layers: int,
    num_attention_heads: int,
    hidden_size: int,
    num_query_groups: int | None = None,
    kv_channels: int | None = None,
    q_lora_rank: int | None = None,
    bucket_size_bytes: int = 256 * 1024 * 1024,
) -> tuple[OrderedDict[str, Tensor], tuple[str, ...]]:
    iterator = RuntimeHFWeightIteratorDirect(
        model_chunks=model_chunks,
        model_name=model_name,
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        num_query_groups=num_query_groups,
        kv_channels=kv_channels,
        q_lora_rank=q_lora_rank,
        bucket_size_bytes=bucket_size_bytes,
    )

    tensors: OrderedDict[str, Tensor] = OrderedDict()
    dropped_unconverted: list[str] = []
    for hf_named_tensors, dropped in iterator.get_hf_weight_chunks():
        for name, tensor in hf_named_tensors:
            tensors[name] = tensor
        dropped_unconverted = list(dropped)
    return tensors, tuple(dropped_unconverted)
