from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from torch import Tensor
from transformers import AutoConfig

from .weight_conversion import convert_megatron_to_hf, remove_padding

logger = logging.getLogger(__name__)

_EXPLICIT_HF_EXPORT_MODEL_KEYS = (
    "qwen2",
    "qwen3",
    "glm4",
    "glm4moe",
    "glm4moelite",
    "deepseekv3",
    "llama",
    "mimo",
)


@dataclass(frozen=True)
class MegatronInferenceExport:
    """Inference-ready Megatron weight export.

    This is the explicit boundary between Megatron-local model state and the
    tensor product type we can actually publish to inference.
    """

    tensors: OrderedDict[str, Tensor]
    dropped_none_keys: tuple[str, ...] = ()
    dropped_unconverted_keys: tuple[str, ...] = ()


def build_megatron_inference_export(
    model_name: str,
    raw_state_dict: dict[str, Any],
) -> MegatronInferenceExport:
    """Normalize Megatron-local state into inference-syncable tensors.

    For known model families, this lowers Megatron parameter names into the HF
    naming/shape contract our inference engines expect. Unsupported or missing
    conversion is rejected instead of silently sending raw Megatron names.
    """
    explicit_hf_export = _requires_explicit_hf_export(model_name)
    conversion_context = _load_conversion_context(model_name) if model_name else None

    tensors: OrderedDict[str, Tensor] = OrderedDict()
    dropped_none_keys: list[str] = []
    dropped_unconverted_keys: list[str] = []

    for name, value in raw_state_dict.items():
        clean_name = _strip_chunk_prefix(name)

        if value is None:
            dropped_none_keys.append(clean_name)
            continue

        if not isinstance(value, Tensor):
            raise TypeError(
                "Megatron inference export only supports tensor-valued weights. "
                f"Got {type(value).__name__} for key {clean_name!r}."
            )

        if conversion_context is None:
            tensors[clean_name] = value
            continue

        try:
            converted_named_tensors = convert_megatron_to_hf(
                model_name=model_name,
                name=clean_name,
                param=value,
                vocab_size=conversion_context["vocab_size"],
                num_layers=conversion_context["num_layers"],
                num_attention_heads=conversion_context["num_attention_heads"],
                hidden_size=conversion_context["hidden_size"],
                num_query_groups=conversion_context["num_query_groups"],
                kv_channels=conversion_context["kv_channels"],
                q_lora_rank=conversion_context["q_lora_rank"],
            )
        except Exception:
            dropped_unconverted_keys.append(clean_name)
            continue

        for hf_name, hf_param in converted_named_tensors:
            normalized_name = _strip_chunk_prefix(hf_name)
            tensors[normalized_name] = remove_padding(
                normalized_name,
                hf_param,
                conversion_context["vocab_size"],
            )

    if explicit_hf_export and not tensors:
        raise RuntimeError(
            "Megatron inference export could not convert any tensors for "
            f"{model_name!r}. The Megatron->HF export path for this model family "
            "is not implemented honestly yet."
        )

    if explicit_hf_export and dropped_unconverted_keys:
        sample = ", ".join(dropped_unconverted_keys[:8])
        raise RuntimeError(
            "Megatron inference export left unconverted tensors for "
            f"{model_name!r}: {sample}. The inference sync contract is incomplete."
        )

    if not tensors:
        raise RuntimeError("Megatron inference export produced zero tensor weights.")

    if dropped_none_keys:
        logger.warning(
            "Megatron inference export dropped %d None-valued weights; first keys: %s",
            len(dropped_none_keys),
            dropped_none_keys[:8],
        )

    return MegatronInferenceExport(
        tensors=tensors,
        dropped_none_keys=tuple(dropped_none_keys),
        dropped_unconverted_keys=tuple(dropped_unconverted_keys),
    )


def _requires_explicit_hf_export(model_name: str) -> bool:
    model_name_lower = model_name.lower()
    return any(key in model_name_lower for key in _EXPLICIT_HF_EXPORT_MODEL_KEYS)


def _load_conversion_context(model_name: str) -> dict[str, int | None] | None:
    try:
        hf_config = AutoConfig.from_pretrained(model_name)
    except Exception as exc:
        logger.warning(
            "Failed to load HF config for Megatron inference export of %s: %s",
            model_name,
            exc,
        )
        return None

    num_layers = getattr(hf_config, "num_hidden_layers", 0) or getattr(hf_config, "n_layers", 0)
    if not num_layers:
        raise RuntimeError(
            f"Unable to infer num_layers from HF config for Megatron inference export: {model_name!r}"
        )

    vocab_size = int(getattr(hf_config, "vocab_size", 0))
    num_attention_heads = int(getattr(hf_config, "num_attention_heads", 0))
    hidden_size = int(getattr(hf_config, "hidden_size", 0))
    num_query_groups = getattr(hf_config, "num_query_groups", num_attention_heads)
    kv_channels = getattr(hf_config, "kv_channels", None)
    q_lora_rank = getattr(hf_config, "q_lora_rank", None)

    return {
        "num_layers": int(num_layers),
        "vocab_size": vocab_size,
        "num_attention_heads": num_attention_heads,
        "hidden_size": hidden_size,
        "num_query_groups": num_query_groups,
        "kv_channels": kv_channels,
        "q_lora_rank": q_lora_rank,
    }


def _strip_chunk_prefix(name: str) -> str:
    prefix, separator, remainder = name.partition(".")
    if prefix.startswith("chunk_") and separator and prefix[6:].isdigit():
        return remainder
    return name
