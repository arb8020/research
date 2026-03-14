"""Qwen-family Megatron model helpers.

These helpers implement the plain Qwen3 Megatron path in the same semantic
shape used by slime/miles: explicit Megatron model args derived from normalized
model denotation, rather than pretending a generic HF-native raw bridge path is
valid for every Qwen family model.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from typing import Any

from rollouts.training.models import ModelDenotation


def _parse_megatron_args(argv: list[str]) -> Namespace:
    from megatron.training.arguments import parse_args

    original_argv = sys.argv[:]
    sys.argv = [original_argv[0] if original_argv else ""] + argv
    try:
        try:
            return parse_args()
        except TypeError:
            return parse_args(lambda parser: parser)  # type: ignore[call-arg]
    finally:
        sys.argv = original_argv


def _build_qwen3_cli_args(
    denotation: ModelDenotation,
    *,
    seq_length: int,
    micro_batch_size: int,
    global_batch_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel_size: int,
    sequence_parallel: bool,
    bf16: bool,
    fp16: bool,
) -> list[str]:
    arch = denotation.architecture
    head_dim = arch.metadata.get("head_dim")
    if not isinstance(head_dim, int) or head_dim <= 0:
        head_dim = arch.hidden_size // arch.num_attention_heads

    args = [
        "--swiglu",
        "--num-layers",
        str(arch.num_layers),
        "--hidden-size",
        str(arch.hidden_size),
        "--ffn-hidden-size",
        str(arch.ffn_hidden_size),
        "--num-attention-heads",
        str(arch.num_attention_heads),
        "--vocab-size",
        str(arch.vocab_size),
        "--seq-length",
        str(seq_length),
        "--max-position-embeddings",
        str(seq_length),
        "--micro-batch-size",
        str(micro_batch_size),
        "--global-batch-size",
        str(global_batch_size),
        "--tensor-model-parallel-size",
        str(tensor_parallel_size),
        "--pipeline-model-parallel-size",
        str(pipeline_parallel_size),
        "--expert-model-parallel-size",
        str(expert_parallel_size),
        "--kv-channels",
        str(head_dim),
    ]

    if arch.num_kv_heads not in (None, arch.num_attention_heads):
        args.extend([
            "--group-query-attention",
            "--num-query-groups",
            str(arch.num_kv_heads),
        ])

    if arch.rotary is not None:
        args.extend([
            "--use-rotary-position-embeddings",
            "--rotary-base",
            str(int(arch.rotary.theta) if arch.rotary.theta.is_integer() else arch.rotary.theta),
        ])

    if not arch.uses_bias_linear:
        args.append("--disable-bias-linear")

    args.extend([
        "--normalization",
        "RMSNorm" if arch.norm == "rmsnorm" else "LayerNorm",
        "--norm-epsilon",
        str(arch.norm_epsilon),
    ])

    if arch.uses_qk_layernorm:
        args.append("--qk-layernorm")

    if not arch.tie_embeddings:
        args.append("--untie-embeddings-and-output-weights")

    if sequence_parallel:
        args.append("--sequence-parallel")
    if bf16:
        args.append("--bf16")
    if fp16:
        args.append("--fp16")

    return args


def build_qwen3_transformer_config(
    denotation: ModelDenotation,
    *,
    seq_length: int,
    micro_batch_size: int,
    global_batch_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel_size: int,
    sequence_parallel: bool,
    bf16: bool,
    fp16: bool,
) -> Any:
    """Construct a Megatron TransformerConfig from explicit Qwen3 semantics."""
    from megatron.training.arguments import core_transformer_config_from_args

    args = _parse_megatron_args(
        _build_qwen3_cli_args(
            denotation,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            expert_parallel_size=expert_parallel_size,
            sequence_parallel=sequence_parallel,
            bf16=bf16,
            fp16=fp16,
        )
    )
    args.vocab_size = denotation.architecture.vocab_size
    args.padded_vocab_size = denotation.architecture.vocab_size
    args.max_position_embeddings = seq_length
    args.seq_length = seq_length
    args.untie_embeddings_and_output_weights = not denotation.architecture.tie_embeddings
    return core_transformer_config_from_args(args)
