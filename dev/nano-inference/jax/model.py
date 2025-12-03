#!/usr/bin/env python3
"""JAX GPT-2 model implementation.

Shape suffixes (Shazeer convention):
  B: batch_size
  T: seq_len (time/sequence dimension)
  D: d_model
  V: vocab_size
"""

import sys
from pathlib import Path

import jax.numpy as jnp
from jax import Array

sys.path.append(str(Path(__file__).parent.parent.parent))
from backends.jax.layers import (
    embedding_lookup,
    layer_norm,
    linear_transform,
    mlp_block,
    multi_head_attention,
)
from config import GPT2Config


def gpt2_extract_block_weights(layer_idx: int, weights: dict[str, Array]) -> dict[str, Array]:
    """helper function to extract weights for a GPT2 block at given layer index"""
    return {
        'ln_1': {
            'weight': weights[f"h.{layer_idx}.ln_1.weight"],
            'bias': weights[f"h.{layer_idx}.ln_1.bias"]
        },
        'attn': {
            'c_attn': {
                'weight': weights[f"h.{layer_idx}.attn.c_attn.weight"],
                'bias': weights[f"h.{layer_idx}.attn.c_attn.bias"]
            },
            'c_proj': {
                'weight': weights[f"h.{layer_idx}.attn.c_proj.weight"],
                'bias': weights[f"h.{layer_idx}.attn.c_proj.bias"]
            }
        },
        'ln_2': {
            'weight': weights[f"h.{layer_idx}.ln_2.weight"],
            'bias': weights[f"h.{layer_idx}.ln_2.bias"]
        },
        'mlp': {
            'c_fc': {
                'weight': weights[f"h.{layer_idx}.mlp.c_fc.weight"],
                'bias': weights[f"h.{layer_idx}.mlp.c_fc.bias"]
            },
            'c_proj': {
                'weight': weights[f"h.{layer_idx}.mlp.c_proj.weight"],
                'bias': weights[f"h.{layer_idx}.mlp.c_proj.bias"]
            }
        }
    }


def transformer_block(x_BTD: Array, weights: dict[str, Array], layer_idx: int, config: GPT2Config) -> Array:
    """Single transformer block with attention and MLP."""
    assert layer_idx >= 0
    assert layer_idx < config.n_layers

    attn_input_BTD = layer_norm(x_BTD, weights, f"h.{layer_idx}.ln_1", config.layer_norm_epsilon)
    attn_output_BTD = multi_head_attention(attn_input_BTD, weights, layer_idx, config)
    x_BTD = x_BTD + attn_output_BTD

    mlp_input_BTD = layer_norm(x_BTD, weights, f"h.{layer_idx}.ln_2", config.layer_norm_epsilon)
    mlp_output_BTD = mlp_block(mlp_input_BTD, weights, layer_idx)
    x_BTD = x_BTD + mlp_output_BTD

    return x_BTD


def gpt2_forward(input_ids_BT: Array, weights: dict[str, Array], config: GPT2Config) -> Array:
    """Forward pass through GPT-2.

    Returns: logits_BTV of shape (batch_size, seq_len, vocab_size)
    """
    assert len(input_ids_BT.shape) == 2
    assert "wte" in weights
    assert "wpe" in weights

    B, T = input_ids_BT.shape

    # Token and position embeddings
    token_embeddings_BTD = embedding_lookup(weights, input_ids_BT, "wte")
    positions_1T = jnp.arange(T)[None, :]
    position_embeddings_1TD = embedding_lookup(weights, positions_1T, "wpe")
    x_BTD = token_embeddings_BTD + position_embeddings_1TD

    # Transformer blocks
    for layer_idx in range(config.n_layers):
        x_BTD = transformer_block(x_BTD, weights, layer_idx, config)

    # Final layer norm
    x_BTD = layer_norm(x_BTD, weights, "ln_f", config.layer_norm_epsilon)

    # Language model head
    if "lm_head" in weights:
        logits_BTV = linear_transform(x_BTD, weights, "lm_head")
    else:
        logits_BTV = jnp.matmul(x_BTD, weights["wte"].T)

    return logits_BTV
