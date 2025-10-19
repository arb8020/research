#!/usr/bin/env python3
"""JAX layer implementations for GPT-2.

Shape suffixes (Shazeer convention):
  B: batch_size
  L: num_layers
  T: seq_len (time/sequence dimension for query)
  S: seq_len (sequence dimension for key/value, same as T for self-attention)
  D: d_model
  F: d_ff (feedforward hidden dimension)
  H: head_dim (dimension per attention head)
  N: n_heads (number of query heads)
  K: n_kv_heads (number of key-value heads)
  V: vocab_size
"""

import jax
import jax.numpy as jnp
import einops
from typing import Dict, Callable
from jax import Array

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import GPT2Config


def _validate_ffn_shapes(x_BTD: jnp.ndarray,
                        weight_in_DF: jnp.ndarray,
                        bias_in_F: jnp.ndarray,
                        weight_out_FD: jnp.ndarray,
                        bias_out_D: jnp.ndarray):
    """Validate feedforward network shapes."""
    D = x_BTD.shape[-1]
    F = weight_in_DF.shape[1]

    assert weight_in_DF.shape[0] == D
    assert bias_in_F.shape[0] == F
    assert weight_out_FD.shape[0] == F
    assert weight_out_FD.shape[1] == D
    assert bias_out_D.shape[0] == D


def _validate_linear_shapes(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> None:
    """Validate linear layer shapes."""
    assert x.shape[-1] == weight.shape[0]
    assert weight.shape[1] == bias.shape[0]


def _validate_layer_norm_shapes(x_BTD: jnp.ndarray, gamma_D: jnp.ndarray, beta_D: jnp.ndarray):
    """Validate layer normalization shapes."""
    assert gamma_D.shape[-1] == x_BTD.shape[-1]
    assert beta_D.shape[-1] == x_BTD.shape[-1]


def _validate_attention_shapes(x_BMD: jnp.ndarray, w_qkv_3DD: jnp.ndarray, b_qkv_3D: jnp.ndarray,
                             w_out_DD: jnp.ndarray, b_out_D: jnp.ndarray, config: GPT2Config):
    """Validate all attention layer input shapes."""
    B, M, D = x_BMD.shape

    assert len(x_BMD.shape) == 3
    assert D == config.d_model
    assert w_qkv_3DD.shape == (3, D, D)
    assert b_qkv_3D.shape == (3, D)
    assert w_out_DD.shape == (D, D)
    assert b_out_D.shape == (D,)
    assert D % config.n_heads == 0

    head_dim = D // config.n_heads
    assert head_dim > 0


def gelu(x):
    """Hendrycks & Gimpel (2016) https://arxiv.org/abs/1606.08415"""
    cdf = 0.5 * (1.0 + jnp.tanh(
        jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3))
    ))
    return x * cdf


def gelu_new(x_BTD: Array) -> Array:
    """GELU activation - "gelu_new" variant used by GPT-2 (exact HuggingFace implementation)."""
    return 0.5 * x_BTD * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x_BTD + 0.044715 * jnp.power(x_BTD, 3.0))))


def project_and_embed(input_ids: jnp.ndarray, weights: Dict[str, Array], config: GPT2Config) -> jnp.ndarray:
    """Radford et al. (2019) https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"""

    projected_BLD = weights['wte.weight'][input_ids]
    _, seq_len = input_ids.shape
    position_embeddings = weights['wpe.weight'][:seq_len]
    projected_embedded_BLD = projected_BLD + position_embeddings

    return projected_embedded_BLD


def embedding_lookup(weights: Dict[str, Array], tokens: Array, weight_name: str) -> Array:
    """Look up embeddings for tokens."""
    assert weight_name in weights
    assert len(tokens.shape) >= 1
    return weights[weight_name][tokens]


def layer_norm(x_BTD: jnp.ndarray, weights: Dict[str, Array], weight_prefix: str, epsilon: float = 1e-5) -> jnp.ndarray:
    """Ba et al. (2016) https://arxiv.org/abs/1607.06450"""
    # Extract gamma and beta from weights
    gamma_D = weights[f"{weight_prefix}.weight"] if f"{weight_prefix}.weight" in weights else weights[weight_prefix]
    beta_D = weights[f"{weight_prefix}.bias"]

    _validate_layer_norm_shapes(x_BTD, gamma_D, beta_D)

    mean_BT1 = jnp.mean(x_BTD, axis=-1, keepdims=True)
    variance_BT1 = jnp.var(x_BTD, axis=-1, keepdims=True)

    normalized_BTD = (x_BTD - mean_BT1) / jnp.sqrt(variance_BT1 + epsilon)
    output_BTD = normalized_BTD * gamma_D + beta_D

    return output_BTD


def linear(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Goodfellow et al. (2016) http://www.deeplearningbook.org"""
    _validate_linear_shapes(x, weight, bias)
    return x @ weight + bias


def linear_transform(x: Array, weights: Dict[str, Array], weight_prefix: str) -> Array:
    """Apply linear transformation: x @ W + b"""
    assert len(x.shape) >= 2

    weight = weights[f"{weight_prefix}.weight"] if f"{weight_prefix}.weight" in weights else weights[weight_prefix]
    assert x.shape[-1] == weight.shape[0]

    output = jnp.matmul(x, weight)

    # Add bias if it exists
    bias_key = f"{weight_prefix}.bias"
    if bias_key in weights:
        output = output + weights[bias_key]

    return output


def ffn(x_BTD: jnp.ndarray,
        weight_in_DF: jnp.ndarray,
        bias_in_F: jnp.ndarray,
        weight_out_FD: jnp.ndarray,
        bias_out_D: jnp.ndarray,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """Vaswani et al. (2017) https://arxiv.org/abs/1706.03762"""
    _validate_ffn_shapes(x_BTD, weight_in_DF, bias_in_F, weight_out_FD, bias_out_D)
    hidden_BTF = linear(x_BTD, weight_in_DF, bias_in_F)
    activated_BTF = activation_fn(hidden_BTF)
    output_BTD = linear(activated_BTF, weight_out_FD, bias_out_D)
    return output_BTD


def mlp_block(x_BTD: Array, weights: Dict[str, Array], layer_idx: int) -> Array:
    """MLP block with GELU activation."""
    hidden_BTF = linear_transform(x_BTD, weights, f"h.{layer_idx}.mlp.c_fc")
    hidden_BTF = gelu_new(hidden_BTF)
    output_BTD = linear_transform(hidden_BTF, weights, f"h.{layer_idx}.mlp.c_proj")
    return output_BTD


def multihead_attn(x_BMD: jax.Array, w_qkv_3DD: jax.Array, b_qkv_3D: jax.Array, w_out_DD: jax.Array, b_out_D: jax.Array, config: GPT2Config, training: bool = False) -> jnp.ndarray:
    """Vaswani et al. (2017) https://arxiv.org/abs/1706.03762"""

    _validate_attention_shapes(x_BMD, w_qkv_3DD, b_qkv_3D, w_out_DD, b_out_D, config)

    H = config.n_heads
    K = config.d_model // config.n_heads

    x_BLD = x_BMD
    if not training:
        x_BLD = x_BMD[:, -1:, :] # take just the last token

    # split W_qkv
    w_qkv_3DHK = einops.rearrange(w_qkv_3DD, 'THREE D (H K) -> THREE D H K', H=H, K=K)
    w_q_DHK, w_k_DHK, w_v_DHK = w_qkv_3DHK[0], w_qkv_3DHK[1], w_qkv_3DHK[2]

    # split biases
    b_q_DHK = einops.rearrange(b_qkv_3D[0], '(H K) -> H K', H=H, K=K)
    b_k_DHK = einops.rearrange(b_qkv_3D[1], '(H K) -> H K', H=H, K=K)
    b_v_DHK = einops.rearrange(b_qkv_3D[2], '(H K) -> H K', H=H, K=K)

    # project into query/key/value
    query_BLHK = jnp.einsum('BLD,DHK->BLHK', x_BLD, w_q_DHK) + b_q_DHK
    key_BMHK = jnp.einsum('BMD,DHK->BMHK', x_BMD, w_k_DHK) + b_k_DHK
    value_BMHK = jnp.einsum('BMD,DHK->BMHK', x_BMD, w_v_DHK) + b_v_DHK

    # compute cos similarity over B and H. result matrix should be LM (would be MM for training)
    similarity_score_BHLM = jnp.einsum('BLHK,BMHK->BHLM', query_BLHK, key_BMHK)

    b, l, h, k = query_BLHK.shape

    scaled_score_BHLM = similarity_score_BHLM / (k**0.5) # scale by attention k/v dim

    # causal mask
    l, m = scaled_score_BHLM.shape[-2:]

    block_upper_LM = jnp.triu(jnp.ones((l, m)), k=1) # triu takes in a matrix as input
    causal_mask_LM = jnp.where(block_upper_LM == 1, jnp.finfo(scaled_score_BHLM.dtype).min, 0.0) # -inf for blocked values, 0 otherwise

    causal_mask_BHLM = einops.rearrange(causal_mask_LM, 'L M -> 1 1 L M')

    masked_score_BHLM = scaled_score_BHLM + causal_mask_BHLM

    softmaxed_score_BHLM = jax.nn.softmax(masked_score_BHLM, axis=-1)

    weights_BLHK = jnp.einsum('BHLM,BMHK->BLHK', softmaxed_score_BHLM, value_BMHK) # dot over BH to BHLK, reshape to BL

    weights_BLD = einops.rearrange(weights_BLHK, 'B L H K -> B L (H K)')

    attn_out_BLD = jnp.einsum('BLD,DD->BLD', weights_BLD, w_out_DD) + b_out_D

    return attn_out_BLD


def multi_head_attention(x_BTD: Array, weights: Dict[str, Array], layer_idx: int, config: GPT2Config) -> Array:
    """Multi-head self-attention."""
    B, T, D = x_BTD.shape
    N = config.n_heads
    H = D // N

    assert D % N == 0

    # Compute Q, K, V and split
    qkv_BT3D = linear_transform(x_BTD, weights, f"h.{layer_idx}.attn.c_attn")
    q_BTD, k_BTD, v_BTD = jnp.split(qkv_BT3D, 3, axis=-1)

    # Reshape for multi-head: (B, T, N, H)
    q_BTNH = q_BTD.reshape(B, T, N, H)
    k_BTNH = k_BTD.reshape(B, T, N, H)
    v_BTNH = v_BTD.reshape(B, T, N, H)

    # Transpose to (B, N, T, H) for attention computation
    q_BNTH = jnp.transpose(q_BTNH, (0, 2, 1, 3))
    k_BNTH = jnp.transpose(k_BTNH, (0, 2, 1, 3))
    v_BNTH = jnp.transpose(v_BTNH, (0, 2, 1, 3))

    # Scaled dot-product attention
    scores_BNTT = jnp.matmul(q_BNTH, jnp.transpose(k_BNTH, (0, 1, 3, 2)))
    scores_BNTT = scores_BNTT / (jnp.float32(H) ** 0.5)

    # Causal mask
    causal_mask_TT = jnp.tril(jnp.ones((T, T)))
    mask_value = jnp.finfo(scores_BNTT.dtype).min
    scores_BNTT = jnp.where(causal_mask_TT[None, None, :, :], scores_BNTT, mask_value)

    # Softmax and apply to values
    attn_weights_BNTT = jax.nn.softmax(scores_BNTT, axis=-1)
    attn_weights_BNTT = attn_weights_BNTT.astype(v_BNTH.dtype)
    attn_output_BNTH = jnp.matmul(attn_weights_BNTT, v_BNTH)

    # Transpose back and reshape
    attn_output_BTNH = jnp.transpose(attn_output_BNTH, (0, 2, 1, 3))
    attn_output_BTD = attn_output_BTNH.reshape(B, T, D)

    # Output projection
    output_BTD = linear_transform(attn_output_BTD, weights, f"h.{layer_idx}.attn.c_proj")

    return output_BTD
