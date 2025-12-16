#!/usr/bin/env python3
"""
Simple example showing how to use the JAX GPT-2 implementation.

Usage:
    python backends/jax/example.py
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

# Import local modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backends.jax.loader import load_weights
from backends.jax.model import gpt2_forward
from config import GPT2Config


def main():
    print("🚀 JAX GPT-2 Example")
    print("=" * 50)

    # Load model weights
    print("\n📦 Loading GPT-2 weights...")
    weights = load_weights("gpt2")
    print("✅ Loaded weights")

    # Create config
    config = GPT2Config()
    print(f"📋 Config: {config.n_layers} layers, {config.n_heads} heads, {config.d_model} dim")

    # Example input: "Hello world" tokenized
    # Token IDs from GPT-2 tokenizer: [15496, 995]
    print("\n🔤 Example input: 'Hello world'")
    input_ids_BT = jnp.array([[15496, 995]])
    print(f"   Token IDs: {input_ids_BT.tolist()}")
    print(f"   Shape: {input_ids_BT.shape}")

    # Run forward pass
    print("\n🔥 Running forward pass...")
    logits_BTV = gpt2_forward(input_ids_BT, weights, config)

    print("✅ Forward pass complete!")
    print(f"   Logits shape: {logits_BTV.shape}")
    print(f"   Logits range: [{logits_BTV.min():.3f}, {logits_BTV.max():.3f}]")

    # Get predicted next tokens
    print("\n🎯 Top-5 predicted next tokens:")
    next_token_logits = logits_BTV[0, -1, :]  # Last position
    top_k = 5
    top_indices = jnp.argsort(next_token_logits)[-top_k:][::-1]
    top_probs = jax.nn.softmax(next_token_logits)[top_indices]

    for i, (token_id, prob) in enumerate(zip(top_indices, top_probs, strict=False)):
        print(f"   {i + 1}. Token {token_id}: {prob:.4f}")

    print("\n✅ Example complete!")


if __name__ == "__main__":
    main()
