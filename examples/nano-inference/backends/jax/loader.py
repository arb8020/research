#!/usr/bin/env python3
"""Weight loading utilities for JAX GPT-2."""

import sys
from pathlib import Path

# Add nano-inference root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax.numpy as jnp
from typing import Dict, Optional
from jax import Array

from utils.weights import (
    load_gpt2_weights,
    download_gpt2_weights,
    GPT2Weights
)


def load_weights(model_name: str = "gpt2", cache_dir: Optional[str] = None) -> Dict[str, Array]:
    """Load GPT-2 weights and convert to JAX arrays.

    Args:
        model_name: HuggingFace model name (default: "gpt2")
        cache_dir: Optional cache directory for downloaded weights

    Returns:
        Dictionary mapping weight names to JAX arrays
    """
    # Download weights
    model_dir = download_gpt2_weights(model_name, cache_dir)

    # Load weights as numpy arrays
    weights_obj = load_gpt2_weights(model_dir)

    # Convert to JAX arrays
    weights = {name: jnp.array(param) for name, param in weights_obj.params.items()}

    # Convert HuggingFace format to our format (add convenience aliases)
    weights = convert_hf_weights_to_jax_format(weights)

    return weights


def convert_hf_weights_to_jax_format(hf_weights: Dict[str, Array]) -> Dict[str, Array]:
    """Convert HuggingFace weight names to our expected format.

    Removes 'transformer.' prefix and adds convenience aliases.
    """
    converted = {}

    for name, param in hf_weights.items():
        clean_name = name.replace('transformer.', '')
        converted[clean_name] = param

        if clean_name == 'wte.weight':
            converted['wte'] = param
        elif clean_name == 'wpe.weight':
            converted['wpe'] = param

    return converted
