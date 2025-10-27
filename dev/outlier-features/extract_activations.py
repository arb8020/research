"""Activation extraction from transformer models using nnsight.

Adapted from extract_activations.py
Memory-optimized with chunked layer processing.
"""

import json
import logging
import torch
from pathlib import Path
from datetime import datetime
from nnsight import LanguageModel

logger = logging.getLogger(__name__)


def get_model_layers(model):
    """Get the layers module from a model, handling multimodal and decoder-only architectures.

    Args:
        model: nnsight LanguageModel

    Returns:
        The layers ModuleList (handles multiple architecture patterns)
    """
    # Access underlying model (nnsight uses either .model or ._model)
    underlying_model = None
    if hasattr(model, 'model'):
        underlying_model = model.model
    elif hasattr(model, '_model'):
        underlying_model = model._model
    else:
        # Fallback: assume model is the wrapped object itself
        underlying_model = model

    # Handle multimodal models (e.g., Gemma-3 VLMs) that have language_model.layers
    if hasattr(underlying_model, 'language_model') and hasattr(underlying_model.language_model, 'layers'):
        return underlying_model.language_model.layers
    # Handle decoder-only models (e.g., OPT, GPT-J) that have decoder.layers
    elif hasattr(underlying_model, 'decoder') and hasattr(underlying_model.decoder, 'layers'):
        return underlying_model.decoder.layers
    # Handle GPT2 models that have transformer.h
    elif hasattr(underlying_model, 'transformer') and hasattr(underlying_model.transformer, 'h'):
        return underlying_model.transformer.h
    # Handle standard models (Llama, Qwen, Mistral, etc.) with model.layers
    else:
        return underlying_model.layers


def get_layernorm_outputs(layer):
    """Get layernorm outputs from a layer, handling different architectures.

    Args:
        layer: A single transformer layer module

    Returns:
        Tuple of (ln_attn_output, ln_mlp_output) from the layer

    Note:
        Architecture patterns:
        - Pre-norm (Llama, Qwen, Mistral): input_layernorm, post_attention_layernorm
        - Pre-norm (GPT2): ln_1, ln_2
        - Post-norm (OPT): Use input to self_attn and fc1 (MLP input) instead of layernorm outputs
    """
    # GPT2 models use ln_1 (before attn) and ln_2 (before MLP)
    if hasattr(layer, 'ln_1') and hasattr(layer, 'ln_2'):
        return layer.ln_1.output, layer.ln_2.output
    # OPT models (post-norm): layernorms are AFTER sublayers, so we use sublayer inputs instead
    elif hasattr(layer, 'self_attn') and hasattr(layer, 'fc1'):
        # For OPT: get input to self_attn and input to fc1 (first MLP layer)
        return layer.self_attn.input[0][0], layer.fc1.input[0][0]
    # Standard pre-norm models use input_layernorm and post_attention_layernorm
    else:
        return layer.input_layernorm.output, layer.post_attention_layernorm.output


def extract_activations_batch(
    model: LanguageModel,
    texts: list[str],
    layers: list[int]
) -> dict[str, torch.Tensor]:
    """Pure function: extract activations from batch of texts.

    Args:
        model: Loaded nnsight LanguageModel
        texts: List of input texts to process
        layers: List of layer indices to extract from

    Returns:
        Dict mapping layer names to activation tensors

    Raises:
        AssertionError: If inputs are invalid or outputs have unexpected shapes
    """
    assert isinstance(texts, list), f"Expected list of texts, got {type(texts)}"
    assert len(texts) > 0, "texts cannot be empty"
    assert isinstance(layers, list), f"Expected list of layers, got {type(layers)}"
    assert len(layers) > 0, "layers cannot be empty"

    activations = {}
    model_layers = get_model_layers(model)
    with model.trace(texts) as tracer:
        for layer_idx in layers:
            ln_attn_output, ln_mlp_output = get_layernorm_outputs(model_layers[layer_idx])
            ln_into_attn = ln_attn_output.save()
            ln_into_mlp = ln_mlp_output.save()

            activations[f"layer_{layer_idx}_ln_attn"] = ln_into_attn
            activations[f"layer_{layer_idx}_ln_mlp"] = ln_into_mlp

    # Convert proxies to tensors and validate
    result = {}
    for layer_name, activation_proxy in activations.items():
        tensor = activation_proxy.detach().cpu()
        assert tensor.dim() == 3, f"Expected 3D tensor for {layer_name}, got shape {tensor.shape}"
        result[layer_name] = tensor

    return result


def extract_activations_optimized(
    llm: LanguageModel,
    texts: list[str],
    layers: list[int] | None = None,
    save_dir: str = "./activations",
    chunk_size: int | None = None
) -> tuple[str, dict]:
    """Memory-optimized activation extraction with chunked layer processing.

    Args:
        llm: Pre-loaded nnsight LanguageModel
        texts: List of input texts to process
        layers: List of layer indices to extract from (None = all layers)
        save_dir: Directory to save results
        chunk_size: Number of layers to process at once (None = all together)

    Returns:
        Tuple of (run_dir, metadata)

    Raises:
        AssertionError: If inputs are invalid
    """
    assert isinstance(texts, list), f"Expected list of texts, got {type(texts)}"
    assert len(texts) > 0, "texts cannot be empty"

    # Auto-detect all layers if None provided
    if layers is None:
        model_layers = get_model_layers(llm)
        num_layers = len(model_layers)
        if hasattr(llm.model, 'language_model'):
            logger.info(f"Detected multimodal model, using language_model.layers")
        layers = list(range(num_layers))
        logger.info(f"Auto-detected {num_layers} layers: {layers[0]}-{layers[-1]}")
    else:
        assert isinstance(layers, list), f"Expected list of layers, got {type(layers)}"
        assert len(layers) > 0, "layers cannot be empty"

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(save_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Process layers in chunks to reduce peak memory
    if chunk_size is None:
        layer_chunks = [layers]
        logger.info(f"Processing all {len(layers)} layers together (no chunking)")
    else:
        assert chunk_size > 0, f"chunk_size must be positive, got {chunk_size}"
        layer_chunks = [layers[i:i + chunk_size] for i in range(0, len(layers), chunk_size)]
        logger.info(f"Processing {len(layers)} layers in {len(layer_chunks)} chunks of {chunk_size}")

    saved_files = []

    for chunk_idx, layers_chunk in enumerate(layer_chunks):
        logger.info(f"  Chunk {chunk_idx + 1}/{len(layer_chunks)}: layers {layers_chunk[0]}-{layers_chunk[-1]}")

        # Extract activations for this chunk only
        activations = {}
        model_layers = get_model_layers(llm)
        with torch.inference_mode(), llm.trace(texts) as tracer:
            for layer_idx in layers_chunk:
                ln_attn_output, ln_mlp_output = get_layernorm_outputs(model_layers[layer_idx])
                ln_into_attn = ln_attn_output.save()
                ln_into_mlp = ln_mlp_output.save()

                activations[f"layer_{layer_idx}_ln_attn"] = ln_into_attn
                activations[f"layer_{layer_idx}_ln_mlp"] = ln_into_mlp

        # Immediately convert to CPU and save to disk
        for layer_name, activation_proxy in activations.items():
            tensor = activation_proxy.detach().to(torch.bfloat16).cpu()
            assert tensor.dim() == 3, f"Expected 3D tensor for {layer_name}, got shape {tensor.shape}"

            # Save activation
            activation_file = run_dir / f"{layer_name}_activations.pt"
            torch.save(tensor, activation_file)
            saved_files.append(str(activation_file))

            logger.info(f"    Saved {layer_name}: shape={tuple(tensor.shape)} -> {activation_file.name}")

        # Clear chunk activations and GPU cache
        del activations
        torch.cuda.empty_cache()

    # Save metadata
    metadata = {
        "model_name": llm.tokenizer.name_or_path if hasattr(llm.tokenizer, 'name_or_path') else str(llm),
        "num_sequences": len(texts),
        "sequence_texts": texts,
        "layers_extracted": layers,
        "chunk_size": chunk_size,
        "num_chunks": len(layer_chunks),
        "saved_files": saved_files,
        "timestamp": timestamp
    }

    metadata_file = run_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"  Saved metadata: {metadata_file.name}")

    return str(run_dir), metadata
