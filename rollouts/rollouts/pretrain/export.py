"""Export pretrain weights to HuggingFace format.

Usage:
    from rollouts.pretrain.export import export_to_hf

    # Load checkpoint
    ckpt = torch.load("output/step_00001000.pt")
    weights = ckpt["weights"]

    # Export to HF format
    export_to_hf(weights, config, "output/hf_model")

    # Now loadable via transformers or SGLang
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("output/hf_model")
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from .config import ModelConfig


def export_to_hf(
    weights: dict[str, torch.Tensor],
    config: ModelConfig,
    output_dir: str | Path,
) -> Path:
    """Export pretrain weights to HuggingFace LlamaForCausalLM format.

    Args:
        weights: Functional weight dict from pretrain
        config: Model config
        output_dir: Directory to save HF model

    Returns:
        Path to output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert weights to HF naming (add "model." prefix)
    hf_weights = {}
    for name, tensor in weights.items():
        # Remove grad tracking for export
        tensor = tensor.detach().clone()

        # Map to HF naming
        if name in ("embed_tokens.weight", "norm.weight"):
            hf_name = f"model.{name}"
        elif name.startswith("layers."):
            hf_name = f"model.{name}"
        elif name == "lm_head.weight":
            hf_name = name  # lm_head stays at top level
        else:
            raise ValueError(f"Unknown weight name: {name}")

        hf_weights[hf_name] = tensor

    # Save weights as safetensors
    save_file(hf_weights, output_dir / "model.safetensors")

    # Write config.json (LlamaConfig format)
    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": config.dim,
        "intermediate_size": config.mlp_dim,
        "num_hidden_layers": config.n_layers,
        "num_attention_heads": config.n_heads,
        "num_key_value_heads": config.n_kv_heads,
        "vocab_size": config.vocab_size,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "head_dim": config.head_dim,
        "hidden_act": "relu2" if config.use_relu2 else "silu",
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.40.0",
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)

    return output_dir


def load_and_export(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    config: ModelConfig | None = None,
) -> Path:
    """Load a checkpoint and export to HF format.

    Args:
        checkpoint_path: Path to pretrain checkpoint (.pt file)
        output_dir: Directory to save HF model
        config: Model config (if None, tries to load from checkpoint metadata)

    Returns:
        Path to output directory
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = ckpt["weights"]

    if config is None:
        raise ValueError("config is required. Future: load from checkpoint metadata or config.json")

    return export_to_hf(weights, config, output_dir)
