"""Outlier detection and analysis for transformer activations.

Implements methodology from Dettmers et al. (2022) "LLM.int8()":
- Magnitude threshold: ≥6.0 activation magnitude
- Layer coverage: ≥25% of transformer layers affected
- Sequence coverage: ≥6% of sequence positions affected

Adapted from analyze_activations.py
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def load_activations(run_dir: str) -> tuple[dict[str, torch.Tensor], dict]:
    """Load activations from a run directory.

    Args:
        run_dir: Path to run directory containing activation files

    Returns:
        Tuple of (activations dict, metadata dict)

    Raises:
        FileNotFoundError: If metadata or activation files not found
    """
    run_path = Path(run_dir)
    assert run_path.exists(), f"Run directory not found: {run_dir}"

    # Load metadata
    metadata_file = run_path / "metadata.json"
    assert metadata_file.exists(), f"Metadata file not found: {metadata_file}"

    with open(metadata_file) as f:
        metadata = json.load(f)

    # Load activation tensors
    activations = {}
    for layer_idx in metadata["layers_extracted"]:
        attn_file = run_path / f"layer_{layer_idx}_ln_attn_activations.pt"
        mlp_file = run_path / f"layer_{layer_idx}_ln_mlp_activations.pt"

        assert attn_file.exists(), f"Activation file not found: {attn_file}"
        assert mlp_file.exists(), f"Activation file not found: {mlp_file}"

        attn_tensor = torch.load(attn_file, map_location="cpu")
        mlp_tensor = torch.load(mlp_file, map_location="cpu")

        activations[f"layer_{layer_idx}_ln_attn"] = attn_tensor
        activations[f"layer_{layer_idx}_ln_mlp"] = mlp_tensor

        logger.debug(
            f"Loaded layer_{layer_idx} activations: "
            f"ln_attn={tuple(attn_tensor.shape)}, ln_mlp={tuple(mlp_tensor.shape)}"
        )

    return activations, metadata


def find_outliers_in_activations(
    activations: dict[str, torch.Tensor], magnitude_threshold: float = 6.0
) -> dict:
    """Find outlier features across all layers and sequence positions.

    Paper methodology: "We track dimensions h_i, 0 ≤ i ≤ h, which have at least
    one value with a magnitude of α ≥ 6"

    Args:
        activations: Dict of {layer_name: tensor} from load_activations()
        magnitude_threshold: Minimum magnitude to consider as outlier (paper uses 6.0)

    Returns:
        Dict with outlier statistics and locations
    """
    assert magnitude_threshold > 0, (
        f"magnitude_threshold must be positive, got {magnitude_threshold}"
    )
    logger.info(f"Searching for outliers with magnitude >= {magnitude_threshold}")

    # Track outliers by feature dimension h_i across all layers
    # Paper: "we only collect statistics if these outliers occur in the SAME feature dimension h_i"
    feature_outliers = defaultdict(list)  # feature_dim -> [(layer, seq_pos, value), ...]
    layer_stats = {}

    for layer_name, tensor in activations.items():
        logger.debug(f"Analyzing {layer_name}: shape={tuple(tensor.shape)}")

        # tensor shape: (batch, seq_len, d_model) where d_model contains feature dimensions h_i
        batch_size, seq_len, d_model = tensor.shape
        assert batch_size > 0 and seq_len > 0 and d_model > 0, (
            f"Invalid tensor shape: {tensor.shape}"
        )

        # Find outliers using paper's magnitude criterion (≥ 6.0)
        outlier_mask = torch.abs(tensor) >= magnitude_threshold
        outlier_positions = torch.where(outlier_mask)

        num_outliers = outlier_mask.sum().item()
        max_val = tensor.abs().max().item()

        layer_stats[layer_name] = {
            "num_outliers": num_outliers,
            "max_magnitude": max_val,
            "outlier_percentage": (num_outliers / tensor.numel()) * 100,
            "shape": tuple(tensor.shape),
        }

        logger.debug(
            f"  Found {num_outliers} outliers ({layer_stats[layer_name]['outlier_percentage']:.3f}%)"
        )
        logger.debug(f"  Max magnitude: {max_val:.3f}")

        # Record outliers by feature dimension h_i
        if num_outliers > 0:
            batch_indices, seq_indices, feature_indices = outlier_positions
            values = tensor[outlier_positions]

            # Group outliers by their feature dimension h_i
            for i in range(len(feature_indices)):
                feature_dim = feature_indices[i].item()  # This is h_i in paper notation
                seq_pos = seq_indices[i].item()  # Position in sequence dimension s
                value = values[i].item()  # Activation value with magnitude ≥ 6.0

                feature_outliers[feature_dim].append({
                    "layer": layer_name,
                    "seq_pos": seq_pos,
                    "value": value,
                })

    return {
        "feature_outliers": dict(feature_outliers),
        "layer_stats": layer_stats,
        "threshold": magnitude_threshold,
    }


def analyze_systematic_outliers(
    outlier_info: dict, min_layer_percentage: float = 0.25, min_seq_percentage: float = 0.06
) -> list[dict]:
    """Find systematic outliers that appear consistently across layers and positions.

    Paper's criteria: "magnitude ≥ 6.0, affects ≥ 25% of layers, affects ≥ 6% of
    sequence dimensions"

    Args:
        outlier_info: Output from find_outliers_in_activations()
        min_layer_percentage: Minimum percentage of layers (paper uses 25%)
        min_seq_percentage: Minimum percentage of sequence positions (paper uses 6%)

    Returns:
        List of systematic outlier features meeting all criteria
    """
    assert 0 < min_layer_percentage <= 1, (
        f"min_layer_percentage must be in (0, 1], got {min_layer_percentage}"
    )
    assert 0 < min_seq_percentage <= 1, (
        f"min_seq_percentage must be in (0, 1], got {min_seq_percentage}"
    )

    feature_outliers = outlier_info["feature_outliers"]
    layer_stats = outlier_info["layer_stats"]

    total_layers = len(layer_stats)
    assert total_layers > 0, "No layers in outlier_info"

    # Get sequence length from any layer (they should all be the same)
    seq_len = next(iter(layer_stats.values()))["shape"][1]
    assert seq_len > 0, f"Invalid sequence length: {seq_len}"

    systematic_features = []

    logger.info("Analyzing systematic outliers:")
    logger.info(
        f"Criteria: ≥{min_layer_percentage * 100}% of {total_layers} layers, "
        f"≥{min_seq_percentage * 100}% of {seq_len} seq positions"
    )

    for feature_dim, outlier_list in feature_outliers.items():
        # Paper criterion 1: "affects at least 25% of layers"
        layers_with_outlier = set(item["layer"] for item in outlier_list)
        layer_percentage = len(layers_with_outlier) / total_layers

        # Paper criterion 2: "affects at least 6% of the sequence dimensions"
        seq_positions_with_outlier = set(item["seq_pos"] for item in outlier_list)
        seq_percentage = len(seq_positions_with_outlier) / seq_len

        # Apply paper's systematic criteria (magnitude ≥6.0 already applied)
        meets_layer_criteria = layer_percentage >= min_layer_percentage
        meets_seq_criteria = seq_percentage >= min_seq_percentage

        if meets_layer_criteria and meets_seq_criteria:
            max_magnitude = max(abs(item["value"]) for item in outlier_list)

            systematic_features.append({
                "feature_dim": feature_dim,
                "layer_percentage": layer_percentage,
                "seq_percentage": seq_percentage,
                "max_magnitude": max_magnitude,
                "total_occurrences": len(outlier_list),
                "layers_affected": sorted(set(item["layer"] for item in outlier_list)),
                "example_values": [item["value"] for item in outlier_list[:5]],
            })

            logger.debug(
                f"  Feature {feature_dim}: {layer_percentage:.1%} layers, "
                f"{seq_percentage:.1%} seq_pos, max_mag={max_magnitude:.2f}"
            )

    systematic_features.sort(key=lambda x: x["max_magnitude"], reverse=True)

    logger.info(f"Found {len(systematic_features)} systematic outlier features")

    return systematic_features


def analyze_run_for_outliers(
    run_dir: str,
    magnitude_threshold: float = 6.0,
    min_layer_percentage: float = 0.25,
    min_seq_percentage: float = 0.06,
) -> tuple[list[dict], dict]:
    """Complete pipeline: load activations and analyze for outliers.

    Args:
        run_dir: Path to run directory
        magnitude_threshold: Minimum magnitude for outliers
        min_layer_percentage: Minimum percentage of layers
        min_seq_percentage: Minimum percentage of sequence positions

    Returns:
        Tuple of (systematic_outliers list, outlier_info dict)
    """
    logger.info(f"Loading activations from: {run_dir}")
    activations, metadata = load_activations(run_dir)

    logger.info(f"Analyzing model: {metadata['model_name']}")
    logger.debug(f"Input sequences: {metadata['num_sequences']}")

    # Find all outliers
    outlier_info = find_outliers_in_activations(activations, magnitude_threshold)

    # Find systematic outliers
    systematic_outliers = analyze_systematic_outliers(
        outlier_info, min_layer_percentage, min_seq_percentage
    )

    return systematic_outliers, outlier_info
