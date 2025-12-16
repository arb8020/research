#!/usr/bin/env python3
"""Create residual stream visualizations showing outlier features."""

import sys

sys.path.append("/Users/chiraagbalu/research")

import numpy as np

from utils.render import render

# Set random seed for reproducibility
np.random.seed(42)


def create_residual_stream_example(batch_size, seq_len, hidden_dim, outlier_dim=None):
    """
    Create example showing residual stream + layer contribution = new residual stream.

    Args:
        batch_size: Number of sequences in batch
        seq_len: Sequence length
        hidden_dim: Hidden dimension size
        outlier_dim: If specified, this dimension will have outlier values
    """
    # Create residual stream with small values
    residual_in = np.random.randint(-3, 4, size=(batch_size, seq_len, hidden_dim))

    # Create layer contribution (also small values)
    layer_contrib = np.random.randint(-2, 3, size=(batch_size, seq_len, hidden_dim))

    # If outlier_dim specified, add outlier features
    if outlier_dim is not None:
        # Add large outlier values across all layers and most sequence positions
        residual_in[:, :, outlier_dim] = np.random.randint(6, 10, size=(batch_size, seq_len))
        # Layer contribution for outlier dimension
        layer_contrib[:, :, outlier_dim] = np.random.randint(-2, 2, size=(batch_size, seq_len))

    # Compute new residual stream
    residual_out = residual_in + layer_contrib

    return residual_in, layer_contrib, residual_out


def main():
    """Generate three examples with different sizes."""

    print("=" * 80)
    print("RESIDUAL STREAM VISUALIZATIONS")
    print("=" * 80)
    print()
    print("These visualizations show how transformer layers process information through")
    print("the residual stream. Format: [Batch, Sequence, Hidden_Dim]")
    print()

    # Example 1: Small tensor without outliers (2x3x4)
    print("\n" + "=" * 80)
    print("Example 1: Normal Residual Stream [2, 3, 4] - No Outlier Features")
    print("=" * 80)
    print()

    res_in, layer, res_out = create_residual_stream_example(2, 3, 4, outlier_dim=None)

    print("Input Residual Stream:")
    print(render(res_in))
    print()
    print("           +")
    print()
    print("Layer Contribution:")
    print(render(layer))
    print()
    print("           =")
    print()
    print("Output Residual Stream:")
    print(render(res_out))
    print()

    # Example 2: With outlier in dimension 2 (2x3x4)
    print("\n" + "=" * 80)
    print("Example 2: With Outlier Feature [2, 3, 4] - Dimension 2 has Outliers")
    print("=" * 80)
    print()

    res_in, layer, res_out = create_residual_stream_example(2, 3, 4, outlier_dim=2)

    print("Input Residual Stream (notice dim 2 has large values ~6-9):")
    print(render(res_in))
    print()
    print("           +")
    print()
    print("Layer Contribution:")
    print(render(layer))
    print()
    print("           =")
    print()
    print("Output Residual Stream (dim 2 still has outliers):")
    print(render(res_out))
    print()

    # Example 3: Slightly larger with outlier (3x4x5)
    print("\n" + "=" * 80)
    print("Example 3: Larger Tensor [3, 4, 5] - Dimension 3 has Outliers")
    print("=" * 80)
    print()

    res_in, layer, res_out = create_residual_stream_example(3, 4, 5, outlier_dim=3)

    print("Input Residual Stream:")
    print(render(res_in))
    print()
    print("           +")
    print()
    print("Layer Contribution:")
    print(render(layer))
    print()
    print("           =")
    print()
    print("Output Residual Stream:")
    print(render(res_out))
    print()

    # Save to file
    output_file = "residual_stream_examples.txt"
    with open(output_file, "w") as f:
        # Redirect stdout to file
        original_stdout = sys.stdout
        sys.stdout = f

        # Re-run all examples
        print("=" * 80)
        print("RESIDUAL STREAM VISUALIZATIONS")
        print("=" * 80)
        print()
        print("These visualizations show how transformer layers process information through")
        print("the residual stream. Format: [Batch, Sequence, Hidden_Dim]")
        print()

        print("\n" + "=" * 80)
        print("Example 1: Normal Residual Stream [2, 3, 4] - No Outlier Features")
        print("=" * 80)
        print()

        np.random.seed(42)
        res_in, layer, res_out = create_residual_stream_example(2, 3, 4, outlier_dim=None)

        print("Input Residual Stream:")
        print(render(res_in))
        print()
        print("           +")
        print()
        print("Layer Contribution:")
        print(render(layer))
        print()
        print("           =")
        print()
        print("Output Residual Stream:")
        print(render(res_out))
        print()

        print("\n" + "=" * 80)
        print("Example 2: With Outlier Feature [2, 3, 4] - Dimension 2 has Outliers")
        print("=" * 80)
        print()

        res_in, layer, res_out = create_residual_stream_example(2, 3, 4, outlier_dim=2)

        print("Input Residual Stream (notice dim 2 has large values ~6-9):")
        print(render(res_in))
        print()
        print("           +")
        print()
        print("Layer Contribution:")
        print(render(layer))
        print()
        print("           =")
        print()
        print("Output Residual Stream (dim 2 still has outliers):")
        print(render(res_out))
        print()

        print("\n" + "=" * 80)
        print("Example 3: Larger Tensor [3, 4, 5] - Dimension 3 has Outliers")
        print("=" * 80)
        print()

        res_in, layer, res_out = create_residual_stream_example(3, 4, 5, outlier_dim=3)

        print("Input Residual Stream:")
        print(render(res_in))
        print()
        print("           +")
        print()
        print("Layer Contribution:")
        print(render(layer))
        print()
        print("           =")
        print()
        print("Output Residual Stream:")
        print(render(res_out))

        sys.stdout = original_stdout

    print(f"\n✅ Examples saved to: {output_file}")


if __name__ == "__main__":
    main()
