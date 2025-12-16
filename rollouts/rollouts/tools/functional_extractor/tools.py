"""Helper tools for exploring HuggingFace model structure.

Usage:
    from transformers import AutoModelForCausalLM
    from tools.functional_extractor.tools import (
        read_module_source,
        get_weight_info,
        capture_intermediate,
    )

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

    # Get source code of a module
    source = read_module_source(model, "model.layers.0.self_attn")

    # Get weight shapes matching a pattern
    weights = get_weight_info(model, "layers.0")

    # Capture activation at a layer
    input_ids = torch.tensor([[1, 2, 3, 4]])
    hidden = capture_intermediate(model, "model.layers.0", input_ids)
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class WeightInfo:
    """Info about a model weight."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str


@dataclass(frozen=True)
class ModuleSource:
    """Source code info for a module."""

    path: str
    class_name: str
    source: str
    file_path: str


@dataclass(frozen=True)
class IntermediateCapture:
    """Captured intermediate activation."""

    path: str
    output: Any  # Tensor or tuple of Tensors
    output_shape: tuple[int, ...] | list[tuple[int, ...]]


def _get_module_by_path(model: Any, path: str) -> Any:
    """Navigate to a submodule by dot-separated path.

    Args:
        model: The root model
        path: Dot-separated path like "model.layers.0.self_attn"

    Returns:
        The submodule at that path
    """
    assert path, "Path cannot be empty"

    current = model
    parts = path.split(".")

    for part in parts:
        if part.isdigit():
            # Handle indexed access (e.g., layers.0)
            current = current[int(part)]
        else:
            # Handle attribute access
            assert hasattr(current, part), f"Module has no attribute '{part}' at path '{path}'"
            current = getattr(current, part)

    return current


def read_module_source(model: Any, path: str) -> ModuleSource:
    """Get source code of a module at the given path.

    Args:
        model: The HuggingFace model
        path: Dot-separated path like "model.layers.0.self_attn"

    Returns:
        ModuleSource with class name, source code, and file path

    Example:
        >>> source = read_module_source(model, "model.layers.0.self_attn")
        >>> print(source.class_name)  # "Qwen2Attention"
        >>> print(source.source[:100])  # First 100 chars of source
    """
    assert model is not None, "Model cannot be None"
    assert path, "Path cannot be empty"

    module = _get_module_by_path(model, path)
    module_class = type(module)

    source = inspect.getsource(module_class)
    file_path = inspect.getfile(module_class)

    return ModuleSource(
        path=path,
        class_name=module_class.__name__,
        source=source,
        file_path=file_path,
    )


def get_weight_info(model: Any, pattern: str = "") -> list[WeightInfo]:
    """Get info about model weights matching a pattern.

    Args:
        model: The HuggingFace model
        pattern: Regex pattern to filter weight names (empty = all weights)

    Returns:
        List of WeightInfo for matching weights

    Example:
        >>> weights = get_weight_info(model, "layers.0")
        >>> for w in weights:
        ...     print(f"{w.name}: {w.shape}")
    """
    assert model is not None, "Model cannot be None"

    results = []
    compiled_pattern = re.compile(pattern) if pattern else None

    for name, param in model.named_parameters():
        if compiled_pattern is None or compiled_pattern.search(name):
            results.append(
                WeightInfo(
                    name=name,
                    shape=tuple(param.shape),
                    dtype=str(param.dtype),
                    device=str(param.device),
                )
            )

    # Also check buffers (like positional embeddings that aren't parameters)
    for name, buffer in model.named_buffers():
        if compiled_pattern is None or compiled_pattern.search(name):
            results.append(
                WeightInfo(
                    name=name,
                    shape=tuple(buffer.shape),
                    dtype=str(buffer.dtype),
                    device=str(buffer.device),
                )
            )

    assert len(results) > 0 or pattern, "Model has no weights (this shouldn't happen)"

    return results


def capture_intermediate(
    model: Any,
    layer_path: str,
    input_ids: Tensor,
    **forward_kwargs: Any,
) -> IntermediateCapture:
    """Run forward pass and capture activation at a specific layer.

    Args:
        model: The HuggingFace model
        layer_path: Dot-separated path to the layer to capture
        input_ids: Input tensor to run through the model
        **forward_kwargs: Additional kwargs for model.forward()

    Returns:
        IntermediateCapture with the output tensor(s) at that layer

    Example:
        >>> input_ids = torch.tensor([[1, 2, 3, 4]])
        >>> capture = capture_intermediate(model, "model.layers.0", input_ids)
        >>> print(capture.output_shape)  # (1, 4, 896) for Qwen2.5-0.5B
    """
    import torch

    assert model is not None, "Model cannot be None"
    assert layer_path, "Layer path cannot be empty"
    assert input_ids is not None, "input_ids cannot be None"

    module = _get_module_by_path(model, layer_path)
    captured_output = None

    def hook(module: Any, input: Any, output: Any) -> None:
        nonlocal captured_output
        captured_output = output

    handle = module.register_forward_hook(hook)

    try:
        with torch.no_grad():
            model(input_ids, **forward_kwargs)
    finally:
        handle.remove()

    assert captured_output is not None, f"Hook did not capture output at '{layer_path}'"

    # Get shape(s) of output
    if isinstance(captured_output, tuple):
        # Some layers return tuples (hidden_states, attention_weights, etc.)
        output_shape = [tuple(t.shape) if hasattr(t, "shape") else None for t in captured_output]
    else:
        output_shape = tuple(captured_output.shape)

    return IntermediateCapture(
        path=layer_path,
        output=captured_output,
        output_shape=output_shape,
    )


def list_modules(model: Any, max_depth: int = 3) -> list[str]:
    """List all module paths in the model up to a certain depth.

    Args:
        model: The HuggingFace model
        max_depth: Maximum depth to traverse (default 3)

    Returns:
        List of dot-separated module paths

    Example:
        >>> paths = list_modules(model, max_depth=2)
        >>> print(paths[:5])
        ['model', 'model.embed_tokens', 'model.layers', 'model.norm', 'lm_head']
    """
    assert model is not None, "Model cannot be None"
    assert max_depth > 0, "max_depth must be positive"

    paths = []

    def recurse(module: Any, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return

        for name, child in module.named_children():
            path = f"{prefix}.{name}" if prefix else name
            paths.append(path)
            recurse(child, path, depth + 1)

    recurse(model, "", 1)
    return paths
