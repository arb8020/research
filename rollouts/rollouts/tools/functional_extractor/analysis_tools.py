#!/usr/bin/env python3
"""Analysis tools for functional model extraction.

Tools for tracing, coverage analysis, and comparison to help convert
HuggingFace models to functional PyTorch code.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class OpNode:
    """A single operation in the traced graph."""

    name: str
    op: str  # placeholder, call_function, call_method, call_module, output
    target: str
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)


@dataclass
class TraceResult:
    """Result of tracing a model."""

    nodes: list[OpNode]
    inputs: list[str]
    outputs: list[str]

    def get_ops(self) -> list[OpNode]:
        """Get only call_function nodes (actual ops)."""
        return [n for n in self.nodes if n.op == "call_function"]

    def op_counts(self) -> Counter:
        """Count occurrences of each op type."""
        return Counter(n.target for n in self.get_ops())

    def find_ops(self, pattern: str) -> list[OpNode]:
        """Find ops matching a pattern (case-insensitive)."""
        pattern = pattern.lower()
        return [n for n in self.get_ops() if pattern in n.target.lower()]


def trace_model(
    model: torch.nn.Module,
    example_inputs: tuple,
    example_kwargs: dict | None = None,
) -> TraceResult:
    """Trace a model using torch.export.

    Args:
        model: The model to trace
        example_inputs: Tuple of example input tensors
        example_kwargs: Optional dict of keyword arguments

    Returns:
        TraceResult with the traced graph
    """
    from torch.export import export

    example_kwargs = example_kwargs or {}

    exported = export(model, example_inputs, kwargs=example_kwargs)

    nodes = []
    for node in exported.graph.nodes:
        nodes.append(
            OpNode(
                name=node.name,
                op=node.op,
                target=str(node.target),
                args=tuple(str(a)[:100] for a in node.args),
                kwargs={k: str(v)[:100] for k, v in node.kwargs.items()},
            )
        )

    inputs = [n.name for n in nodes if n.op == "placeholder"]
    outputs = [n.name for n in nodes if n.op == "output"]

    return TraceResult(nodes=nodes, inputs=inputs, outputs=outputs)


def compare_traces(trace1: TraceResult, trace2: TraceResult) -> dict:
    """Compare two traces and find differences.

    Args:
        trace1: First trace (e.g., HuggingFace model)
        trace2: Second trace (e.g., functional implementation)

    Returns:
        Dict with comparison results
    """
    ops1 = trace1.op_counts()
    ops2 = trace2.op_counts()

    all_ops = set(ops1.keys()) | set(ops2.keys())

    differences = {}
    for op in all_ops:
        c1, c2 = ops1.get(op, 0), ops2.get(op, 0)
        if c1 != c2:
            differences[op] = {"trace1": c1, "trace2": c2, "diff": c1 - c2}

    return {
        "trace1_ops": len(trace1.get_ops()),
        "trace2_ops": len(trace2.get_ops()),
        "differences": differences,
        "match": len(differences) == 0,
    }


@dataclass
class CoverageResult:
    """Result of code coverage analysis."""

    executed_lines: dict[str, set[int]]  # file -> line numbers
    executed_branches: dict[str, set[tuple[int, int]]]  # file -> (line, branch_id)
    total_lines: int
    covered_lines: int

    @property
    def coverage_pct(self) -> float:
        return (self.covered_lines / self.total_lines * 100) if self.total_lines > 0 else 0.0


def run_with_coverage(
    func: Callable,
    *args,
    source_filter: str | None = None,
    **kwargs,
) -> tuple[Any, CoverageResult]:
    """Run a function with code coverage tracking.

    Args:
        func: Function to run
        *args: Arguments to pass to func
        source_filter: Only track files containing this string (e.g., "transformers")
        **kwargs: Keyword arguments to pass to func

    Returns:
        Tuple of (function result, CoverageResult)
    """
    import coverage

    cov = coverage.Coverage(branch=True)
    cov.start()

    try:
        result = func(*args, **kwargs)
    finally:
        cov.stop()

    # Get coverage data
    data = cov.get_data()

    executed_lines = {}
    executed_branches = {}

    for filename in data.measured_files():
        if source_filter and source_filter not in filename:
            continue

        lines = data.lines(filename)
        if lines:
            executed_lines[filename] = set(lines)

        branches = data.arcs(filename)
        if branches:
            executed_branches[filename] = set(branches)

    total_lines = sum(len(lines) for lines in executed_lines.values())

    return result, CoverageResult(
        executed_lines=executed_lines,
        executed_branches=executed_branches,
        total_lines=total_lines,
        covered_lines=total_lines,  # All lines we tracked were executed
    )


def find_dead_branches(
    model: torch.nn.Module,
    example_inputs: list[tuple],
    example_kwargs_list: list[dict] | None = None,
    source_filter: str = "transformers",
) -> dict[str, set[int]]:
    """Find code branches that are never executed across multiple inputs.

    Args:
        model: Model to analyze
        example_inputs: List of example input tuples to try
        example_kwargs_list: Optional list of kwargs for each input
        source_filter: Only analyze files containing this string

    Returns:
        Dict mapping filename to set of never-executed line numbers
    """
    import coverage

    example_kwargs_list = example_kwargs_list or [{}] * len(example_inputs)

    # Run with all inputs and collect coverage
    all_executed = {}

    for inputs, kwargs in zip(example_inputs, example_kwargs_list, strict=False):
        cov = coverage.Coverage(branch=True)
        cov.start()

        with torch.no_grad():
            model(*inputs, **kwargs)

        cov.stop()
        data = cov.get_data()

        for filename in data.measured_files():
            if source_filter not in filename:
                continue

            lines = data.lines(filename) or []
            if filename not in all_executed:
                all_executed[filename] = set()
            all_executed[filename].update(lines)

    return all_executed


@dataclass
class IntermediateCapture:
    """Captured intermediate values from model execution."""

    name: str
    shape: tuple
    dtype: torch.dtype
    value: Tensor  # Stored on CPU to save GPU memory
    max_val: float
    min_val: float
    mean_val: float


def capture_intermediates(
    model: torch.nn.Module,
    inputs: tuple,
    kwargs: dict | None = None,
    layer_pattern: str = "",
) -> list[IntermediateCapture]:
    """Capture intermediate tensor values during forward pass.

    Args:
        model: Model to run
        inputs: Input tensors
        kwargs: Optional keyword arguments
        layer_pattern: Only capture layers matching this pattern

    Returns:
        List of IntermediateCapture objects
    """
    kwargs = kwargs or {}
    captures = []

    def make_hook(name: str):
        def hook(module, input, output):
            if isinstance(output, Tensor):
                captures.append(
                    IntermediateCapture(
                        name=name,
                        shape=tuple(output.shape),
                        dtype=output.dtype,
                        value=output.detach().cpu(),
                        max_val=output.max().item(),
                        min_val=output.min().item(),
                        mean_val=output.float().mean().item(),
                    )
                )
            elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], Tensor):
                captures.append(
                    IntermediateCapture(
                        name=f"{name}[0]",
                        shape=tuple(output[0].shape),
                        dtype=output[0].dtype,
                        value=output[0].detach().cpu(),
                        max_val=output[0].max().item(),
                        min_val=output[0].min().item(),
                        mean_val=output[0].float().mean().item(),
                    )
                )

        return hook

    hooks = []
    for name, module in model.named_modules():
        if layer_pattern and layer_pattern not in name:
            continue
        hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(*inputs, **kwargs)

    for h in hooks:
        h.remove()

    return captures


def compare_intermediates(
    captures1: list[IntermediateCapture],
    captures2: list[IntermediateCapture],
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> list[dict]:
    """Compare intermediate values between two runs.

    Args:
        captures1: First set of captures
        captures2: Second set of captures
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        List of dicts with comparison results for each layer
    """
    results = []

    # Match by name
    caps2_by_name = {c.name: c for c in captures2}

    for c1 in captures1:
        c2 = caps2_by_name.get(c1.name)

        if c2 is None:
            results.append({
                "name": c1.name,
                "match": False,
                "error": "Not found in second capture",
            })
            continue

        if c1.shape != c2.shape:
            results.append({
                "name": c1.name,
                "match": False,
                "error": f"Shape mismatch: {c1.shape} vs {c2.shape}",
            })
            continue

        # Compare values
        match = torch.allclose(c1.value, c2.value, rtol=rtol, atol=atol)
        max_diff = (c1.value - c2.value).abs().max().item()

        results.append({
            "name": c1.name,
            "match": match,
            "max_diff": max_diff,
            "shape": c1.shape,
        })

    return results


def print_trace_summary(trace: TraceResult, top_n: int = 15):
    """Print a summary of a trace."""
    print(f"Total nodes: {len(trace.nodes)}")
    print(f"Call nodes: {len(trace.get_ops())}")
    print(f"Inputs: {len(trace.inputs)}")
    print(f"\nTop {top_n} ops:")
    for op, count in trace.op_counts().most_common(top_n):
        print(f"  {op}: {count}")


def print_comparison_results(results: list[dict], show_passing: bool = False):
    """Print intermediate comparison results."""
    failures = [r for r in results if not r["match"]]
    passes = [r for r in results if r["match"]]

    print(f"Results: {len(passes)} passed, {len(failures)} failed")

    if failures:
        print("\nFailures:")
        for r in failures:
            if "error" in r:
                print(f"  {r['name']}: {r['error']}")
            else:
                print(f"  {r['name']}: max_diff={r['max_diff']:.2e}")

    if show_passing and passes:
        print("\nPassing:")
        for r in passes:
            print(f"  {r['name']}: max_diff={r.get('max_diff', 0):.2e}")
