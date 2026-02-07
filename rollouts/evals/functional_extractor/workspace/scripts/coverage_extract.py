#!/usr/bin/env python3
"""Extract forward pass code with coverage annotations.

Runs a forward pass with coverage.py instrumentation and outputs
annotated source showing which lines/branches were hit vs missed.

Usage:
    # Default: simple forward pass
    python coverage_extract.py HuggingFaceTB/SmolLM2-135M

    # Custom test script for specific mode (kvcache, training, etc)
    python coverage_extract.py HuggingFaceTB/SmolLM2-135M --test-script test_kvcache.py

    # Output formats
    python coverage_extract.py HuggingFaceTB/SmolLM2-135M --format annotated
    python coverage_extract.py HuggingFaceTB/SmolLM2-135M -o coverage.json

Test scripts should define a run(model, tokenizer) function that exercises
the model in the desired mode. Example for kvcache:

    def run(model, tokenizer):
        import torch
        input_ids = torch.tensor([[1, 2, 3, 4]])
        out = model(input_ids, use_cache=True)
        # Decode step
        next_token = out.logits[:, -1:].argmax(dim=-1)
        model(next_token, past_key_values=out.past_key_values, use_cache=True)

Output formats:
    json: Structured data for LLM consumption
    annotated: Source code with HIT/MISS/BRANCH_MISS markers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def run_coverage(model_name: str, test_inputs: list[list[int]] | None = None, test_script: str | None = None):
    """Run forward pass with coverage instrumentation.

    Args:
        model_name: HuggingFace model name
        test_inputs: Simple list of input sequences (used if no test_script)
        test_script: Path to a Python script with run(model, tokenizer) function

    Returns:
        Tuple of (modeling_file_path, lines_hit, missing_branches, source_lines)
    """
    import coverage
    import torch

    if test_inputs is None:
        test_inputs = [[1, 2, 3, 4]]

    # Determine which modeling file to cover based on model architecture
    # We'll discover this dynamically by loading the model first
    cov = coverage.Coverage(branch=True, include=['**/modeling_*.py'])
    cov.start()

    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    # Load model
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    ).cpu()
    model.eval()

    if test_script:
        # Load and run custom test script
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_script", test_script)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load test script: {test_script}")
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)

        if not hasattr(test_module, 'run'):
            raise ValueError(f"Test script {test_script} must define run(model, tokenizer)")

        # Load tokenizer for test script
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            tokenizer = None  # Some models don't have tokenizers

        # Run the test script
        test_module.run(model, tokenizer)
    else:
        # Default: simple forward passes
        for input_seq in test_inputs:
            input_ids = torch.tensor([input_seq])
            with torch.no_grad():
                _ = model(input_ids)

    cov.stop()
    cov.save()

    # Find the primary modeling file (the one with most hits)
    data = cov.get_data()
    files = list(data.measured_files())

    if not files:
        raise RuntimeError("No files covered - something went wrong")

    # Find the main modeling file (usually the one matching the model type)
    model_type = config.model_type.lower()
    main_file = None
    for f in files:
        fname = Path(f).name.lower()
        if f'modeling_{model_type}' in fname or model_type in fname:
            main_file = f
            break

    if main_file is None:
        # Fall back to file with most coverage
        main_file = max(files, key=lambda f: len(data.lines(f) or []))

    # Get coverage data
    lines_hit = set(data.lines(main_file) or [])
    arcs = set(data.arcs(main_file) or [])

    # Get missing branches
    analysis = cov._analyze(main_file)
    missing_branches = analysis.missing_branch_arcs()  # dict: from_line -> [to_lines]

    # Read source
    with open(main_file) as f:
        source_lines = f.readlines()

    return main_file, lines_hit, missing_branches, source_lines, arcs


def extract_functions(source_lines: list[str], lines_hit: set[int]) -> list[dict]:
    """Extract function definitions and their coverage status."""
    import re

    functions = []
    current_func = None
    current_indent = 0

    func_pattern = re.compile(r'^(\s*)def\s+(\w+)\s*\(')
    class_pattern = re.compile(r'^(\s*)class\s+(\w+)')

    current_class = None

    for i, line in enumerate(source_lines, 1):
        # Track class context
        class_match = class_pattern.match(line)
        if class_match:
            current_class = class_match.group(2)
            continue

        # Track function definitions
        func_match = func_pattern.match(line)
        if func_match:
            if current_func:
                functions.append(current_func)

            indent = len(func_match.group(1))
            func_name = func_match.group(2)

            # Reset class tracking if this is a top-level function
            if indent == 0:
                current_class = None

            current_func = {
                'name': func_name,
                'class': current_class,
                'full_name': f"{current_class}.{func_name}" if current_class else func_name,
                'start_line': i,
                'end_line': i,
                'lines_hit': [],
                'lines_missed': [],
                'indent': indent,
            }
            current_indent = indent

        # Track lines belonging to current function
        if current_func:
            line_stripped = line.rstrip()
            if line_stripped:  # Non-empty line
                line_indent = len(line) - len(line.lstrip())
                # Still in function if more indented or same level continuation
                if line_indent > current_indent or (line_indent == current_indent and not func_pattern.match(line)):
                    current_func['end_line'] = i
                    if i in lines_hit:
                        current_func['lines_hit'].append(i)
                    elif not line_stripped.startswith('#') and not line_stripped.startswith('"""'):
                        current_func['lines_missed'].append(i)

    if current_func:
        functions.append(current_func)

    return functions


def generate_json_report(
    model_name: str,
    main_file: str,
    lines_hit: set[int],
    missing_branches: dict[int, list[int]],
    source_lines: list[str],
) -> dict:
    """Generate structured JSON report for LLM consumption."""

    # Extract functions
    functions = extract_functions(source_lines, lines_hit)

    # Find all if/elif statements with branch info
    branches = []
    for i, line in enumerate(source_lines, 1):
        stripped = line.strip()
        if stripped.startswith('if ') or stripped.startswith('elif '):
            hit = i in lines_hit
            missing = missing_branches.get(i, [])
            branches.append({
                'line': i,
                'code': stripped[:100],
                'hit': hit,
                'missing_targets': missing,
                'prunable': len(missing) > 0 and hit,  # Hit but didn't take all branches
            })

    # Identify dead code regions (consecutive missed lines)
    dead_regions = []
    missed_lines = sorted(set(range(1, len(source_lines) + 1)) - lines_hit)

    if missed_lines:
        region_start = missed_lines[0]
        region_end = missed_lines[0]

        for line in missed_lines[1:]:
            if line == region_end + 1:
                region_end = line
            else:
                if region_end - region_start >= 2:  # At least 3 lines
                    dead_regions.append({
                        'start': region_start,
                        'end': region_end,
                        'lines': region_end - region_start + 1,
                        'preview': source_lines[region_start-1].strip()[:60],
                    })
                region_start = line
                region_end = line

        # Don't forget last region
        if region_end - region_start >= 2:
            dead_regions.append({
                'start': region_start,
                'end': region_end,
                'lines': region_end - region_start + 1,
                'preview': source_lines[region_start-1].strip()[:60],
            })

    # Summary stats
    total_lines = len(source_lines)
    executable_lines = sum(1 for line in source_lines if line.strip() and not line.strip().startswith('#'))

    return {
        'model_name': model_name,
        'modeling_file': main_file,
        'summary': {
            'total_lines': total_lines,
            'executable_lines': executable_lines,
            'lines_hit': len(lines_hit),
            'coverage_pct': round(len(lines_hit) / executable_lines * 100, 1) if executable_lines else 0,
            'branches_with_missing': len(missing_branches),
            'dead_regions': len(dead_regions),
        },
        'functions': [f for f in functions if f['lines_hit'] or f['lines_missed']],
        'branches': branches,
        'dead_regions': dead_regions,
        'pruning_candidates': [
            {
                'type': 'branch',
                'line': b['line'],
                'code': b['code'],
                'reason': f"Condition evaluated but branch to {b['missing_targets']} never taken",
            }
            for b in branches if b['prunable']
        ],
    }


def generate_annotated_source(
    source_lines: list[str],
    lines_hit: set[int],
    missing_branches: dict[int, list[int]],
) -> str:
    """Generate annotated source with coverage markers."""
    output = []

    for i, line in enumerate(source_lines, 1):
        stripped = line.rstrip()

        # Determine line status
        if i in lines_hit:
            if i in missing_branches:
                marker = f"BRANCH_MISS->{missing_branches[i]}"
            else:
                marker = "HIT"
        else:
            if stripped and not stripped.startswith('#') and not stripped.startswith('"""'):
                marker = "MISS"
            else:
                marker = "---"  # Comment or blank

        output.append(f"{i:4d} [{marker:20s}] {stripped}")

    return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(description="Extract forward pass with coverage")
    parser.add_argument("model_name", help="HuggingFace model name")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    parser.add_argument("--format", "-f", choices=["json", "annotated"], default="json",
                        help="Output format (default: json)")
    parser.add_argument("--test-inputs", type=str,
                        help="JSON list of test inputs (default: [[1,2,3,4]])")
    parser.add_argument("--test-script", type=str,
                        help="Path to test script with run(model, tokenizer) function")
    args = parser.parse_args()

    test_inputs = json.loads(args.test_inputs) if args.test_inputs else None

    if args.test_script:
        print(f"Running coverage on {args.model_name} with {args.test_script}...", file=sys.stderr)
    else:
        print(f"Running coverage on {args.model_name}...", file=sys.stderr)

    main_file, lines_hit, missing_branches, source_lines, arcs = run_coverage(
        args.model_name, test_inputs, args.test_script
    )

    print(f"Covered {main_file}", file=sys.stderr)
    print(f"Lines hit: {len(lines_hit)}, Missing branches: {len(missing_branches)}", file=sys.stderr)

    if args.format == "json":
        report = generate_json_report(
            args.model_name, main_file, lines_hit, missing_branches, source_lines
        )
        output = json.dumps(report, indent=2)
    else:
        output = generate_annotated_source(source_lines, lines_hit, missing_branches)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
