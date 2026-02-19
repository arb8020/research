"""lm-eval-harness integration for REAP evaluation.

Provides comprehensive evaluation using the official lm-eval-harness
with support for:
- Standard lm-eval tasks (winogrande, arc, hellaswag, mmlu, etc.)
- Custom task suites
- Perplexity evaluation
- Code evaluation (evalplus, humaneval, mbpp)
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .export import apply_pruning_recipe, load_pruning_recipe

logger = logging.getLogger(__name__)


# Default task suites matching Cerebras REAP
DEFAULT_TASKS = [
    "winogrande",
    "arc_challenge",
    "arc_easy",
    "boolq",
    "hellaswag",
    "mmlu",
    "openbookqa",
    "rte",
    "gsm8k",  # Grade school math - now included by default
]

# Comprehensive task suite with all major benchmarks
COMPREHENSIVE_TASKS = [
    # Reasoning
    "winogrande",
    "arc_challenge",
    "arc_easy",
    "boolq",
    "hellaswag",
    "openbookqa",
    "rte",
    # Knowledge
    "mmlu",
    # Math
    "gsm8k",
    # Additional useful tasks
    "piqa",
    "siqa",
    "copa",
]

CODE_TASKS = ["mbpp", "humaneval"]

# Math tasks - gsm8k via lm-eval, others via custom or evalscope
MATH_TASKS = [
    "gsm8k",  # Grade school math (via lm-eval)
    "math_qa",  # Math QA dataset
]

# Extended math suite (requires additional setup)
EXTENDED_MATH_TASKS = [
    "gsm8k",
    "hendrycks_math_algebra",
    "hendrycks_math_counting_and_probability",
    "hendrycks_math_geometry",
    "hendrycks_math_intermediate_algebra",
    "hendrycks_math_number_theory",
    "hendrycks_math_prealgebra",
    "hendrycks_math_precalculus",
]


class SGLangServer:
    """Manages SGLang server for evaluation."""

    def __init__(
        self,
        model_path: str | Path,
        port: int = 30000,
        mem_fraction: float = 0.85,
    ):
        self.model_path = Path(model_path)
        self.port = port
        self.mem_fraction = mem_fraction
        self.process: subprocess.Popen | None = None
        self.base_url = f"http://localhost:{port}"

    def start(self) -> None:
        """Start the SGLang server."""
        import sys

        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(self.model_path),
            "--port",
            str(self.port),
            "--trust-remote-code",
            "--mem-fraction-static",
            str(self.mem_fraction),
        ]

        logger.info(f"Starting SGLang server on port {self.port}")
        logger.info(f"Command: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Wait for server to be ready
        self._wait_for_ready()

    def _wait_for_ready(self, timeout: int = 180) -> None:
        """Wait for server to be ready."""
        logger.info(f"Waiting for server at {self.base_url}...")

        for attempt in range(timeout):
            try:
                resp = requests.get(f"{self.base_url}/health", timeout=2)
                if resp.status_code == 200:
                    logger.info(f"Server ready after {attempt}s")
                    return
            except requests.RequestException:
                pass

            # Check if process died
            if self.process and self.process.poll() is not None:
                stdout = self.process.stdout.read().decode() if self.process.stdout else ""
                logger.error(f"Server process died. Output:\n{stdout[-2000:]}")
                raise RuntimeError("SGLang server process died during startup")

            time.sleep(1)

        self.stop()
        raise TimeoutError(f"Server failed to start within {timeout} seconds")

    def stop(self) -> None:
        """Stop the server."""
        if self.process:
            logger.info("Stopping SGLang server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def __enter__(self) -> SGLangServer:
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


def run_lm_eval(
    model_path: str | Path,
    tasks: list[str] | None = None,
    batch_size: int = 8,
    port: int = 30000,
    use_server: bool = True,
) -> dict[str, Any]:
    """Run lm-eval-harness on a model.

    Args:
        model_path: Path to model or HuggingFace model ID
        tasks: List of task names (defaults to DEFAULT_TASKS)
        batch_size: Batch size for evaluation
        port: Port for SGLang server (if use_server=True)
        use_server: Whether to use SGLang server or HF backend

    Returns:
        Dict with evaluation results
    """
    try:
        import lm_eval
    except ImportError:
        raise ImportError("lm-eval-harness not installed. Install with: pip install lm-eval")

    tasks = tasks or DEFAULT_TASKS
    model_path = Path(model_path)

    if use_server:
        # Start server and evaluate
        with SGLangServer(model_path, port=port):
            results = lm_eval.simple_evaluate(
                model="local-completions",
                model_args={
                    "pretrained": str(model_path),
                    "base_url": f"http://localhost:{port}/v1/completions",
                    "tokenizer": str(model_path),
                    "tokenized_requests": False,
                },
                tasks=tasks,
                batch_size=batch_size,
            )
    else:
        # Use HF backend directly
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args={
                "pretrained": str(model_path),
                "dtype": "bfloat16",
                "trust_remote_code": True,
            },
            tasks=tasks,
            batch_size=batch_size,
        )

    # Extract summary
    summary = {}
    if "results" in results:
        for task, metrics in results["results"].items():
            summary[task] = {
                k: v
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and not k.startswith("_")
            }

    return {
        "results": summary,
        "raw_results": results,
    }


def run_code_eval(
    model_path: str | Path,
    tasks: list[str] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    port: int = 30000,
    parallel: int = 32,
) -> dict[str, Any]:
    """Run code evaluation using evalplus.

    Args:
        model_path: Path to model
        tasks: Code tasks to run (defaults to ["mbpp", "humaneval"])
        temperature: Sampling temperature (0 for greedy)
        max_tokens: Max tokens to generate
        port: Port for SGLang server
        parallel: Number of parallel workers

    Returns:
        Dict with evaluation results
    """
    try:
        from evalplus.evaluate import evaluate
    except ImportError:
        logger.warning("evalplus not installed, skipping code eval")
        return {"error": "evalplus not installed"}

    tasks = tasks or CODE_TASKS
    model_path = Path(model_path)

    results = {}

    with SGLangServer(model_path, port=port):
        for task in tasks:
            logger.info(f"Running {task} evaluation...")

            try:
                task_results = evaluate(
                    model=f"http://localhost:{port}/v1/completions",
                    dataset=task,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    parallel=parallel,
                )
                results[task] = task_results
            except Exception as e:
                logger.error(f"Failed to evaluate {task}: {e}")
                results[task] = {"error": str(e)}

    return results


def run_livecodebench(
    model_path: str | Path,
    tasks: list[str] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    port: int = 30000,
    backend: str = "sglang",
) -> dict[str, Any]:
    """Run LiveCodeBench evaluation.

    LiveCodeBench tests competitive programming capabilities
    with time-bounded coding challenges (Codeforces/LeetCode style).

    Args:
        model_path: Path to model
        tasks: LCB tasks - ["all"] or specific contests like ["lcb_release", "lcb_test"]
        temperature: Sampling temperature (0.0 for greedy)
        max_tokens: Max tokens to generate
        port: Port for API server
        backend: "sglang" or "vllm"

    Returns:
        Dict with evaluation results including pass@1, pass@5 metrics

    Example:
        >>> results = run_livecodebench(
        ...     model_path="Qwen/Qwen3-30B-A3B",
        ...     tasks=["all"],
        ...     temperature=0.0,
        ... )
        >>> print(results["livecodebench"]["pass@1"])
    """
    try:
        from livecodebench.eval import evaluate as lcb_evaluate
    except ImportError:
        logger.error(
            "\n" + "=" * 60 + "\n"
            "livecodebench not installed!\n\n"
            "Install with:\n"
            "  pip install livecodebench\n\n"
            "For the latest version:\n"
            "  pip install git+https://github.com/LiveCodeBench/LiveCodeBench.git\n" + "=" * 60
        )
        return {
            "error": "livecodebench not installed",
            "install": "pip install livecodebench",
            "metrics": {},
        }

    tasks = tasks or ["all"]
    model_path = Path(model_path)

    logger.info(f"Running LiveCodeBench evaluation on {tasks}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Temperature: {temperature}, Max tokens: {max_tokens}")

    results = {}

    try:
        with SGLangServer(model_path, port=port):
            # Construct API endpoint
            base_url = f"http://localhost:{port}"
            api_endpoint = f"{base_url}/v1/completions"

            logger.info(f"Using API endpoint: {api_endpoint}")

            # LiveCodeBench evaluation
            lcb_results = lcb_evaluate(
                model=api_endpoint,
                tasks=tasks,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Extract key metrics
            metrics = {}
            if isinstance(lcb_results, dict):
                metrics = {
                    "pass@1": lcb_results.get("pass@1"),
                    "pass@5": lcb_results.get("pass@5"),
                    "total_problems": lcb_results.get("total"),
                    "solved_problems": lcb_results.get("solved"),
                }

            results = {
                "livecodebench": lcb_results,
                "metrics": metrics,
                "status": "success",
            }

            logger.info("LiveCodeBench Results:")
            for key, value in metrics.items():
                if value is not None:
                    logger.info(f"  {key}: {value}")

    except Exception as e:
        logger.error(f"LiveCodeBench evaluation failed: {e}")
        results = {
            "error": str(e),
            "metrics": {},
            "status": "failed",
        }

    return results


def run_math_eval(
    model_path: str | Path,
    tasks: list[str] | None = None,
    port: int = 30000,
    use_server: bool = True,
) -> dict[str, Any]:
    """Run math evaluation using lm-eval.

    Args:
        model_path: Path to model
        tasks: Math tasks (defaults to MATH_TASKS)
        port: Port for SGLang server
        use_server: Whether to use SGLang server

    Returns:
        Dict with evaluation results
    """
    tasks = tasks or MATH_TASKS

    logger.info(f"Running math evaluation on tasks: {tasks}")

    # Use lm-eval for math tasks
    return run_lm_eval(
        model_path=model_path,
        tasks=tasks,
        use_server=use_server,
        port=port,
    )


def run_wildbench(
    model_path: str | Path,
    original_model_name: str | None = None,
    subset: str = "v2",
    port: int = 30000,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run WildBench evaluation using HELM framework.

    WildBench tests real-world capabilities with practical tasks.
    Requires HELM framework and specific configuration.

    Args:
        model_path: Path to model
        original_model_name: Original model name for HELM registry (e.g., "openai/gpt-4")
        subset: WildBench subset ("v2" for latest)
        port: Port for API server
        output_path: Where to save results

    Returns:
        Dict with evaluation results

    Example:
        >>> results = run_wildbench(
        ...     model_path="path/to/pruned_model",
        ...     original_model_name="Qwen/Qwen3-30B-A3B",
        ...     subset="v2",
        ... )

    Setup Required:
        1. Install HELM:
           pip install crfm-helm

        2. Create WildBench config directory:
           mkdir -p config/wildbench_prod_env_{port}

        3. Add model_deployments.yaml:
           model_deployments:
             - name: your-model
               model_name: your-model
               tokenizer_name: your-tokenizer
               max_sequence_length: 32768
               client_spec:
                 class_name: helm.clients.openai_client.OpenAIClient
                 args:
                   base_url: http://localhost:{port}/v1

        4. Add credentials.conf:
           openai_api_key=not_needed_for_local

    Note:
        This is a complex integration requiring HELM framework setup.
        See: https://crfm-helm.readthedocs.io/
    """
    try:
        from helm.benchmark.run import create_helm_run_args, helm_run
        from helm.common.hierarchical_logger import setup_default_logging
    except ImportError:
        logger.error(
            "\n" + "=" * 60 + "\n"
            "HELM framework not installed!\n\n"
            "WildBench requires HELM which is a complex framework.\n\n"
            "To install:\n"
            "  pip install crfm-helm\n\n"
            "However, you'll also need to:\n"
            "  1. Create config files for model deployment\n"
            "  2. Set up HELM run environment\n"
            "  3. Configure model mappings\n\n"
            "For simpler evaluation, use:\n"
            "  - lm-eval tasks (standard benchmarks)\n"
            "  - evalplus (code evaluation)\n"
            "  - livecodebench (competitive programming)\n" + "=" * 60
        )
        return {
            "error": "HELM not installed",
            "install": "pip install crfm-helm",
            "note": "Complex setup required - see docstring",
            "alternatives": ["lm-eval", "evalplus", "livecodebench"],
        }

    model_path = Path(model_path)
    output_path = output_path or model_path.parent / "wildbench_results"
    output_path.mkdir(parents=True, exist_ok=True)

    # Use model name from path if not provided
    if original_model_name is None:
        original_model_name = model_path.name

    logger.info("Running WildBench evaluation")
    logger.info(f"Model: {original_model_name}")
    logger.info(f"Subset: {subset}")
    logger.info(f"Output: {output_path}")

    try:
        with SGLangServer(model_path, port=port):
            # HELM requires specific config setup
            # This is a simplified version - full implementation needs config files

            suite = "wildbench_eval"
            run_entries = [f"wildbench:subset={subset},model={original_model_name}"]

            # Note: This requires proper HELM config setup
            # See Cerebras REAP for full implementation reference

            logger.warning(
                "WildBench requires HELM configuration files. This is a stub implementation."
            )

            return {
                "status": "stub",
                "message": "WildBench stub - HELM config required",
                "run_entries": run_entries,
                "suite": suite,
                "note": "Full implementation requires config/wildbench_prod_env/ setup",
            }

    except Exception as e:
        logger.error(f"WildBench evaluation failed: {e}")
        return {
            "error": str(e),
            "status": "failed",
        }


def evaluate_pruned_model(
    recipe_path: Path,
    tasks: list[str] | None = None,
    code_tasks: list[str] | None = None,
    run_code_eval: bool = False,
    run_livecodebench: bool = False,
    livecodebench_tasks: list[str] | None = None,
    output_dir: Path | None = None,
    use_server: bool = True,
    port: int = 30000,
) -> dict[str, Any]:
    """Evaluate a pruned model from a recipe.

    This loads the base model, applies the pruning recipe, saves the pruned model,
    and runs evaluation.

    Args:
        recipe_path: Path to pruning_recipe.json
        tasks: lm-eval tasks (defaults to DEFAULT_TASKS)
        code_tasks: Code evaluation tasks
        run_code_eval: Whether to run code evaluation
        output_dir: Where to save results (defaults to recipe directory)
        use_server: Whether to use SGLang server
        port: Port for SGLang server

    Returns:
        Dict with all evaluation results
    """
    recipe = load_pruning_recipe(recipe_path)
    base_model = recipe["base_model"]
    experts_to_keep = recipe["experts_to_keep"]

    output_dir = output_dir or recipe_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Evaluating pruned model: {base_model}")
    logger.info(f"Compression ratio: {recipe.get('compression_ratio', 'unknown')}")

    # Load and prune model
    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    logger.info("Applying pruning recipe...")
    apply_pruning_recipe(model, experts_to_keep)

    # Save pruned model temporarily
    pruned_dir = output_dir / "_temp_pruned_model"
    logger.info(f"Saving pruned model to {pruned_dir}")
    model.save_pretrained(pruned_dir)
    tokenizer.save_pretrained(pruned_dir)

    # Clean up memory
    del model
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    # Run evaluations
    all_results = {
        "recipe": recipe,
        "evaluations": {},
    }

    # lm-eval
    logger.info("Running lm-eval...")
    try:
        lm_results = run_lm_eval(
            pruned_dir,
            tasks=tasks,
            use_server=use_server,
            port=port,
        )
        all_results["evaluations"]["lm_eval"] = lm_results["results"]
    except Exception as e:
        logger.error(f"lm-eval failed: {e}")
        all_results["evaluations"]["lm_eval"] = {"error": str(e)}

    # Code eval
    if run_code_eval and code_tasks:
        logger.info("Running code evaluation...")
        try:
            code_results = run_code_eval(
                pruned_dir,
                tasks=code_tasks,
                port=port + 1,  # Use different port
            )
            all_results["evaluations"]["code_eval"] = code_results
        except Exception as e:
            logger.error(f"Code eval failed: {e}")
            all_results["evaluations"]["code_eval"] = {"error": str(e)}

    # LiveCodeBench eval
    if run_livecodebench:
        logger.info("Running LiveCodeBench evaluation...")
        try:
            lcb_results = run_livecodebench(
                pruned_dir,
                tasks=livecodebench_tasks,
                port=port + 2,  # Use different port
            )
            all_results["evaluations"]["livecodebench"] = lcb_results
        except Exception as e:
            logger.error(f"LiveCodeBench failed: {e}")
            all_results["evaluations"]["livecodebench"] = {"error": str(e)}

    # Save results
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Cleanup temp model
    import shutil

    shutil.rmtree(pruned_dir, ignore_errors=True)

    return all_results


def compare_models(
    base_model: str,
    pruned_recipe: Path,
    tasks: list[str] | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Compare base model vs pruned model on the same tasks.

    Args:
        base_model: HuggingFace model ID
        pruned_recipe: Path to pruning recipe
        tasks: Tasks to evaluate
        output_path: Where to save comparison results

    Returns:
        Dict with comparison results
    """
    tasks = tasks or DEFAULT_TASKS

    logger.info("=" * 60)
    logger.info("COMPARING MODELS")
    logger.info("=" * 60)

    # Evaluate base model
    logger.info(f"\nEvaluating BASE model: {base_model}")
    base_results = run_lm_eval(base_model, tasks=tasks)

    # Evaluate pruned model
    logger.info(f"\nEvaluating PRUNED model from {pruned_recipe}")
    pruned_results = evaluate_pruned_model(
        pruned_recipe,
        tasks=tasks,
        output_dir=pruned_recipe.parent,
    )

    # Compare
    comparison = {
        "base_model": base_model,
        "pruned_recipe": str(pruned_recipe),
        "tasks": tasks,
        "results": {},
    }

    for task in tasks:
        base_metrics = base_results.get("results", {}).get(task, {})
        pruned_metrics = pruned_results.get("evaluations", {}).get("lm_eval", {}).get(task, {})

        task_comparison = {}
        for metric in set(base_metrics.keys()) | set(pruned_metrics.keys()):
            base_val = base_metrics.get(metric)
            pruned_val = pruned_metrics.get(metric)

            if base_val is not None and pruned_val is not None:
                change = ((pruned_val - base_val) / base_val * 100) if base_val != 0 else 0
                task_comparison[metric] = {
                    "base": base_val,
                    "pruned": pruned_val,
                    "change_pct": change,
                }

        comparison["results"][task] = task_comparison

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)

    for task, metrics in comparison["results"].items():
        logger.info(f"\n{task}:")
        for metric, vals in metrics.items():
            change = vals.get("change_pct", 0)
            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
            logger.info(
                f"  {metric}: {vals['base']:.4f} → {vals['pruned']:.4f} ({arrow}{change:+.2f}%)"
            )

    # Save
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"\nComparison saved to {output_path}")

    return comparison


def main():
    """CLI entry point."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Evaluate pruned models with lm-eval")
    parser.add_argument("--recipe", type=Path, help="Path to pruning_recipe.json")
    parser.add_argument("--model", type=str, help="HuggingFace model ID (for non-recipe eval)")
    parser.add_argument("--tasks", nargs="+", default=None, help="Tasks to evaluate")
    parser.add_argument("--code-tasks", nargs="+", default=None, help="Code evaluation tasks")
    parser.add_argument(
        "--run-code-eval", action="store_true", help="Run code evaluation (evalplus)"
    )
    parser.add_argument(
        "--run-livecodebench", action="store_true", help="Run LiveCodeBench evaluation"
    )
    parser.add_argument(
        "--livecodebench-tasks", nargs="+", default=None, help="LiveCodeBench tasks (default: all)"
    )
    parser.add_argument("--compare", type=str, help="Base model to compare against (HF ID)")
    parser.add_argument("--output", type=Path, help="Output path for results")
    parser.add_argument("--port", type=int, default=30000, help="SGLang server port")
    parser.add_argument("--no-server", action="store_true", help="Use HF backend instead of SGLang")

    args = parser.parse_args()

    if args.compare and args.recipe:
        # Comparison mode
        results = compare_models(
            base_model=args.compare,
            pruned_recipe=args.recipe,
            tasks=args.tasks,
            output_path=args.output,
        )
    elif args.recipe:
        # Single recipe evaluation
        results = evaluate_pruned_model(
            recipe_path=args.recipe,
            tasks=args.tasks,
            code_tasks=args.code_tasks,
            run_code_eval=args.run_code_eval,
            run_livecodebench=args.run_livecodebench,
            livecodebench_tasks=args.livecodebench_tasks,
            output_dir=args.recipe.parent,
            use_server=not args.no_server,
            port=args.port,
        )
        # Print summary
        if "lm_eval" in results.get("evaluations", {}):
            logger.info("\nlm-eval Results:")
            for task, metrics in results["evaluations"]["lm_eval"].items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            logger.info(f"  {task}/{metric}: {value:.4f}")
    elif args.model:
        # Direct model evaluation
        results = run_lm_eval(
            args.model,
            tasks=args.tasks,
            use_server=not args.no_server,
            port=args.port,
        )
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")
    else:
        parser.error("Either --recipe, --model, or both --recipe and --compare are required")


if __name__ == "__main__":
    main()
