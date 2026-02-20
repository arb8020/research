"""Evaluate multi-turn KernelBench with SGLang endpoint.

This script:
1. Starts SGLang server with a specified model
2. Waits for it to be ready
3. Runs multi-turn evaluation
4. Shuts down SGLang server

Usage:
    # Evaluate with a small model (fast, for testing)
    python eval_with_sglang.py --model Nanbeige/Nanbeige4.1-3B --num-problems 3

    # Evaluate with Qwen Coder
    python eval_with_sglang.py --model Qwen/Qwen2.5-Coder-3B-Instruct --num-problems 5 --max-turns 4

    # Use existing SGLang server (don't start/stop)
    python eval_with_sglang.py --endpoint http://localhost:30000/v1 --num-problems 5

Requirements:
    - SGLang installed: pip install sglang[all]
    - CUDA GPU available
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
import urllib.request


def wait_for_server(url: str, timeout: float = 120.0, interval: float = 1.0) -> bool:
    """Wait for server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"{url}/models", timeout=5)
            return True
        except Exception:
            time.sleep(interval)
    return False


def start_sglang_server(
    model: str,
    port: int = 30000,
    tp_size: int = 1,
    additional_args: list[str] | None = None,
) -> subprocess.Popen:
    """Start SGLang server.

    Args:
        model: HuggingFace model name
        port: Server port
        tp_size: Tensor parallelism size
        additional_args: Additional arguments for sglang launch

    Returns:
        Server process
    """
    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model",
        model,
        "--port",
        str(port),
        "--tp",
        str(tp_size),
        "--host",
        "0.0.0.0",
    ]

    if additional_args:
        cmd.extend(additional_args)

    print("Starting SGLang server...")
    print(f"  Model: {model}")
    print(f"  Port: {port}")
    print(f"  TP size: {tp_size}")
    print(f"  Command: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for server to be ready
    url = f"http://localhost:{port}/v1"
    print(f"\nWaiting for server at {url}...")

    if wait_for_server(url, timeout=120.0):
        print("✓ Server is ready!")
        return process
    else:
        process.terminate()
        raise RuntimeError("Server failed to start within 120 seconds")


def stop_sglang_server(process: subprocess.Popen) -> None:
    """Stop SGLang server."""
    print("\nStopping SGLang server...")
    process.terminate()
    try:
        process.wait(timeout=10)
        print("✓ Server stopped")
    except subprocess.TimeoutExpired:
        process.kill()
        print("✓ Server killed")


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate KernelBench with SGLang",
    )

    # SGLang options
    parser.add_argument(
        "--model",
        type=str,
        default="Nanbeige/Nanbeige4.1-3B",
        help="HuggingFace model name (default: Nanbeige/Nanbeige4.1-3B)",
    )
    parser.add_argument(
        "--port", type=int, default=30000, help="SGLang server port (default: 30000)"
    )
    parser.add_argument(
        "--tp-size", type=int, default=1, help="Tensor parallelism size (default: 1)"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="Use existing SGLang endpoint (don't start server)",
    )

    # Evaluation options
    parser.add_argument(
        "--num-problems", type=int, default=5, help="Number of problems (default: 5)"
    )
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[1], help="KernelBench levels (default: [1])"
    )
    parser.add_argument(
        "--max-turns", type=int, default=8, help="Max turns per problem (default: 8)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cuda",
        choices=["cuda", "hip"],
        help="Kernel backend (default: cuda)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output file for results")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")

    args = parser.parse_args()

    # Import here to avoid issues if dependencies not installed
    from eval_multi_turn import evaluate_multi_turn

    server_process = None
    endpoint_url = args.endpoint

    try:
        # Start SGLang if no endpoint provided
        if endpoint_url is None:
            server_process = start_sglang_server(
                model=args.model,
                port=args.port,
                tp_size=args.tp_size,
            )
            endpoint_url = f"http://localhost:{args.port}/v1"
        else:
            print(f"Using existing SGLang endpoint: {endpoint_url}")

        # Run evaluation
        print(f"\n{'=' * 60}")
        print("Starting evaluation...")
        print(f"{'=' * 60}")

        summary = await evaluate_multi_turn(
            model=args.model,
            provider="sglang",
            endpoint_url=endpoint_url,
            num_problems=args.num_problems,
            levels=args.levels,
            max_turns=args.max_turns,
            backend=args.backend,
            output_file=args.output,
            verbose=not args.quiet,
        )

        # Print final results
        print(f"\n{'=' * 60}")
        print("FINAL RESULTS")
        print(f"{'=' * 60}")
        print(f"Model: {args.model}")
        print(f"Problems: {summary['total']}")
        print(
            f"Correct: {summary['correct']}/{summary['successful']} ({summary['correct_rate']:.1%})"
        )
        print(f"Avg speedup: {summary['avg_speedup']:.2f}x")
        print(f"Max speedup: {summary['max_speedup']:.2f}x")
        print(f"Avg turns: {summary['avg_turns']:.1f}")

        if summary.get("correct", 0) > 0:
            print("\n✓ Evaluation successful!")
            return 0
        else:
            print("\n⚠ No correct kernels - environment needs debugging")
            return 1

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        # Stop SGLang if we started it
        if server_process is not None:
            stop_sglang_server(server_process)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
