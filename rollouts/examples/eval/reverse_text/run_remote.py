#!/usr/bin/env python3
"""Run reverse text eval on a remote GPU with SGLang.

This script:
1. Provisions a GPU (Modal or RunPod)
2. Launches SGLang server with Qwen2.5-7B-Instruct
3. Runs the eval against the SGLang endpoint
4. Prints results and cleans up

Usage:
    # Modal (faster cold start ~30s)
    python examples/eval/reverse_text/run_remote.py --provider modal

    # RunPod
    python examples/eval/reverse_text/run_remote.py --provider runpod

    # With different model
    python examples/eval/reverse_text/run_remote.py --model Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add rollouts to path
REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


async def run_on_modal(model: str, limit: int) -> dict:
    """Run eval on Modal sandbox."""
    import modal

    # Define Modal app
    app = modal.App("reverse-text-eval")

    # Create image with SGLang
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "sglang[all]",
            "torch",
            "transformers",
            "httpx",
            "trio",
        )
        .env({"HF_HOME": "/cache/huggingface"})
    )

    # Run function
    @app.function(
        image=image,
        gpu="A10G",
        timeout=600,
        volumes={"/cache": modal.Volume.from_name("hf-cache", create_if_missing=True)},
    )
    def run_eval_on_gpu(model_name: str, num_samples: int) -> dict:
        import subprocess
        import time

        import httpx

        # Start SGLang server in background
        print(f"Starting SGLang server with {model_name}...")
        server_proc = subprocess.Popen(
            [
                "python",
                "-m",
                "sglang.launch_server",
                "--model",
                model_name,
                "--port",
                "30000",
                "--mem-fraction-static",
                "0.85",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Wait for server to be ready
        base_url = "http://localhost:30000"
        for i in range(120):  # 2 minutes timeout
            try:
                resp = httpx.get(f"{base_url}/health", timeout=2)
                if resp.status_code == 200:
                    print(f"SGLang server ready after {i}s")
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            server_proc.kill()
            raise TimeoutError("SGLang server failed to start")

        # Run evaluation
        print(f"Running eval with {num_samples} samples...")

        import re
        from dataclasses import replace
        from difflib import SequenceMatcher

        import trio

        from rollouts.agents import AgentState
        from rollouts.agents import RunConfig as AgentRunConfig
        from rollouts.agents.handlers import handle_stop_max_turns
        from rollouts.core import Endpoint, EvalConfig, Message, Metric, Score, StopReason
        from rollouts.eval import evaluate
        from rollouts.training.scoring import FunctionScorer
        from rollouts.training.types import AttemptResult

        # Tasks
        tasks = [
            {"text": "hello"},
            {"text": "world"},
            {"text": "python"},
            {"text": "machine learning"},
            {"text": "artificial intelligence"},
            {"text": "deep neural network"},
            {"text": "transformer architecture"},
            {"text": "attention mechanism"},
            {"text": "gradient descent"},
            {"text": "backpropagation"},
        ][:num_samples]

        def prepare_messages(sample: dict) -> list[Message]:
            text = sample["text"]
            return [
                Message(
                    role="user",
                    content=f"Reverse the following text character-by-character. "
                    f"Put your answer in <reversed_text> tags.\n\n"
                    f"Text to reverse: {text}",
                ),
            ]

        def score_fn(sample: AttemptResult, _context: object) -> Score:
            input_data = sample.input
            expected = input_data["text"][::-1]

            response = ""
            if sample.trajectory:
                for msg in reversed(sample.trajectory.messages):
                    if msg.role == "assistant":
                        content = msg.content
                        if isinstance(content, str):
                            response = content
                        elif isinstance(content, list) and content:
                            response = (
                                content[0].text if hasattr(content[0], "text") else str(content[0])
                            )
                        break

            match = re.search(r"<reversed_text>\s*(.*?)\s*</reversed_text>", response, re.DOTALL)
            parsed = match.group(1).strip() if match else response.strip()
            exact_match = parsed == expected
            similarity = SequenceMatcher(None, parsed, expected).ratio()

            return Score(
                metrics=(
                    Metric("exact_match", 1.0 if exact_match else 0.0, weight=1.0),
                    Metric("similarity", similarity, weight=0.0),
                )
            )

        endpoint = Endpoint(
            model=f"sglang/{model_name}",
            base_url=f"{base_url}/v1",
            api_format="openai-chat",
            temperature=0.0,
            max_tokens=256,
        )

        async def silent_on_chunk(_: object) -> None:
            pass

        async def stop_on_no_tool(state: AgentState, _: object) -> AgentState:
            return replace(state, stop=StopReason.TASK_COMPLETED)

        agent_run_config = AgentRunConfig(
            on_chunk=silent_on_chunk,
            handle_stop=handle_stop_max_turns(1),
            handle_no_tool=stop_on_no_tool,
        )

        eval_config = EvalConfig(
            endpoint=endpoint,
            scorer=FunctionScorer(score_fn),
            prepare_messages=prepare_messages,
            run_config=agent_run_config,
            max_samples=len(tasks),
            max_concurrent=5,
            verbose=True,
            show_progress=False,
        )

        async def _run() -> dict:
            report = await evaluate(iter(tasks), eval_config)
            return {
                "total": report.total_samples,
                **report.summary_metrics,
            }

        results = trio.run(_run)

        # Cleanup
        server_proc.terminate()
        server_proc.wait(timeout=10)

        return results

    # Run on Modal
    with app.run():
        return run_eval_on_gpu.remote(model, limit)


async def run_on_runpod(model: str, limit: int) -> dict:
    """Run eval on RunPod via bifrost."""
    from bifrost import GPUQuery, ProcessSpec, acquire_node

    print("Provisioning A100 on RunPod...")

    bifrost, instance = await acquire_node(
        provision=GPUQuery(
            type="A100",
            count=1,
            min_cuda="12.0",
            name="reverse-text-eval",
            provider="runpod",
        )
    )

    try:
        node_id = f"{instance.provider}:{instance.id}"
        print(f"Provisioned: {node_id}")

        # Deploy code
        print("Deploying code...")
        workspace = bifrost.push("~/.bifrost/workspaces/rollouts-eval", allow_dirty=True)

        # Install dependencies
        print("Installing dependencies...")
        bifrost.exec(
            "curl -LsSf https://astral.sh/uv/install.sh | sh && "
            "~/.local/bin/uv python install 3.12 && "
            "~/.local/bin/uv pip install sglang[all] torch transformers httpx trio",
            working_dir=workspace,
        )

        # Start SGLang server
        print(f"Starting SGLang with {model}...")
        bifrost.submit(
            ProcessSpec(
                command="python",
                args=("-m", "sglang.launch_server", "--model", model, "--port", "30000"),
                cwd=workspace,
            ),
            name="sglang",
        )

        # Wait for server
        import time

        for i in range(120):
            result = bifrost.exec("curl -s http://localhost:30000/health || echo 'not ready'")
            if "not ready" not in result:
                print(f"SGLang ready after {i}s")
                break
            time.sleep(1)
        else:
            raise TimeoutError("SGLang failed to start")

        # Run eval script remotely
        # For simplicity, just return a placeholder - full implementation would
        # run the eval on the remote node
        print("Running eval...")
        # ... (would execute eval on remote)

        return {"status": "completed", "note": "Full remote eval not yet implemented"}

    finally:
        print("Terminating instance...")
        await instance.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reverse text eval on remote GPU")
    parser.add_argument("--provider", choices=["modal", "runpod"], default="modal")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    import trio

    if args.provider == "modal":
        results = trio.run(run_on_modal, args.model, args.limit)
    else:
        results = trio.run(run_on_runpod, args.model, args.limit)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
