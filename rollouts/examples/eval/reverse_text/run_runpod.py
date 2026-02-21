#!/usr/bin/env python3
"""Run reverse text eval on RunPod with SGLang.

Uses the same bifrost infrastructure as RL training.

Usage:
    python examples/eval/reverse_text/run_runpod.py
    python examples/eval/reverse_text/run_runpod.py --model Qwen/Qwen2.5-7B-Instruct --limit 10
    python examples/eval/reverse_text/run_runpod.py --gpu-type A10G  # cheaper
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent.parent


async def run_eval_on_runpod(
    model: str,
    limit: int,
    gpu_type: str = "A100",
    keep_alive: bool = False,
) -> dict:
    """Provision RunPod GPU, start SGLang, run eval."""
    from bifrost import GPUQuery, ProcessSpec, acquire_node
    from pytui import Console

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"eval_{timestamp}"

    console = Console()
    console.install_logging_handler(logging.getLogger())

    # Provision GPU
    print(f"Provisioning {gpu_type} on RunPod...")
    bifrost, instance = await acquire_node(
        provision=GPUQuery(
            type=gpu_type,
            count=1,
            min_cuda="12.0",
            name=f"rollouts/{run_name}",
            provider="runpod",
        )
    )

    node_id = f"{instance.provider}:{instance.id}"
    print(f"Provisioned: {node_id}")

    try:
        # Deploy code
        with console.spinner("Deploying code..."):
            workspace = bifrost.push("~/.bifrost/workspaces/rollouts-eval", allow_dirty=True)
        print(f"Deployed to {workspace}")

        # Bootstrap (exact same steps as RL training in rollouts/run.py:215-236)
        bootstrap_steps = [
            (
                "Installing system deps",
                "apt-get update && apt-get install -y tmux libnuma1 || true",
            ),
            (
                "Installing uv",
                "curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env",
            ),
            (
                "Syncing Python deps",
                "~/.local/bin/uv python install 3.12 && ~/.local/bin/uv sync --python 3.12 --package rollouts",
            ),
            (
                "Installing ML packages",
                # sglang 0.5.8 (latest PyPI) is incompatible with transformers>=5.x in two ways:
                #   1. transformers 4.57.1 lacks is_offline_mode (removed in huggingface_hub>=1.4)
                #   2. janus_pro.py calls AutoImageProcessor.register() in a way that broke in transformers 5.x
                # Both are fixed in sglang git main. Install from git, then pin transformers/hf_hub.
                # See: https://github.com/sgl-project/sglang/issues/4159
                "~/.local/bin/uv pip install --upgrade torch datasets accelerate curl_cffi peft"
                " 'sglang[all] @ git+https://github.com/sgl-project/sglang.git@main#subdirectory=python'"
                " && ~/.local/bin/uv pip install --upgrade 'transformers>=5.0.0' 'huggingface_hub>=1.4.0'",
            ),
        ]

        for label, cmd in bootstrap_steps:
            with console.spinner(f"{label}..."):
                bifrost.exec(cmd, working_dir=workspace)
            print(f"✓ {label}")

        # Start SGLang server
        print(f"\nStarting SGLang server with {model}...")
        sglang_job = bifrost.submit(
            ProcessSpec(
                command="/root/.local/bin/uv",
                args=(
                    "run",
                    "python",
                    "-m",
                    "sglang.launch_server",
                    "--model",
                    model,
                    "--port",
                    "30000",
                    "--mem-fraction-static",
                    "0.85",
                    "--trust-remote-code",
                ),
                cwd=f"{workspace}/rollouts",
            ),
            name="sglang",
        )
        print(f"SGLang job started: {sglang_job.tmux_session}")

        # Wait for server to be ready
        import time

        print("Waiting for SGLang to be ready...")
        for i in range(180):  # 3 min timeout
            result = bifrost.exec("curl -s http://localhost:30000/health || echo 'not ready'")
            # result is ExecResult, check stdout
            stdout = result.stdout if hasattr(result, "stdout") else str(result)
            if "not ready" not in stdout:
                print(f"SGLang ready after {i}s")
                break
            time.sleep(1)
        else:
            raise TimeoutError("SGLang failed to start within 3 minutes")

        # Run eval remotely
        print(f"\nRunning eval with {limit} samples...")

        # Create eval script to run remotely
        eval_script = f'''
import re
from difflib import SequenceMatcher
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="dummy")
model = "{model}"

tasks = [
    {{"text": "hello"}},
    {{"text": "world"}},
    {{"text": "python"}},
    {{"text": "machine learning"}},
    {{"text": "artificial intelligence"}},
    {{"text": "deep neural network"}},
    {{"text": "transformer architecture"}},
    {{"text": "attention mechanism"}},
    {{"text": "gradient descent"}},
    {{"text": "backpropagation"}},
][:{limit}]

results = []
for i, task in enumerate(tasks):
    text = task["text"]
    expected = text[::-1]

    prompt = f"Reverse the following text character-by-character. Put your answer in <reversed_text> tags.\\n\\nText to reverse: {{text}}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{{"role": "user", "content": prompt}}],
            temperature=0.0,
            max_tokens=256,
        )
        content = response.choices[0].message.content or ""

        match = re.search(r"<reversed_text>\\s*(.*?)\\s*</reversed_text>", content, re.DOTALL)
        parsed = match.group(1).strip() if match else content.strip()

        exact_match = parsed == expected
        similarity = SequenceMatcher(None, parsed, expected).ratio()

        status = "✓" if exact_match else "✗"
        print(f"[{{i+1}}/{{len(tasks)}}] {{status}} '{{text}}' -> '{{parsed}}' (expected: '{{expected}}')")

        results.append({{"exact_match": exact_match, "similarity": similarity}})
    except Exception as e:
        print(f"[{{i+1}}/{{len(tasks)}}] Error: {{e}}")
        results.append({{"exact_match": False, "similarity": 0.0}})

# Summary
exact = sum(1 for r in results if r["exact_match"])
mean_sim = sum(r["similarity"] for r in results) / len(results)
print(f"\\n=== RESULTS ===")
print(f"Accuracy: {{exact}}/{{len(results)}} ({{100*exact/len(results):.1f}}%)")
print(f"Mean similarity: {{mean_sim:.3f}}")
'''

        # Write and run eval script
        eval_script_path = f"{workspace}/rollouts/_eval_script.py"
        bifrost.exec(f"cat > {eval_script_path} << 'EVALSCRIPT'\n{eval_script}\nEVALSCRIPT")

        result = bifrost.exec(
            f"cd {workspace}/rollouts && /root/.local/bin/uv run python {eval_script_path}",
            timeout=300,
        )
        print(result)

        return {"status": "completed", "output": result}

    finally:
        if not keep_alive:
            print("\nTerminating instance...")
            await instance.terminate()
        else:
            print(f"\nInstance kept alive: {node_id}")
            print(f"Reuse with: --node-id {node_id}")


def main() -> None:
    import trio

    parser = argparse.ArgumentParser(description="Run reverse text eval on RunPod")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--gpu-type", default="A100")
    parser.add_argument("--keep-alive", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("Running reverse text eval on RunPod")
    print(f"  Model: {args.model}")
    print(f"  Samples: {args.limit}")
    print(f"  GPU: {args.gpu_type}")
    print()

    trio.run(
        run_eval_on_runpod,
        args.model,
        args.limit,
        args.gpu_type,
        args.keep_alive,
    )


if __name__ == "__main__":
    main()
