"""GPU verification for functional model extraction.

Uses bifrost/broker to run verification on remote GPU instances.

Usage:
    from tools.functional_extractor.verify import verify_functional, run_on_gpu

    # Verify functional code matches original model
    result = verify_functional(
        model_name="Qwen/Qwen2.5-0.5B",
        functional_code=my_code_string,
        test_inputs=[[1, 2, 3, 4], [5, 6, 7, 8]],
    )
    print(result.matches)  # True if torch.allclose passes
    print(result.max_diff)  # Maximum difference

    # Or run arbitrary code on GPU
    run_on_gpu(
        script_path=__file__,
        keep_alive=True,
        gpu_id="abc123",  # Reuse existing instance
    )
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class VerificationResult:
    """Result of comparing functional code to original model."""

    matches: bool
    max_diff: float
    rtol: float
    atol: float
    original_shape: tuple[int, ...]
    functional_shape: tuple[int, ...]
    error: str | None = None


def run_on_gpu(
    script_path: str,
    keep_alive: bool = False,
    gpu_id: str | None = None,
    vram_gb: int = 24,
    max_price: float = 0.5,
) -> None:
    """Run a script on a remote GPU via broker/bifrost.

    Args:
        script_path: Path to the script (__file__ from caller)
        keep_alive: Keep GPU running after completion
        gpu_id: Reuse existing GPU instance ID (skips provisioning)
        vram_gb: Minimum VRAM required (default 24GB)
        max_price: Maximum price per hour (default $0.50)
    """
    from bifrost.client import BifrostClient
    from broker.client import GPUClient
    from dotenv import load_dotenv

    # Load from parent .env (research/.env)
    env_path = Path(__file__).resolve().parents[3] / ".env"
    load_dotenv(env_path)

    # Get script path relative to git root
    script = Path(script_path).resolve()
    git_root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    )
    rel_path = script.relative_to(git_root)

    # Provision or reuse GPU
    runpod_key = os.getenv("RUNPOD_API_KEY")
    assert runpod_key, f"RUNPOD_API_KEY not set (looked in {env_path})"
    ssh_key_path = os.getenv("SSH_KEY_PATH", "~/.ssh/id_ed25519")

    client = GPUClient(credentials={"runpod": runpod_key}, ssh_key_path=ssh_key_path)
    gpu = None

    try:
        if gpu_id:
            print(f"Reusing GPU: {gpu_id}")
            gpu = client.get_instance(gpu_id, provider="runpod")
            if not gpu:
                print(f"GPU {gpu_id} not found (is it still running?)")
                return
            keep_alive = True  # Always keep alive when reusing
        else:
            print("Provisioning GPU...")
            gpu = client.create(
                query=(
                    (client.vram_gb >= vram_gb) &
                    (client.price_per_hour <= max_price) &
                    (client.manufacturer == 'Nvidia')
                ),
                name=f"verify-{script.stem}",
                cloud_type="secure",  # Required for direct SSH access
                sort=lambda x: x.price_per_hour,
                reverse=False,
            )
            if not gpu:
                print("Failed to provision GPU")
                return
            print(f"GPU ready: {gpu.id}")

            if not gpu.wait_until_ssh_ready(timeout=300):
                print("SSH timeout")
                client.terminate_instance(gpu.id, gpu.provider)
                return

        print(f"SSH: {gpu.ssh_connection_string()}")

        # Deploy code
        workspace = "~/.bifrost/workspaces/rollouts"
        bifrost = BifrostClient(gpu.ssh_connection_string(), ssh_key_path)
        bootstrap = [
            "cd rollouts && uv python install 3.12 && uv sync --python 3.12",
            "uv pip install torch 'huggingface_hub>=0.26.0' datasets accelerate",
            "uv pip install 'git+https://github.com/huggingface/transformers.git'",
        ]
        bifrost.push(workspace_path=workspace, bootstrap_cmd=bootstrap)
        print("Code deployed")

        # Run with streaming output
        remote_script = f"{workspace}/{rel_path}"
        cmd = f"cd {workspace}/rollouts && uv run python {remote_script}"
        print(f"Running: {cmd}")
        print("-" * 50)
        for line in bifrost.exec_stream(cmd):
            print(line, end="")
        print("-" * 50)

    except KeyboardInterrupt:
        print("\n\nInterrupted!")
        keep_alive = True

    finally:
        if gpu is None:
            return
        if keep_alive:
            print()
            print("=" * 50)
            print(f"GPU kept alive: {gpu.id}")
            print(f"SSH: {gpu.ssh_connection_string()}")
            print()
            print(f"Rerun with:   --gpu-id {gpu.id}")
            print(f"Terminate:    broker terminate {gpu.id}")
            print("=" * 50)
        else:
            print("Cleaning up...")
            client.terminate_instance(gpu.id, gpu.provider)


def verify_functional(
    model_name: str,
    functional_code: str,
    test_inputs: list[list[int]],
    rtol: float = 1e-5,
    atol: float = 1e-5,
    gpu_id: str | None = None,
    keep_alive: bool = True,
) -> VerificationResult:
    """Verify functional code produces same output as original model.

    This function:
    1. Provisions a GPU (or reuses gpu_id)
    2. Pushes the functional code to remote
    3. Runs comparison with torch.allclose
    4. Returns results

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-0.5B")
        functional_code: The functional Python code as a string
        test_inputs: List of input token sequences to test
        rtol: Relative tolerance for torch.allclose
        atol: Absolute tolerance for torch.allclose
        gpu_id: Reuse existing GPU instance
        keep_alive: Keep GPU running after verification

    Returns:
        VerificationResult with matches, max_diff, and shapes
    """
    from bifrost.client import BifrostClient
    from broker.client import GPUClient
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parents[3] / ".env"
    load_dotenv(env_path)

    runpod_key = os.getenv("RUNPOD_API_KEY")
    assert runpod_key, f"RUNPOD_API_KEY not set"
    ssh_key_path = os.getenv("SSH_KEY_PATH", "~/.ssh/id_ed25519")

    # Create verification script
    verify_script = f'''
import torch
from transformers import AutoModelForCausalLM
import json

# The functional code to verify
{functional_code}

def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "{model_name}",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    # Extract weights dict
    weights = dict(model.state_dict())

    test_inputs = {test_inputs!r}
    rtol = {rtol}
    atol = {atol}

    results = []
    for input_seq in test_inputs:
        input_ids = torch.tensor([input_seq], device="cuda:0")

        with torch.no_grad():
            original_output = model(input_ids).logits

            # Run functional version
            functional_output = qwen_forward(input_ids, weights)

        matches = torch.allclose(original_output, functional_output, rtol=rtol, atol=atol)
        max_diff = (original_output - functional_output).abs().max().item()

        results.append({{
            "matches": matches,
            "max_diff": max_diff,
            "original_shape": list(original_output.shape),
            "functional_shape": list(functional_output.shape),
        }})

        print(f"Input {{input_seq}}: matches={{matches}}, max_diff={{max_diff:.6f}}")

    # Aggregate results
    all_match = all(r["matches"] for r in results)
    max_diff = max(r["max_diff"] for r in results)

    final = {{
        "matches": all_match,
        "max_diff": max_diff,
        "rtol": rtol,
        "atol": atol,
        "original_shape": results[0]["original_shape"],
        "functional_shape": results[0]["functional_shape"],
    }}

    print("RESULT_JSON:" + json.dumps(final))

if __name__ == "__main__":
    main()
'''

    client = GPUClient(credentials={"runpod": runpod_key}, ssh_key_path=ssh_key_path)
    gpu = None
    result_json = None

    try:
        if gpu_id:
            print(f"Reusing GPU: {gpu_id}")
            gpu = client.get_instance(gpu_id, provider="runpod")
            if not gpu:
                return VerificationResult(
                    matches=False,
                    max_diff=float("inf"),
                    rtol=rtol,
                    atol=atol,
                    original_shape=(),
                    functional_shape=(),
                    error=f"GPU {gpu_id} not found",
                )
            keep_alive = True
        else:
            print("Provisioning GPU...")
            gpu = client.create(
                query=(client.vram_gb >= 16) & (client.price_per_hour <= 0.5),
                name="verify-functional",
                cloud_type="secure",  # Required for direct SSH access
            )
            if not gpu:
                return VerificationResult(
                    matches=False,
                    max_diff=float("inf"),
                    rtol=rtol,
                    atol=atol,
                    original_shape=(),
                    functional_shape=(),
                    error="Failed to provision GPU",
                )
            print(f"GPU ready: {gpu.id}")

            if not gpu.wait_until_ssh_ready(timeout=300):
                client.terminate_instance(gpu.id, gpu.provider)
                return VerificationResult(
                    matches=False,
                    max_diff=float("inf"),
                    rtol=rtol,
                    atol=atol,
                    original_shape=(),
                    functional_shape=(),
                    error="SSH timeout",
                )

        print(f"SSH: {gpu.ssh_connection_string()}")

        bifrost = BifrostClient(gpu.ssh_connection_string(), ssh_key_path)

        # Bootstrap environment
        bootstrap = [
            "uv python install 3.12",
            "uv venv .venv --python 3.12 || true",
            "source .venv/bin/activate && uv pip install torch 'transformers<4.52'",
        ]
        for cmd in bootstrap:
            print(f"Running: {cmd}")
            result = bifrost.exec(cmd)
            if result.exit_code != 0:
                print(f"Bootstrap failed: {result.stderr}")

        # Write verification script to remote
        escaped_script = verify_script.replace("'", "'\\''")
        bifrost.exec(f"cat > /tmp/verify_functional.py << 'SCRIPT_EOF'\n{verify_script}\nSCRIPT_EOF")

        # Run verification
        print("Running verification...")
        print("-" * 50)
        for line in bifrost.exec_stream("source .venv/bin/activate && python /tmp/verify_functional.py"):
            print(line, end="")
            if line.startswith("RESULT_JSON:"):
                import json
                result_json = json.loads(line[len("RESULT_JSON:"):])
        print("-" * 50)

        if result_json:
            return VerificationResult(
                matches=result_json["matches"],
                max_diff=result_json["max_diff"],
                rtol=result_json["rtol"],
                atol=result_json["atol"],
                original_shape=tuple(result_json["original_shape"]),
                functional_shape=tuple(result_json["functional_shape"]),
            )
        else:
            return VerificationResult(
                matches=False,
                max_diff=float("inf"),
                rtol=rtol,
                atol=atol,
                original_shape=(),
                functional_shape=(),
                error="Failed to parse result",
            )

    except KeyboardInterrupt:
        print("\n\nInterrupted!")
        keep_alive = True
        return VerificationResult(
            matches=False,
            max_diff=float("inf"),
            rtol=rtol,
            atol=atol,
            original_shape=(),
            functional_shape=(),
            error="Interrupted",
        )

    finally:
        if gpu is None:
            return
        if keep_alive:
            print()
            print("=" * 50)
            print(f"GPU kept alive: {gpu.id}")
            print(f"SSH: {gpu.ssh_connection_string()}")
            print()
            print(f"Rerun with:   gpu_id='{gpu.id}'")
            print(f"Terminate:    broker terminate {gpu.id}")
            print("=" * 50)
        else:
            print("Cleaning up...")
            client.terminate_instance(gpu.id, gpu.provider)
