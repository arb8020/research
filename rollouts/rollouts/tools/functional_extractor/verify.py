"""GPU verification for functional model extraction.

Uses bifrost/broker/kerbal to run verification on remote GPU instances.

CLI Usage:
    # Verify functional code against model
    python -m tools.functional_extractor.verify configs/qwen3_0.6b.py qwen3_functional.py

    # Reuse existing GPU
    python -m tools.functional_extractor.verify configs/qwen3_0.6b.py qwen3_functional.py --gpu-id abc123

    # Keep GPU alive after completion
    python -m tools.functional_extractor.verify configs/qwen3_0.6b.py qwen3_functional.py --keep-alive

Programmatic Usage:
    from tools.functional_extractor.verify import verify_functional
    from tools.functional_extractor.config import DeploymentConfig, VerificationConfig

    result = verify_functional(
        functional_code=Path("qwen3_functional.py").read_text(),
        verification=VerificationConfig(model_name="Qwen/Qwen3-0.6B", forward_fn_name="qwen3_forward"),
        deployment=DeploymentConfig(vram_gb=16),
    )
    assert result.matches, f"Max diff: {result.max_diff}"
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import DeploymentConfig, VerificationConfig

if TYPE_CHECKING:
    from bifrost.client import BifrostClient
    from broker.client import ClientGPUInstance, GPUClient


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


@dataclass(frozen=True)
class GPUHandle:
    """Handle to a provisioned GPU instance."""

    instance: ClientGPUInstance
    client: GPUClient
    bifrost: BifrostClient
    ssh_key_path: str
    workspace: str  # Expanded workspace path


def _load_env() -> tuple[str, str]:
    """Load environment variables for GPU provisioning."""
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parents[3] / ".env"
    load_dotenv(env_path)

    runpod_key = os.getenv("RUNPOD_API_KEY")
    assert runpod_key, f"RUNPOD_API_KEY not set (looked in {env_path})"
    ssh_key_path = os.getenv("SSH_KEY_PATH", "~/.ssh/id_ed25519")

    return runpod_key, ssh_key_path


def provision_gpu(
    deployment: DeploymentConfig,
    gpu_id: str | None = None,
    name: str = "verify-functional",
) -> GPUHandle | None:
    """Provision or reuse a GPU instance.

    Args:
        deployment: Deployment configuration
        gpu_id: Existing GPU instance ID to reuse
        name: Name for the GPU instance

    Returns:
        GPUHandle if successful, None if provisioning failed
    """
    from bifrost.client import BifrostClient
    from broker.client import GPUClient

    runpod_key, ssh_key_path = _load_env()
    client = GPUClient(credentials={"runpod": runpod_key}, ssh_key_path=ssh_key_path)

    if gpu_id:
        print(f"Reusing GPU: {gpu_id}")
        instance = client.get_instance(gpu_id, provider="runpod")
        if not instance:
            print(f"GPU {gpu_id} not found (is it still running?)")
            return None
    else:
        print("Provisioning GPU...")

        # Build query
        query = (
            (client.vram_gb >= deployment.vram_gb)
            & (client.price_per_hour <= deployment.max_price)
            & (client.manufacturer == "Nvidia")
        )

        if deployment.min_cpu_ram > 0:
            query = query & (client.memory_gb >= deployment.min_cpu_ram)

        if deployment.gpu_filter:
            query = query & (client.gpu_type.contains(deployment.gpu_filter))

        instance = client.create(
            query=query,
            name=name,
            cloud_type=deployment.cloud_type,
            gpu_count=deployment.gpu_count,
            sort=lambda x: x.price_per_hour,
            reverse=False,
            container_disk_gb=deployment.container_disk,
            volume_disk_gb=deployment.volume_disk if deployment.volume_disk > 0 else None,
        )

        if not instance:
            print("Failed to provision GPU - no matching offers")
            return None

        print(f"GPU ready: {instance.id}")

        if not instance.wait_until_ssh_ready(timeout=deployment.ssh_timeout):
            print("SSH timeout")
            client.terminate_instance(instance.id, instance.provider)
            return None

    print(f"SSH: {instance.ssh_connection_string()}")
    bifrost = BifrostClient(instance.ssh_connection_string(), ssh_key_path)

    # Expand workspace path
    workspace = bifrost.expand_path("~/.bifrost/workspaces/verify")

    return GPUHandle(
        instance=instance,
        client=client,
        bifrost=bifrost,
        ssh_key_path=ssh_key_path,
        workspace=workspace,
    )


def terminate_gpu(handle: GPUHandle) -> None:
    """Terminate a GPU instance."""
    print("Cleaning up...")
    handle.client.terminate_instance(handle.instance.id, handle.instance.provider)


def print_gpu_info(handle: GPUHandle) -> None:
    """Print GPU connection info for reuse."""
    print()
    print("=" * 50)
    print(f"GPU kept alive: {handle.instance.id}")
    print(f"SSH: {handle.instance.ssh_connection_string()}")
    print()
    print(f"Rerun with:   --gpu-id {handle.instance.id}")
    print(f"Terminate:    broker terminate {handle.instance.id}")
    print("=" * 50)


def setup_env(handle: GPUHandle, requirements: list[str]) -> str:
    """Setup Python environment using kerbal.

    Args:
        handle: GPU handle with bifrost client
        requirements: List of pip packages to install

    Returns:
        Path to venv python binary
    """
    # TODO: Migrate to bifrost v2 API - kerbal has been deleted
    # See bifrost.ProcessSpec + client.submit() for the new pattern
    raise NotImplementedError("kerbal.python_env has been removed - migrate to bifrost v2 API")


def run_on_gpu(
    script_path: str,
    deployment: DeploymentConfig | None = None,
    requirements: list[str] | None = None,
    keep_alive: bool = False,
    gpu_id: str | None = None,
) -> None:
    """Run a script on a remote GPU via broker/bifrost/kerbal.

    Args:
        script_path: Path to the script (__file__ from caller)
        deployment: Deployment configuration (uses defaults if None)
        requirements: Pip packages to install (uses defaults if None)
        keep_alive: Keep GPU running after completion
        gpu_id: Reuse existing GPU instance ID (skips provisioning)
    """
    if deployment is None:
        deployment = DeploymentConfig()

    if requirements is None:
        requirements = [
            "torch",
            "transformers>=4.50",
            "accelerate",
            "safetensors",
        ]

    # Get script path relative to git root
    script = Path(script_path).resolve()
    git_root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    )
    rel_path = script.relative_to(git_root)

    handle = None
    try:
        handle = provision_gpu(deployment, gpu_id=gpu_id, name=f"run-{script.stem}")
        if not handle:
            return

        if gpu_id:
            keep_alive = True  # Always keep alive when reusing

        # Deploy code via git sync (no bootstrap - use kerbal instead)
        print("Deploying code...")
        handle.bifrost.push(workspace_path=handle.workspace)
        print("Code deployed")

        # Setup Python environment with kerbal
        venv_python = setup_env(handle, requirements)

        # Run with streaming output
        remote_script = f"{handle.workspace}/{rel_path}"
        cmd = f"{venv_python} {remote_script}"
        print(f"Running: {cmd}")
        print("-" * 50)
        for line in handle.bifrost.exec_stream(cmd, working_dir=handle.workspace):
            print(line)
        print("-" * 50)

    except KeyboardInterrupt:
        print("\n\nInterrupted!")
        keep_alive = True

    finally:
        if handle is None:
            return
        if keep_alive:
            print_gpu_info(handle)
        else:
            terminate_gpu(handle)


def _build_verification_script(
    functional_code: str,
    verification: VerificationConfig,
) -> str:
    """Build the remote verification script."""
    # Strip __future__ imports from functional code - we'll add them at the top
    functional_lines = functional_code.split("\n")
    filtered_lines = [
        line for line in functional_lines if not line.strip().startswith("from __future__")
    ]
    functional_code_clean = "\n".join(filtered_lines)

    return f'''from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM
import json
import torch.nn.functional as F
from torch import Tensor

# The functional code to verify (with __future__ imports stripped)
{functional_code_clean}

def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "{verification.model_name}",
        torch_dtype=torch.{verification.torch_dtype},
        device_map="{verification.device_map}",
    )
    model.eval()

    # Extract weights dict
    weights = dict(model.state_dict())

    test_inputs = {verification.test_inputs!r}
    rtol = {verification.rtol}
    atol = {verification.atol}

    results = []
    for input_seq in test_inputs:
        input_ids = torch.tensor([input_seq], device="cuda:0")

        with torch.no_grad():
            original_output = model(input_ids).logits

            # Run functional version
            functional_output = {verification.forward_fn_name}(input_ids, weights)

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


def verify_functional(
    functional_code: str,
    verification: VerificationConfig,
    deployment: DeploymentConfig | None = None,
    gpu_id: str | None = None,
    keep_alive: bool = True,
) -> VerificationResult:
    """Verify functional code produces same output as original model.

    This function:
    1. Provisions a GPU (or reuses gpu_id)
    2. Sets up Python environment with kerbal
    3. Runs comparison with torch.allclose
    4. Returns results

    Args:
        functional_code: The functional Python code as a string
        verification: Verification configuration (model, tolerances, etc.)
        deployment: Deployment configuration (GPU specs, etc.)
        gpu_id: Reuse existing GPU instance
        keep_alive: Keep GPU running after verification

    Returns:
        VerificationResult with matches, max_diff, and shapes
    """
    import json

    if deployment is None:
        deployment = DeploymentConfig()

    handle = None
    result_json = None

    try:
        handle = provision_gpu(deployment, gpu_id=gpu_id, name="verify-functional")
        if not handle:
            return VerificationResult(
                matches=False,
                max_diff=float("inf"),
                rtol=verification.rtol,
                atol=verification.atol,
                original_shape=(),
                functional_shape=(),
                error="Failed to provision GPU",
            )

        if gpu_id:
            keep_alive = True

        # Setup Python environment using kerbal
        requirements = [
            "torch",
            "transformers>=4.50",
            "accelerate",
            "safetensors",
        ]
        venv_python = setup_env(handle, requirements)

        # Write verification script to remote
        verify_script = _build_verification_script(functional_code, verification)
        script_path = f"{handle.workspace}/verify_functional.py"
        handle.bifrost.exec(f"cat > {script_path} << 'SCRIPT_EOF'\n{verify_script}\nSCRIPT_EOF")

        # Run verification
        print("Running verification...")
        print("-" * 50)
        for line in handle.bifrost.exec_stream(
            f"{venv_python} {script_path}",
            working_dir=handle.workspace,
        ):
            print(line)
            if line.startswith("RESULT_JSON:"):
                result_json = json.loads(line[len("RESULT_JSON:") :])
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
                rtol=verification.rtol,
                atol=verification.atol,
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
            rtol=verification.rtol,
            atol=verification.atol,
            original_shape=(),
            functional_shape=(),
            error="Interrupted",
        )

    finally:
        if handle is None:
            return
        if keep_alive:
            print_gpu_info(handle)
        else:
            terminate_gpu(handle)


def load_config_from_file(config_path: str) -> tuple[DeploymentConfig, VerificationConfig]:
    """Load deployment and verification configs from a Python file.

    Args:
        config_path: Path to config .py file

    Returns:
        Tuple of (DeploymentConfig, VerificationConfig)
    """
    assert config_path.endswith(".py"), f"Config must be .py file, got {config_path}"

    spec = importlib.util.spec_from_file_location("config_module", config_path)
    assert spec is not None, f"Failed to load spec from {config_path}"
    assert spec.loader is not None, f"Spec loader is None for {config_path}"

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "deployment"), "Config file must define 'deployment' variable"
    assert hasattr(module, "verification"), "Config file must define 'verification' variable"

    deployment: DeploymentConfig = module.deployment
    verification: VerificationConfig = module.verification

    assert isinstance(deployment, DeploymentConfig), (
        f"Expected DeploymentConfig, got {type(deployment)}"
    )
    assert isinstance(verification, VerificationConfig), (
        f"Expected VerificationConfig, got {type(verification)}"
    )

    return deployment, verification


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify functional code matches original model on remote GPU"
    )
    parser.add_argument("config", help="Path to config file (e.g., configs/qwen3_0.6b.py)")
    parser.add_argument("functional_code", help="Path to functional code .py file")
    parser.add_argument("--gpu-id", type=str, help="Reuse existing GPU instance ID")
    parser.add_argument("--keep-alive", action="store_true", help="Keep GPU after completion")
    args = parser.parse_args()

    # Load config
    deployment, verification = load_config_from_file(args.config)

    print(f"Model: {verification.model_name}")
    print(f"Forward function: {verification.forward_fn_name}")
    print(f"GPU: {deployment.vram_gb}GB VRAM, max ${deployment.max_price}/hr")
    if deployment.gpu_filter:
        print(f"GPU filter: {deployment.gpu_filter}")
    if deployment.gpu_count > 1:
        print(f"GPU count: {deployment.gpu_count}")
    print()

    # Load functional code
    functional_code = Path(args.functional_code).read_text()

    # Run verification
    result = verify_functional(
        functional_code=functional_code,
        verification=verification,
        deployment=deployment,
        gpu_id=args.gpu_id,
        keep_alive=args.keep_alive,
    )

    # Report results
    print()
    if result.error:
        print(f"ERROR: {result.error}")
        exit(1)
    elif result.matches:
        print(f"PASSED: Outputs match (max diff: {result.max_diff:.2e})")
        exit(0)
    else:
        print(f"FAILED: Outputs do not match (max diff: {result.max_diff:.2e})")
        print(f"  Original shape: {result.original_shape}")
        print(f"  Functional shape: {result.functional_shape}")
        exit(1)


if __name__ == "__main__":
    main()
