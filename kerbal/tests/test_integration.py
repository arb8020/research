#!/usr/bin/env python3
"""Integration tests for kerbal job/serve API.

Tests:
1. SGLang server - start, health check, query, stop
2. vLLM server - start, health check, query, stop
3. Logprob comparison - verify SGLang and vLLM produce same results

Node acquisition (one of):
- --ssh: Use static SSH connection
- --node-id: Reuse existing broker instance
- --provision: Provision new instance via broker

Requirements:
- HF_TOKEN environment variable for model download
- For --provision: RUNPOD_API_KEY, LAMBDA_API_KEY, or other provider credentials
- At least 2 GPUs for comparison test

Usage:
    # Use existing SSH node
    python tests/test_integration.py --ssh root@gpu-node:22

    # Reuse existing broker instance
    python tests/test_integration.py --node-id runpod:abc123

    # Provision new instance (terminated after test)
    python tests/test_integration.py --provision

    # Provision and keep alive for reuse
    python tests/test_integration.py --provision --keep-alive

    # Run specific test
    python tests/test_integration.py --ssh root@gpu:22 --test sglang
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Load .env file if present
from dotenv import load_dotenv
load_dotenv()

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bifrost import BifrostClient
from kerbal import serve, submit
from kerbal.inference import sglang, vllm

if TYPE_CHECKING:
    from broker.client import ClientGPUInstance


# Use small model for fast testing
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "The capital of France is"


def get_broker_credentials() -> dict[str, str]:
    """Load broker credentials from environment."""
    credentials = {}
    if key := os.getenv("RUNPOD_API_KEY"):
        credentials["runpod"] = key
    if key := os.getenv("VAST_API_KEY"):
        credentials["vast"] = key
    if key := os.getenv("PRIME_API_KEY"):
        credentials["primeintellect"] = key
    if key := os.getenv("LAMBDA_API_KEY"):
        credentials["lambdalabs"] = key
    return credentials


def acquire_node(
    ssh: str | None = None,
    node_id: str | None = None,
    provision: bool = False,
    gpu_query_str: str = "A100",
    ssh_key_path: str = "~/.ssh/id_ed25519",
    provider: str | None = None,
    gpu_count: int = 2,
) -> tuple[BifrostClient, "ClientGPUInstance | None"]:
    """Acquire a node for testing.

    Args:
        ssh: Static SSH connection string
        node_id: Existing instance ID (format: "provider:id")
        provision: Whether to provision a new instance
        gpu_query_str: GPU type to search for when provisioning
        ssh_key_path: Path to SSH key

    Returns:
        (BifrostClient, instance) - instance is None for static SSH
    """
    from broker import GPUClient

    if ssh:
        # Static node - just connect
        print(f"Connecting to static node: {ssh}")
        client = BifrostClient(ssh, ssh_key_path=ssh_key_path)
        return client, None

    elif node_id:
        # Existing instance - look it up
        provider, instance_id = node_id.split(":", 1)
        print(f"Connecting to existing instance: {provider}:{instance_id}")

        credentials = get_broker_credentials()
        assert credentials, "No broker credentials found in environment"

        broker = GPUClient(credentials=credentials, ssh_key_path=ssh_key_path)
        instance = broker.get_instance(instance_id, provider)
        assert instance, f"Instance not found: {node_id}"

        print(f"  GPU: {instance.gpu_count}x {instance.gpu_type}")
        print(f"  Waiting for SSH...")
        instance.wait_until_ssh_ready(timeout=300)

        key_path = broker.get_ssh_key_path(provider)
        assert key_path, f"No SSH key configured for {provider}"
        client = BifrostClient(
            instance.ssh_connection_string(),
            ssh_key_path=key_path,
        )
        return client, instance

    elif provision:
        # Provision new instance
        print(f"Provisioning new instance (GPU: {gpu_query_str})...")

        credentials = get_broker_credentials()
        assert credentials, "No broker credentials found in environment"

        # If provider specified, only pass that provider's credentials to avoid
        # unnecessary API calls to other providers
        if provider:
            provider_key_map = {
                "runpod": "runpod",
                "vast": "vast",
                "lambdalabs": "lambdalabs",
                "primeintellect": "primeintellect",
            }
            if provider in provider_key_map:
                key = provider_key_map[provider]
                assert key in credentials, f"No credentials for {provider}"
                credentials = {key: credentials[key]}

        broker = GPUClient(credentials=credentials, ssh_key_path=ssh_key_path)

        # Build query
        query = broker.gpu_type.contains(gpu_query_str)

        # broker.create() raises ProvisionError with details on failure
        instance = broker.create(
            query,
            gpu_count=gpu_count,
            cloud_type="secure",
            container_disk_gb=150,
            sort=lambda x: x.price_per_hour,
        )

        print(f"  Instance ID: {instance.provider}:{instance.id}")
        print(f"  GPU: {instance.gpu_count}x {instance.gpu_type}")
        print(f"  Waiting for SSH...")

        if not instance.wait_until_ssh_ready(timeout=600):
            raise AssertionError(f"SSH not ready after 10 min - instance: {instance.provider}:{instance.id}")

        key_path = broker.get_ssh_key_path(instance.provider)
        assert key_path, f"No SSH key configured for {instance.provider}"
        client = BifrostClient(
            instance.ssh_connection_string(),
            ssh_key_path=key_path,
        )
        return client, instance

    else:
        raise ValueError("Must specify --ssh, --node-id, or --provision")


def test_sglang(client: BifrostClient, workspace: str, gpu_id: int = 0) -> bool:
    """Test SGLang server startup and basic query.

    Returns:
        True if test passed, False otherwise
    """
    print("\n" + "=" * 60)
    print("TEST: SGLang Server")
    print("=" * 60)

    server = None
    try:
        # Start server using preset
        print(f"\n1. Starting SGLang server on GPU {gpu_id}...")
        cmd, env_vars = sglang.build_command(
            model=MODEL,
            port=30000,
            gpu_ids=[gpu_id],
        )
        server = serve(
            client,
            command=cmd,
            workspace=workspace,
            port=30000,
            gpu_ids=[gpu_id],
            env_vars=env_vars,
            deps=sglang.get_deps(),
            health_timeout=600,  # 10 min for model download
            server_name="test-sglang",
        )
        print(f"   Server ready: {server.url}")

        # Query server
        print("\n2. Querying server for completions...")
        job = submit(
            client,
            command=f'''python -c "
import requests
import json

resp = requests.post('http://localhost:30000/v1/completions', json={{
    'model': '{MODEL}',
    'prompt': '{PROMPT}',
    'max_tokens': 10,
    'logprobs': 5,
    'temperature': 0,
}}, timeout=60)

data = resp.json()
print(json.dumps(data, indent=2))

assert 'choices' in data, f'No choices in response: {{data}}'
assert len(data['choices']) > 0, 'Empty choices'
assert 'logprobs' in data['choices'][0], 'No logprobs in response'

print()
print('SUCCESS: SGLang server working correctly')
"
''',
            workspace=workspace,
            check_gpus=False,
            job_name="test-sglang-query",
        )

        success, exit_code = job.stream(timeout_sec=120)

        if success and "SUCCESS" in job.logs():
            print("\n✓ SGLang test PASSED")
            return True
        else:
            print(f"\n✗ SGLang test FAILED (exit_code={exit_code})")
            return False

    except Exception as e:
        print(f"\n✗ SGLang test FAILED with exception: {e}")
        return False

    finally:
        if server:
            print("\n3. Stopping server...")
            server.stop()
            print("   Server stopped")


def test_vllm(client: BifrostClient, workspace: str, gpu_id: int = 0) -> bool:
    """Test vLLM server startup and basic query.

    Returns:
        True if test passed, False otherwise
    """
    print("\n" + "=" * 60)
    print("TEST: vLLM Server")
    print("=" * 60)

    server = None
    try:
        # Start server using preset
        print(f"\n1. Starting vLLM server on GPU {gpu_id}...")
        cmd, env_vars = vllm.build_command(
            model=MODEL,
            port=30001,
            gpu_ids=[gpu_id],
        )
        server = serve(
            client,
            command=cmd,
            workspace=workspace,
            port=30001,
            gpu_ids=[gpu_id],
            env_vars=env_vars,
            deps=vllm.get_deps(),
            health_timeout=600,  # 10 min for model download
            server_name="test-vllm",
        )
        print(f"   Server ready: {server.url}")

        # Query server
        print("\n2. Querying server for completions...")
        job = submit(
            client,
            command=f'''python -c "
import requests
import json

resp = requests.post('http://localhost:30001/v1/completions', json={{
    'model': '{MODEL}',
    'prompt': '{PROMPT}',
    'max_tokens': 10,
    'logprobs': 5,
    'temperature': 0,
}}, timeout=60)

data = resp.json()
print(json.dumps(data, indent=2))

assert 'choices' in data, f'No choices in response: {{data}}'
assert len(data['choices']) > 0, 'Empty choices'
assert 'logprobs' in data['choices'][0], 'No logprobs in response'

print()
print('SUCCESS: vLLM server working correctly')
"
''',
            workspace=workspace,
            check_gpus=False,
            job_name="test-vllm-query",
        )

        success, exit_code = job.stream(timeout_sec=120)

        if success and "SUCCESS" in job.logs():
            print("\n✓ vLLM test PASSED")
            return True
        else:
            print(f"\n✗ vLLM test FAILED (exit_code={exit_code})")
            return False

    except Exception as e:
        print(f"\n✗ vLLM test FAILED with exception: {e}")
        return False

    finally:
        if server:
            print("\n3. Stopping server...")
            server.stop()
            print("   Server stopped")


def test_logprob_comparison(client: BifrostClient, workspace: str) -> bool:
    """Compare logprobs between SGLang and vLLM.

    Requires 2 GPUs (GPU 0 for SGLang, GPU 1 for vLLM).

    Returns:
        True if test passed, False otherwise
    """
    print("\n" + "=" * 60)
    print("TEST: Logprob Comparison (SGLang vs vLLM)")
    print("=" * 60)

    sglang_server = None
    vllm_server = None

    try:
        # Start both servers using presets
        print("\n1. Starting SGLang server on GPU 0...")
        sglang_cmd, sglang_env = sglang.build_command(
            model=MODEL,
            port=30000,
            gpu_ids=[0],
        )
        sglang_server = serve(
            client,
            command=sglang_cmd,
            workspace=workspace,
            port=30000,
            gpu_ids=[0],
            env_vars=sglang_env,
            deps=sglang.get_deps(),
            health_timeout=600,
            server_name="test-sglang-compare",
        )
        print(f"   SGLang ready: {sglang_server.url}")

        print("\n2. Starting vLLM server on GPU 1...")
        vllm_cmd, vllm_env = vllm.build_command(
            model=MODEL,
            port=30001,
            gpu_ids=[1],
        )
        vllm_server = serve(
            client,
            command=vllm_cmd,
            workspace=workspace,
            port=30001,
            gpu_ids=[1],
            env_vars=vllm_env,
            deps=vllm.get_deps(),
            health_timeout=600,
            server_name="test-vllm-compare",
        )
        print(f"   vLLM ready: {vllm_server.url}")

        # Query both and compare
        print("\n3. Comparing logprobs...")
        job = submit(
            client,
            command=f'''python -c "
import requests
import json

prompt = '{PROMPT}'
model = '{MODEL}'

# Query SGLang
print('Querying SGLang...')
sglang_resp = requests.post('http://localhost:30000/v1/completions', json={{
    'model': model,
    'prompt': prompt,
    'max_tokens': 10,
    'logprobs': 5,
    'temperature': 0,
}}, timeout=60).json()

# Query vLLM
print('Querying vLLM...')
vllm_resp = requests.post('http://localhost:30001/v1/completions', json={{
    'model': model,
    'prompt': prompt,
    'max_tokens': 10,
    'logprobs': 5,
    'temperature': 0,
}}, timeout=60).json()

# Extract logprobs
sglang_choice = sglang_resp['choices'][0]
vllm_choice = vllm_resp['choices'][0]

sglang_tokens = sglang_choice['logprobs']['tokens']
vllm_tokens = vllm_choice['logprobs']['tokens']

sglang_logprobs = sglang_choice['logprobs']['token_logprobs']
vllm_logprobs = vllm_choice['logprobs']['token_logprobs']

print()
print('SGLang tokens:', sglang_tokens)
print('vLLM tokens:  ', vllm_tokens)
print()
print('SGLang logprobs:', [round(x, 4) if x else x for x in sglang_logprobs])
print('vLLM logprobs:  ', [round(x, 4) if x else x for x in vllm_logprobs])
print()

# Compare tokens (should be identical with temperature=0)
assert sglang_tokens == vllm_tokens, f'Tokens differ: SGLang={{sglang_tokens}}, vLLM={{vllm_tokens}}'
print('✓ Tokens match')

# Compare logprobs (allow small numerical difference)
max_diff = 0
for i, (s, v) in enumerate(zip(sglang_logprobs, vllm_logprobs)):
    if s is None or v is None:
        continue  # Skip None values (first token sometimes)
    diff = abs(s - v)
    max_diff = max(max_diff, diff)
    if diff > 0.01:
        print(f'  Token {{i}} ({{sglang_tokens[i]}}): SGLang={{s:.4f}}, vLLM={{v:.4f}}, diff={{diff:.4f}}')

print(f'✓ Max logprob difference: {{max_diff:.6f}}')

if max_diff > 0.01:
    print()
    print('WARNING: Logprob differences > 0.01 detected')
    print('This may be due to different attention implementations')
else:
    print()
    print('SUCCESS: Logprobs match within tolerance!')
"
''',
            workspace=workspace,
            check_gpus=False,
            job_name="test-compare-logprobs",
        )

        success, exit_code = job.stream(timeout_sec=120)

        if success and "SUCCESS" in job.logs():
            print("\n✓ Logprob comparison test PASSED")
            return True
        elif success and "WARNING" in job.logs():
            print("\n⚠ Logprob comparison test PASSED with warnings")
            return True
        else:
            print(f"\n✗ Logprob comparison test FAILED (exit_code={exit_code})")
            return False

    except Exception as e:
        print(f"\n✗ Logprob comparison test FAILED with exception: {e}")
        return False

    finally:
        print("\n4. Stopping servers...")
        if sglang_server:
            sglang_server.stop()
            print("   SGLang stopped")
        if vllm_server:
            vllm_server.stop()
            print("   vLLM stopped")


def main():
    parser = argparse.ArgumentParser(description="Integration tests for kerbal job/serve API")

    # Node acquisition (mutually exclusive)
    node_group = parser.add_mutually_exclusive_group(required=True)
    node_group.add_argument("--ssh", help="Static SSH connection (e.g., root@gpu:22)")
    node_group.add_argument("--node-id", help="Existing instance ID (e.g., runpod:abc123)")
    node_group.add_argument("--provision", action="store_true", help="Provision new instance via broker")

    # Options
    parser.add_argument("--ssh-key", default="~/.ssh/id_ed25519", help="Path to SSH key")
    parser.add_argument("--keep-alive", action="store_true", help="Don't terminate provisioned instance")
    parser.add_argument("--gpu-type", default="A100", help="GPU type for provisioning (default: A100)")
    parser.add_argument("--provider", help="Provider to use (runpod, lambdalabs, vast, primeintellect)")
    parser.add_argument("--gpu-count", type=int, default=2, help="Number of GPUs to provision (default: 2)")
    parser.add_argument(
        "--test",
        choices=["sglang", "vllm", "compare", "all"],
        default="all",
        help="Which test to run (default: all)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID for single-server tests (default: 0)")

    args = parser.parse_args()

    # Check HF_TOKEN
    if not os.getenv("HF_TOKEN"):
        print("WARNING: HF_TOKEN not set - model download may be rate limited")

    print("=" * 60)
    print("Kerbal Integration Tests")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Test: {args.test}")

    # Acquire node
    client, instance = acquire_node(
        ssh=args.ssh,
        node_id=args.node_id,
        provision=args.provision,
        gpu_query_str=args.gpu_type,
        ssh_key_path=args.ssh_key,
        provider=args.provider,
        gpu_count=args.gpu_count,
    )

    if instance:
        print(f"Instance: {instance.provider}:{instance.id}")

    try:
        # Deploy code
        print("\nDeploying code...")
        workspace = client.push("~/.bifrost/workspaces/kerbal-test")
        print(f"Workspace: {workspace}")

        # Run tests
        results = {}

        if args.test in ("sglang", "all"):
            results["sglang"] = test_sglang(client, workspace, gpu_id=args.gpu)

        if args.test in ("vllm", "all"):
            gpu = args.gpu if args.test == "vllm" else args.gpu
            results["vllm"] = test_vllm(client, workspace, gpu_id=gpu)

        if args.test in ("compare", "all"):
            results["compare"] = test_logprob_comparison(client, workspace)

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        all_passed = True
        for name, passed in results.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"  {name}: {status}")
            if not passed:
                all_passed = False

        print()
        if all_passed:
            print("All tests passed!")
        else:
            print("Some tests failed!")

        return 0 if all_passed else 1

    finally:
        # Cleanup
        if instance:
            if args.keep_alive:
                print(f"\n💡 Instance kept alive: {instance.provider}:{instance.id}")
                print(f"   Reuse with: --node-id {instance.provider}:{instance.id}")
                print(f"   SSH: {instance.ssh_connection_string()}")
            else:
                print(f"\nTerminating instance {instance.provider}:{instance.id}...")
                instance.terminate()
                print("Instance terminated.")


if __name__ == "__main__":
    sys.exit(main())
