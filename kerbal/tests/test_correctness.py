#!/usr/bin/env python3
"""Correctness tests comparing inference backends.

Compares outputs from different inference engines to verify they produce
identical results for greedy decoding (temperature=0).

Backends supported:
- sglang: SGLang server
- vllm: vLLM server
- transformers: HuggingFace Transformers (reference)

Usage:
    # Compare SGLang vs vLLM (requires 2 GPUs)
    python tests/test_correctness.py --provision --backends sglang vllm

    # Compare both against Transformers reference
    python tests/test_correctness.py --provision --backends sglang vllm transformers

    # Reuse existing instance
    python tests/test_correctness.py --node-id runpod:abc123 --backends sglang vllm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from bifrost import BifrostClient  # noqa: E402

from kerbal import serve  # noqa: E402
from kerbal.inference import sglang, vllm  # noqa: E402
from kerbal.protocol import DependencyConfig  # noqa: E402

if TYPE_CHECKING:
    from broker.client import ClientGPUInstance


# Test configuration
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TEST_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In 1969, humans first",
]
MAX_TOKENS = 10
LOGPROB_TOLERANCE = 0.01  # Max allowed difference in logprobs


@dataclass
class CompletionResult:
    """Result from an inference backend."""

    prompt: str
    text: str
    tokens: list[int]
    logprobs: list[float]
    backend: str


def get_broker_credentials() -> dict[str, str]:
    """Load broker credentials from environment."""
    credentials = {}
    if key := os.getenv("RUNPOD_API_KEY"):
        credentials["runpod"] = key
    if key := os.getenv("VAST_API_KEY"):
        credentials["vast"] = key
    if key := os.getenv("LAMBDA_API_KEY"):
        credentials["lambdalabs"] = key
    return credentials


def acquire_node(
    ssh: str | None = None,
    node_id: str | None = None,
    provision: bool = False,
    gpu_query_str: str = "RTX 4090",
    ssh_key_path: str = "~/.ssh/id_ed25519",
    provider: str | None = None,
    gpu_count: int = 2,
) -> tuple[BifrostClient, ClientGPUInstance | None]:
    """Acquire a node for testing."""
    from broker import GPUClient

    if ssh:
        print(f"Connecting to static node: {ssh}")
        return BifrostClient(ssh, ssh_key_path=ssh_key_path), None

    credentials = get_broker_credentials()
    assert credentials, "No broker credentials found in environment"

    if provider:
        assert provider in credentials, f"No credentials for {provider}"
        credentials = {provider: credentials[provider]}

    broker = GPUClient(credentials=credentials, ssh_key_path=ssh_key_path)

    if node_id:
        prov, instance_id = node_id.split(":", 1)
        print(f"Connecting to existing instance: {node_id}")
        instance = broker.get_instance(instance_id, prov)
        assert instance, f"Instance not found: {node_id}"
    else:
        assert provision, "Must specify --ssh, --node-id, or --provision"
        print(f"Provisioning new instance (GPU: {gpu_query_str}, count: {gpu_count})...")
        instance = broker.create(
            broker.gpu_type.contains(gpu_query_str),
            gpu_count=gpu_count,
            cloud_type="secure",
            container_disk_gb=150,
            sort=lambda x: x.price_per_hour,
        )
        print(f"  Instance ID: {instance.provider}:{instance.id}")

    print(f"  GPU: {instance.gpu_count}x {instance.gpu_type}")
    print("  Waiting for SSH...")
    assert instance.wait_until_ssh_ready(timeout=600), "SSH not ready"

    key_path = broker.get_ssh_key_path(instance.provider)
    client = BifrostClient(instance.ssh_connection_string(), ssh_key_path=key_path)
    return client, instance


# =============================================================================
# Backend implementations
# =============================================================================


def query_server(
    client: BifrostClient,
    port: int,
    prompt: str,
    backend_name: str,
) -> CompletionResult:
    """Query an OpenAI-compatible server for completions with logprobs."""
    # Request with logprobs enabled
    result = client.exec(f'''
curl -s -X POST http://localhost:{port}/v1/completions \
  -H "Content-Type: application/json" \
  -d '{{"model": "{MODEL}", "prompt": "{prompt}", "max_tokens": {MAX_TOKENS}, "temperature": 0, "logprobs": 1}}'
''')

    if result.exit_code != 0:
        msg = f"Query failed: {result.stderr}"
        raise RuntimeError(msg)

    resp = json.loads(result.stdout)
    choice = resp["choices"][0]

    # Extract logprobs - format varies slightly between vLLM and SGLang
    logprobs_data = choice.get("logprobs", {})
    token_logprobs = logprobs_data.get("token_logprobs", [])
    tokens = logprobs_data.get("tokens", [])

    # Filter out None values (first token sometimes has None logprob)
    valid_logprobs = [lp for lp in token_logprobs if lp is not None]

    return CompletionResult(
        prompt=prompt,
        text=choice["text"],
        tokens=tokens,
        logprobs=valid_logprobs,
        backend=backend_name,
    )


def run_transformers_on_remote(
    client: BifrostClient,
    workspace: str,
    prompts: list[str],
    gpu_id: int = 0,
) -> list[CompletionResult]:
    """Run HuggingFace Transformers on remote GPU as reference."""

    # Write a script to run on the remote
    script = f'''
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "{MODEL}"
PROMPTS = {json.dumps(prompts)}
MAX_TOKENS = {MAX_TOKENS}

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda:{gpu_id}",
)
model.eval()

results = []
for prompt in PROMPTS:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = outputs.sequences[0, input_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Extract logprobs
    logprobs = []
    for token_id, score in zip(generated_ids, outputs.scores):
        log_probs = torch.log_softmax(score[0], dim=-1)
        logprobs.append(log_probs[token_id].item())

    results.append({{
        "prompt": prompt,
        "text": text,
        "tokens": generated_ids.tolist(),
        "logprobs": logprobs,
    }})

print(json.dumps(results))
'''

    script_path = f"{workspace}/run_transformers.py"
    client.exec(f"cat > {script_path} << 'SCRIPT_EOF'\n{script}\nSCRIPT_EOF")

    # Run with transformers deps
    from kerbal import submit

    job = submit(
        client,
        command=f"python {script_path}",
        workspace=workspace,
        gpu_ids=[gpu_id],
        deps=DependencyConfig(
            project_name="transformers-ref",
            dependencies=["torch>=2.0", "transformers", "accelerate"],
        ),
        job_name="transformers-reference",
    )

    success, _ = job.wait(timeout_sec=600)
    assert success, "Transformers reference failed"

    # Parse output
    output = job.logs()
    # Find the JSON line (last non-empty line)
    for raw_line in reversed(output.strip().split("\n")):
        stripped = raw_line.strip()
        if stripped.startswith("["):
            raw_results = json.loads(stripped)
            return [
                CompletionResult(
                    prompt=r["prompt"],
                    text=r["text"],
                    tokens=r["tokens"],
                    logprobs=r["logprobs"],
                    backend="transformers",
                )
                for r in raw_results
            ]

    msg = f"Could not parse Transformers output: {output}"
    raise RuntimeError(msg)


# =============================================================================
# Comparison logic
# =============================================================================


def compare_results(
    results_a: list[CompletionResult],
    results_b: list[CompletionResult],
    strict: bool = True,
) -> tuple[bool, list[dict]]:
    """Compare two sets of results.

    Args:
        results_a: Results from backend A
        results_b: Results from backend B
        strict: If True, fail on any text mismatch

    Returns:
        (all_passed, details)
    """
    assert len(results_a) == len(results_b), "Result count mismatch"

    all_passed = True
    details = []

    for a, b in zip(results_a, results_b, strict=False):
        assert a.prompt == b.prompt, f"Prompt mismatch: {a.prompt} vs {b.prompt}"

        text_match = a.text == b.text

        # Compare logprobs
        max_logprob_diff = 0.0
        if a.logprobs and b.logprobs:
            min_len = min(len(a.logprobs), len(b.logprobs))
            for i in range(min_len):
                diff = abs(a.logprobs[i] - b.logprobs[i])
                max_logprob_diff = max(max_logprob_diff, diff)

        logprobs_close = max_logprob_diff < LOGPROB_TOLERANCE

        passed = text_match if strict else (text_match or logprobs_close)
        if not passed:
            all_passed = False

        details.append({
            "prompt": a.prompt[:40],
            "text_match": text_match,
            "logprobs_close": logprobs_close,
            "max_logprob_diff": max_logprob_diff,
            f"{a.backend}_text": a.text[:50],
            f"{b.backend}_text": b.text[:50],
            "passed": passed,
        })

    return all_passed, details


def run_correctness_test(
    client: BifrostClient,
    workspace: str,
    backends: list[str],
    strict: bool = True,
) -> bool:
    """Run correctness comparison between specified backends."""

    print("\n" + "=" * 60)
    print(f"Correctness Test: {' vs '.join(backends)}")
    print(f"Model: {MODEL}")
    print(f"Prompts: {len(TEST_PROMPTS)}")
    print(f"Strict mode: {strict}")
    print("=" * 60)

    servers = {}
    results_by_backend: dict[str, list[CompletionResult]] = {}

    try:
        gpu_idx = 0

        # Start servers for server-based backends
        for backend in backends:
            if backend == "sglang":
                print(f"\n1. Starting SGLang server on GPU {gpu_idx}...")
                cmd, env_vars = sglang.build_command(
                    model=MODEL,
                    port=30000 + gpu_idx,
                    gpu_ids=[gpu_idx],
                )
                servers["sglang"] = serve(
                    client,
                    command=cmd,
                    workspace=workspace,
                    port=30000 + gpu_idx,
                    gpu_ids=[gpu_idx],
                    env_vars=env_vars,
                    deps=sglang.get_deps(),
                    health_timeout=600,
                    server_name=f"test-sglang-{gpu_idx}",
                )
                print(f"   SGLang ready on port {30000 + gpu_idx}")
                gpu_idx += 1

            elif backend == "vllm":
                print(f"\n2. Starting vLLM server on GPU {gpu_idx}...")
                cmd, env_vars = vllm.build_command(
                    model=MODEL,
                    port=30000 + gpu_idx,
                    gpu_ids=[gpu_idx],
                )
                servers["vllm"] = serve(
                    client,
                    command=cmd,
                    workspace=workspace,
                    port=30000 + gpu_idx,
                    gpu_ids=[gpu_idx],
                    env_vars=env_vars,
                    deps=vllm.get_deps(),
                    health_timeout=600,
                    server_name=f"test-vllm-{gpu_idx}",
                )
                print(f"   vLLM ready on port {30000 + gpu_idx}")
                gpu_idx += 1

        # Query all backends
        print("\n3. Running queries...")

        for backend in backends:
            print(f"   Querying {backend}...")

            if backend == "transformers":
                results_by_backend[backend] = run_transformers_on_remote(
                    client, workspace, TEST_PROMPTS, gpu_id=0
                )
            else:
                # Query server
                port = 30000 + (0 if backend == "sglang" else 1)
                results = []
                for prompt in TEST_PROMPTS:
                    result = query_server(client, port, prompt, backend)
                    results.append(result)
                results_by_backend[backend] = results

        # Compare pairwise
        print("\n4. Comparing results...")

        all_passed = True
        backend_list = list(backends)

        for i in range(len(backend_list)):
            for j in range(i + 1, len(backend_list)):
                ba, bb = backend_list[i], backend_list[j]
                print(f"\n   {ba} vs {bb}:")

                passed, details = compare_results(
                    results_by_backend[ba],
                    results_by_backend[bb],
                    strict=strict,
                )

                for d in details:
                    status = "✓" if d["passed"] else "✗"
                    print(f"   {status} {d['prompt']}...")
                    if not d["text_match"]:
                        print(f"      {ba}: {d[f'{ba}_text']}")
                        print(f"      {bb}: {d[f'{bb}_text']}")
                    print(f"      max logprob diff: {d['max_logprob_diff']:.6f}")

                if not passed:
                    all_passed = False

        # Summary
        print("\n" + "=" * 60)
        if all_passed:
            print("✓ All comparisons PASSED")
        else:
            print("✗ Some comparisons FAILED")
        print("=" * 60)

        return all_passed

    finally:
        print("\n5. Stopping servers...")
        for name, server in servers.items():
            server.stop()
            print(f"   {name} stopped")


def main() -> int:
    parser = argparse.ArgumentParser(description="Correctness tests for inference backends")

    node_group = parser.add_mutually_exclusive_group(required=True)
    node_group.add_argument("--ssh", help="Static SSH connection")
    node_group.add_argument("--node-id", help="Existing instance ID")
    node_group.add_argument("--provision", action="store_true", help="Provision new instance")

    parser.add_argument("--ssh-key", default="~/.ssh/id_ed25519")
    parser.add_argument("--keep-alive", action="store_true", help="Keep instance after test")
    parser.add_argument("--gpu-type", default="RTX 4090")
    parser.add_argument("--provider", help="Cloud provider")
    parser.add_argument("--gpu-count", type=int, default=2)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["sglang", "vllm"],
        choices=["sglang", "vllm", "transformers"],
        help="Backends to compare",
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="Pass if logprobs are close even if text differs",
    )

    args = parser.parse_args()

    if not os.getenv("HF_TOKEN"):
        print("WARNING: HF_TOKEN not set")

    client, instance = acquire_node(
        ssh=args.ssh,
        node_id=args.node_id,
        provision=args.provision,
        gpu_query_str=args.gpu_type,
        ssh_key_path=args.ssh_key,
        provider=args.provider,
        gpu_count=args.gpu_count,
    )

    try:
        print("\nDeploying code...")
        workspace = client.push("~/.bifrost/workspaces/kerbal-correctness")
        print(f"Workspace: {workspace}")

        passed = run_correctness_test(
            client,
            workspace,
            backends=args.backends,
            strict=not args.lenient,
        )

        return 0 if passed else 1

    finally:
        if instance:
            if args.keep_alive:
                print(f"\n💡 Instance kept alive: {instance.provider}:{instance.id}")
            else:
                print("\nTerminating instance...")
                instance.terminate()


if __name__ == "__main__":
    sys.exit(main())
