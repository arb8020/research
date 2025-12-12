#!/usr/bin/env python3
"""Correctness tests for nano-inference engine.

Verifies that nano-inference produces outputs matching HuggingFace Transformers
(the ground truth). See kerbal/inference/CORRECTNESS_TESTING.md for methodology.

Tests:
1. Exact match - greedy decoding (temp=0) should exactly match Transformers
2. Logprobs similarity - verify logprobs are within tolerance

Node acquisition (one of):
- --ssh: Use static SSH connection
- --node-id: Reuse existing broker instance
- --provision: Provision new instance via broker

Usage:
    # Use existing SSH node
    python tests/test_inference_correctness.py --ssh root@gpu-node:22

    # Reuse existing broker instance
    python tests/test_inference_correctness.py --node-id runpod:abc123

    # Provision new instance (terminated after test)
    python tests/test_inference_correctness.py --provision

    # Provision and keep alive for reuse
    python tests/test_inference_correctness.py --provision --keep-alive
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
load_dotenv()

from bifrost import BifrostClient
from broker import GPUClient
from kerbal import submit
from kerbal.protocol import DependencyConfig

if TYPE_CHECKING:
    from broker.client import ClientGPUInstance


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TEST_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In 1969, humans first",
]
MAX_TOKENS = 10


# =============================================================================
# Remote test script (runs on GPU)
# =============================================================================

REMOTE_TEST_SCRIPT = '''
"""Remote correctness test - runs on GPU node."""

import json
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rollouts.inference import InferenceEngine, EngineConfig, SamplingParams

MODEL = "{model}"
PROMPTS = {prompts}
MAX_TOKENS = {max_tokens}


def get_transformers_reference(model_name: str, prompts: list[str], max_tokens: int):
    """Get reference outputs from HuggingFace Transformers."""
    print(f"Loading Transformers model: {{model_name}}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # Greedy
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = outputs.sequences[0, input_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract logprobs
        logprobs = []
        for i, (token_id, score) in enumerate(zip(generated_ids, outputs.scores)):
            log_probs = torch.log_softmax(score[0], dim=-1)
            token_logprob = log_probs[token_id].item()
            logprobs.append(token_logprob)

        results.append({{
            "prompt": prompt,
            "text": generated_text,
            "tokens": generated_ids.tolist(),
            "logprobs": logprobs,
        }})

    return results


def get_nano_inference_output(model_name: str, prompts: list[str], max_tokens: int):
    """Get outputs from our nano-inference engine."""
    print(f"Loading nano-inference: {{model_name}}")
    config = EngineConfig(
        model_path=model_name,
        block_size=16,
        max_batch_size=len(prompts),
    )
    engine = InferenceEngine(config)

    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy
        max_tokens=max_tokens,
    )

    samples = engine.generate_text(prompts, sampling_params)

    results = []
    for prompt, sample in zip(prompts, samples):
        results.append({{
            "prompt": prompt,
            "text": engine.tokenizer.decode(sample.completion_tokens, skip_special_tokens=True),
            "tokens": list(sample.completion_tokens),
            "logprobs": list(sample.logprobs),
        }})

    engine.shutdown()

    return results


def compare_results(name_a: str, results_a: list, name_b: str, results_b: list):
    """Compare outputs and return (all_match, details)."""
    details = []
    all_exact = True
    all_close = True

    for i, (a, b) in enumerate(zip(results_a, results_b)):
        prompt = a["prompt"]
        text_match = a["text"] == b["text"]
        token_match = a["tokens"] == b["tokens"]

        # Compare logprobs
        lp_a, lp_b = a["logprobs"], b["logprobs"]
        max_lp_diff = 0.0
        if len(lp_a) == len(lp_b):
            for la, lb in zip(lp_a, lp_b):
                if la is not None and lb is not None:
                    max_lp_diff = max(max_lp_diff, abs(la - lb))

        exact = text_match and token_match
        close = max_lp_diff < 0.01  # 1% tolerance

        if not exact:
            all_exact = False
        if not close:
            all_close = False

        details.append({{
            "prompt": prompt[:40],
            "exact_match": exact,
            "logprobs_close": close,
            "max_logprob_diff": max_lp_diff,
            f"{{name_a}}_text": a["text"][:50],
            f"{{name_b}}_text": b["text"][:50],
        }})

    return all_exact, all_close, details


def main():
    print("=" * 60)
    print("nano-inference Correctness Test")
    print(f"Model: {{MODEL}}")
    print(f"Prompts: {{len(PROMPTS)}}")
    print("=" * 60)

    # Get reference from Transformers
    print("\\n--- Getting Transformers reference ---")
    ref_results = get_transformers_reference(MODEL, PROMPTS, MAX_TOKENS)

    # Get nano-inference output
    print("\\n--- Getting nano-inference output ---")
    nano_results = get_nano_inference_output(MODEL, PROMPTS, MAX_TOKENS)

    # Compare
    print("\\n--- Comparing results ---")
    all_exact, all_close, details = compare_results("nano", nano_results, "transformers", ref_results)

    # Print details
    for d in details:
        status = "EXACT" if d["exact_match"] else ("CLOSE" if d["logprobs_close"] else "FAIL")
        print(f"[{{status}}] {{d['prompt']}}...")
        print(f"  nano:         {{d['nano_text']}}")
        print(f"  transformers: {{d['transformers_text']}}")
        print(f"  max logprob diff: {{d['max_logprob_diff']:.6f}}")

    # Summary
    print("\\n" + "=" * 60)
    result = {{"all_exact": all_exact, "all_close": all_close, "details": details}}
    print(f"RESULT: {{json.dumps(result)}}")

    if all_exact:
        print("\\nALL TESTS PASSED (exact match)")
        sys.exit(0)
    elif all_close:
        print("\\nALL TESTS PASSED (logprobs close)")
        sys.exit(0)
    else:
        print("\\nSOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''


# =============================================================================
# Node acquisition (same pattern as kerbal/tests/test_integration.py)
# =============================================================================

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
    gpu_query_str: str = "A100",
    ssh_key_path: str = "~/.ssh/id_ed25519",
    provider: str | None = None,
) -> tuple[BifrostClient, "ClientGPUInstance | None"]:
    """Acquire a node for testing."""
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
        print(f"Provisioning new instance (GPU: {gpu_query_str})...")
        instance = broker.create(
            broker.gpu_type.contains(gpu_query_str),
            gpu_count=1,
            cloud_type="secure",
            container_disk_gb=100,
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
# Test runner
# =============================================================================

def test_correctness(client: BifrostClient, workspace: str) -> bool:
    """Run correctness test on remote GPU."""
    print("\n" + "=" * 60)
    print("TEST: nano-inference vs Transformers")
    print("=" * 60)

    # Write test script to remote
    script_content = REMOTE_TEST_SCRIPT.format(
        model=MODEL,
        prompts=json.dumps(TEST_PROMPTS),
        max_tokens=MAX_TOKENS,
    )
    script_path = f"{workspace}/test_correctness_remote.py"
    client.exec(f"cat > {script_path} << 'SCRIPT_EOF'\n{script_content}\nSCRIPT_EOF")

    # Submit job
    print("\n1. Submitting correctness test job...")
    job = submit(
        client,
        command=f"PYTHONPATH={workspace}/rollouts:$PYTHONPATH python {script_path}",
        workspace=workspace,
        gpu_ids=[0],
        deps=DependencyConfig(
            project_name="correctness-test",
            dependencies=[
                "torch>=2.0",
                "transformers",
                "accelerate",
            ],
        ),
        job_name="test-correctness",
    )

    # Stream logs
    print("\n2. Running test (streaming logs)...")
    print("-" * 40)
    success, exit_code = job.stream(timeout_sec=1800)  # 30 min
    print("-" * 40)

    if success:
        print("\n✓ Correctness test PASSED")
        return True
    else:
        print(f"\n✗ Correctness test FAILED (exit code: {exit_code})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Correctness tests for nano-inference")

    node_group = parser.add_mutually_exclusive_group(required=True)
    node_group.add_argument("--ssh", help="Static SSH connection (e.g., root@gpu:22)")
    node_group.add_argument("--node-id", help="Existing instance ID (e.g., runpod:abc123)")
    node_group.add_argument("--provision", action="store_true", help="Provision new instance")

    parser.add_argument("--ssh-key", default="~/.ssh/id_ed25519", help="Path to SSH key")
    parser.add_argument("--keep-alive", action="store_true", help="Don't terminate instance")
    parser.add_argument("--gpu-type", default="A100", help="GPU type for provisioning")
    parser.add_argument("--provider", help="Provider (runpod, lambdalabs, vast)")

    args = parser.parse_args()

    if not os.getenv("HF_TOKEN"):
        print("WARNING: HF_TOKEN not set - model download may be rate limited")

    print("=" * 60)
    print("nano-inference Correctness Tests")
    print("=" * 60)
    print(f"Model: {MODEL}")

    client, instance = acquire_node(
        ssh=args.ssh,
        node_id=args.node_id,
        provision=args.provision,
        gpu_query_str=args.gpu_type,
        ssh_key_path=args.ssh_key,
        provider=args.provider,
    )

    try:
        print("\nDeploying code...")
        workspace = client.push("~/.bifrost/workspaces/rollouts-test")
        print(f"Workspace: {workspace}")

        passed = test_correctness(client, workspace)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  correctness: {'✓ PASSED' if passed else '✗ FAILED'}")

        return 0 if passed else 1

    finally:
        if instance:
            if args.keep_alive:
                print(f"\n💡 Instance kept alive: {instance.provider}:{instance.id}")
                print(f"   Reuse with: --node-id {instance.provider}:{instance.id}")
            else:
                print(f"\nTerminating instance {instance.provider}:{instance.id}...")
                instance.terminate()


if __name__ == "__main__":
    sys.exit(main())
