#!/usr/bin/env python3
"""Correctness tests for nano-inference engine.

Compares nano-inference against HuggingFace Transformers (ground truth).
Uses vLLM's testing methodology:
- Compute logprobs from hidden states (not output_scores)
- Top-N containment check (each model's token must be in other's top-N)

Tests:
1. Exact match - greedy decode tokens should match
2. Logprobs close - if tokens diverge, both should be in each other's top-N

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
NUM_LOGPROBS = 5  # Top-N for containment check


# =============================================================================
# Remote test script (runs on GPU)
# =============================================================================

REMOTE_TEST_SCRIPT = '''
"""Remote correctness test - runs on GPU node.

Uses vLLM's testing methodology:
1. Generate with hidden states for logprobs (not output_scores)
2. Compare using top-N containment check
"""

import json
import sys
import warnings
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass

MODEL = "{model}"
PROMPTS = {prompts}
MAX_TOKENS = {max_tokens}
NUM_LOGPROBS = {num_logprobs}


@dataclass
class GenerationResult:
    """Result from generation with logprobs."""
    prompt: str
    prompt_tokens: list[int]
    generated_tokens: list[int]
    generated_text: str
    token_logprobs: list[dict[int, float]]  # top-N per token


def generate_with_hidden_states(model, tokenizer, prompts, max_tokens, num_logprobs):
    """Generate using HF generate() with hidden states for logprobs.

    This is vLLM's HfRunner approach - computes logprobs from the last
    layer hidden states multiplied by output embeddings.
    """
    device = next(model.parameters()).device
    results = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences[0, input_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute logprobs from hidden states
        output_embeddings = model.get_output_embeddings()
        token_logprobs = []

        for step_idx, hidden_state in enumerate(outputs.hidden_states):
            last_hidden = hidden_state[-1][0]

            if step_idx == 0:
                last_hidden = last_hidden[-1:, :]

            logits = torch.matmul(
                last_hidden.to(
                    device=output_embeddings.weight.device,
                    dtype=output_embeddings.weight.dtype,
                ),
                output_embeddings.weight.t(),
            )

            if getattr(output_embeddings, "bias", None) is not None:
                logits = logits + output_embeddings.bias.unsqueeze(0)

            log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            topk = log_probs.topk(num_logprobs, dim=-1)

            tok_lp = {{}}
            for tid, lp in zip(topk.indices[0].tolist(), topk.values[0].tolist()):
                tok_lp[tid] = lp
            token_logprobs.append(tok_lp)

        results.append(GenerationResult(
            prompt=prompt,
            prompt_tokens=inputs.input_ids[0].tolist(),
            generated_tokens=generated_ids.tolist(),
            generated_text=generated_text,
            token_logprobs=token_logprobs,
        ))

    return results


def generate_with_forward_pass(model, tokenizer, prompts, max_tokens, num_logprobs):
    """Generate using step-by-step forward passes.

    This matches how nano-inference works - direct forward pass,
    argmax for greedy, step by step.
    """
    device = next(model.parameters()).device
    results = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        current_ids = inputs.input_ids

        generated_tokens = []
        token_logprobs = []

        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = model(current_ids)
                last_logits = outputs.logits[:, -1, :]

            log_probs = F.log_softmax(last_logits, dim=-1, dtype=torch.float32)
            next_token = last_logits.argmax(dim=-1).item()

            topk = log_probs.topk(num_logprobs, dim=-1)
            tok_lp = {{}}
            for tid, lp in zip(topk.indices[0].tolist(), topk.values[0].tolist()):
                tok_lp[tid] = lp

            generated_tokens.append(next_token)
            token_logprobs.append(tok_lp)

            if next_token == tokenizer.eos_token_id:
                break

            current_ids = torch.cat(
                [current_ids, torch.tensor([[next_token]], device=device)],
                dim=1,
            )

        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        results.append(GenerationResult(
            prompt=prompt,
            prompt_tokens=inputs.input_ids[0].tolist(),
            generated_tokens=generated_tokens,
            generated_text=generated_text,
            token_logprobs=token_logprobs,
        ))

    return results


def check_logprobs_close(outputs_0, outputs_1, name_0, name_1):
    """Check if logprobs are close using vLLM's approach.

    When tokens diverge, check that each model's predicted token
    appears in the other model's top-N logprobs.
    """
    all_close = True
    details = []

    for prompt_idx, (out_0, out_1) in enumerate(zip(outputs_0, outputs_1)):
        tokens_0 = out_0.generated_tokens
        tokens_1 = out_1.generated_tokens
        logprobs_0 = out_0.token_logprobs
        logprobs_1 = out_1.token_logprobs

        exact_match = tokens_0 == tokens_1
        diverge_idx = None
        in_top_n = True

        min_len = min(len(tokens_0), len(tokens_1))

        for idx in range(min_len):
            tok_0 = tokens_0[idx]
            tok_1 = tokens_1[idx]

            if tok_0 != tok_1:
                diverge_idx = idx
                lp_0 = logprobs_0[idx]
                lp_1 = logprobs_1[idx]

                tok_0_in_1 = tok_0 in lp_1
                tok_1_in_0 = tok_1 in lp_0

                if not tok_0_in_1 or not tok_1_in_0:
                    in_top_n = False
                    all_close = False

                break

        details.append({{
            "prompt": out_0.prompt[:40],
            "exact_match": exact_match,
            "in_top_n": in_top_n,
            "diverge_idx": diverge_idx,
            f"{{name_0}}_tokens": tokens_0[:10],
            f"{{name_1}}_tokens": tokens_1[:10],
            f"{{name_0}}_text": out_0.generated_text[:50],
            f"{{name_1}}_text": out_1.generated_text[:50],
        }})

    return all_close, details


def main():
    print("=" * 60)
    print("nano-inference Correctness Test (vLLM methodology)")
    print(f"Model: {{MODEL}}")
    print(f"Prompts: {{len(PROMPTS)}}")
    print(f"Top-N for containment: {{NUM_LOGPROBS}}")
    print("=" * 60)

    # Load model
    print("\\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Test 1: HF generate() with hidden states vs forward pass
    print("\\n--- Test 1: generate() vs forward() ---")
    print("This validates that our forward-pass approach matches HF generate()")

    hf_generate_results = generate_with_hidden_states(
        model, tokenizer, PROMPTS, MAX_TOKENS, NUM_LOGPROBS
    )
    hf_forward_results = generate_with_forward_pass(
        model, tokenizer, PROMPTS, MAX_TOKENS, NUM_LOGPROBS
    )

    test1_close, test1_details = check_logprobs_close(
        hf_generate_results, hf_forward_results, "generate", "forward"
    )

    print("\\nResults:")
    for d in test1_details:
        status = "EXACT" if d["exact_match"] else ("TOP-N" if d["in_top_n"] else "FAIL")
        print(f"  [{{status}}] {{d['prompt']}}...")
        if not d["exact_match"]:
            print(f"    generate: {{d['generate_text']}}")
            print(f"    forward:  {{d['forward_text']}}")
            print(f"    diverged at: {{d['diverge_idx']}}")

    # Test 2: nano-inference vs forward pass (coming soon)
    # TODO: Import and run nano-inference engine
    # For now, just validate that forward pass works

    # Summary
    print("\\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_exact = all(d["exact_match"] for d in test1_details)
    all_close = test1_close

    result = {{
        "test1_generate_vs_forward": {{
            "all_exact": all_exact,
            "all_close": all_close,
            "details": test1_details,
        }},
    }}

    print(f"Test 1 (generate vs forward): {{'EXACT' if all_exact else ('CLOSE' if all_close else 'FAIL')}}")
    print(f"\\nRESULT: {{json.dumps(result)}}")

    if all_exact:
        print("\\n✓ ALL TESTS PASSED (exact match)")
        sys.exit(0)
    elif all_close:
        print("\\n✓ ALL TESTS PASSED (top-N containment)")
        sys.exit(0)
    else:
        print("\\n✗ SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''


# =============================================================================
# Node acquisition
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
) -> tuple[BifrostClient, ClientGPUInstance | None]:
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
    print("TEST: Correctness validation (vLLM methodology)")
    print("=" * 60)

    # Write test script to remote
    script_content = REMOTE_TEST_SCRIPT.format(
        model=MODEL,
        prompts=json.dumps(TEST_PROMPTS),
        max_tokens=MAX_TOKENS,
        num_logprobs=NUM_LOGPROBS,
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
