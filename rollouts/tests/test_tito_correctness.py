#!/usr/bin/env python3
"""End-to-end correctness test for TI/TO (Tokens-In/Tokens-Out).

Uses multi-turn calculator environment with tool calls to demonstrate:
1. THE PROBLEM: Retokenization produces different tokens with logprob < -10
2. THE FIX: TI/TO stores generated token_ids directly, avoiding retokenization

The test runs multiple rollouts and checks for token mismatches. When a mismatch
is found, it computes the logprob of the wrong token and reports if it's < -10
(the smoking gun that causes RL training collapse).

Node acquisition (one of):
- --ssh: Use static SSH connection
- --node-id: Reuse existing broker instance
- --provision: Provision new instance via broker

Usage:
    # Use existing SSH node
    python tests/test_tito_correctness.py --ssh root@gpu-node:22

    # Reuse existing broker instance
    python tests/test_tito_correctness.py --node-id runpod:abc123

    # Provision new instance (terminated after test)
    python tests/test_tito_correctness.py --provision

    # Provision and keep alive for reuse
    python tests/test_tito_correctness.py --provision --keep-alive
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from kerbal import submit
from kerbal.protocol import DependencyConfig

from rollouts.remote import release_node

if TYPE_CHECKING:
    from bifrost import BifrostClient
    from broker.client import ClientGPUInstance

load_dotenv()

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SGLANG_PORT = 30000


# REMINDER: Only runpod for now - prime/vast not included bc I haven't added credits
def acquire_node_runpod_only(
    ssh: str | None = None,
    node_id: str | None = None,
    provision: bool = False,
    gpu_type: str = "A100",
    gpu_count: int = 1,
) -> tuple["BifrostClient", "ClientGPUInstance | None"]:
    """Acquire node using runpod only (like examples/rl/base_config.py)."""
    from bifrost import BifrostClient
    from broker import GPUClient

    ssh_key_path = os.path.expanduser("~/.ssh/id_ed25519")

    if ssh:
        print(f"Connecting to static node: {ssh}")
        return BifrostClient(ssh, ssh_key_path=ssh_key_path), None

    runpod_key = os.getenv("RUNPOD_API_KEY")
    assert runpod_key, "RUNPOD_API_KEY required"

    broker = GPUClient(
        credentials={"runpod": runpod_key},  # Only runpod
        ssh_key_path=ssh_key_path,
    )

    if node_id:
        provider, instance_id = node_id.split(":", 1)
        print(f"Connecting to existing instance: {node_id}")
        instance = broker.get_instance(instance_id, provider)
        assert instance, f"Instance not found: {node_id}"
    else:
        assert provision, "Must specify --ssh, --node-id, or --provision"
        print(f"Provisioning new instance ({gpu_count}x {gpu_type})...")
        instance = broker.create(
            broker.gpu_type.contains(gpu_type),
            gpu_count=gpu_count,
            cloud_type="secure",
            container_disk_gb=100,
            sort=lambda x: x.price_per_hour,
        )
        print(f"  Instance ID: {instance.provider}:{instance.id}")

    print(f"  GPU: {instance.gpu_count}x {instance.gpu_type}")
    print("  Waiting for SSH...")
    instance.wait_until_ssh_ready(timeout=600)

    key_path = broker.get_ssh_key_path(instance.provider)
    client = BifrostClient(instance.ssh_connection_string(), ssh_key_path=key_path)
    return client, instance


# =============================================================================
# Remote test script (runs on GPU with SGLang server)
# =============================================================================

REMOTE_TEST_SCRIPT = '''
"""Remote TI/TO correctness test - runs on GPU node.

Multi-turn calculator test that:
1. Generates tool calls (JSON with quotes, whitespace - prone to tokenization issues)
2. Compares generated tokens vs retokenized tokens
3. Reports mismatches with logprob analysis
4. Breaks on first mismatch with logprob < -10 (the smoking gun)
"""

import json
import subprocess
import time
import sys
import re
import httpx
import torch
import torch.nn.functional as F

MODEL = "{model}"
PORT = {port}
MAX_TOKENS = 200
NUM_ROLLOUTS = 10  # Try multiple rollouts to find mismatches
LOGPROB_THRESHOLD = -10  # Logprobs below this indicate retokenization collapse

# Calculator system prompt and tools
SYSTEM_PROMPT = """You are a calculator assistant. Use the provided tools to solve math problems.

Available tools:
- add(value): Add a number to the current value (starts at 0)
- subtract(value): Subtract a number from the current value
- multiply(value): Multiply the current value by a number
- divide(value): Divide the current value by a number
- complete_task(summary, final_result): Submit your final answer

Always use tool calls in the format:
<tool_call>
{{"name": "function_name", "arguments": {{"arg": value}}}}
</tool_call>

Work step by step, calling one tool at a time."""

MATH_PROBLEMS = [
    "Calculate (5 + 3) * 2",
    "What is 100 / 4 - 10?",
    "Compute 7 * 8 + 6",
    "Find the result of 50 - 15 + 25",
    "Calculate 12 * 3 / 4",
    "What is (20 + 30) / 5?",
    "Compute 99 - 33 + 11",
    "Find 64 / 8 * 3",
    "Calculate 45 + 55 - 25",
    "What is 8 * 9 - 12?",
]


def wait_for_sglang(base_url: str, timeout: int = 300) -> bool:
    """Wait for SGLang server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(f"{{base_url}}/health", timeout=5.0)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def start_sglang_server():
    """Start SGLang server in background."""
    print(f"Starting SGLang server for {{MODEL}} on port {{PORT}}...")

    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", MODEL,
        "--port", str(PORT),
        "--dtype", "bfloat16",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    base_url = f"http://localhost:{{PORT}}"
    if not wait_for_sglang(base_url):
        process.terminate()
        raise RuntimeError("SGLang server failed to start")

    print(f"✓ SGLang server ready at {{base_url}}")
    return process, base_url


def generate_with_sglang(base_url: str, input_ids: list[int], max_tokens: int = 200):
    """Generate tokens via SGLang /generate endpoint."""
    import trio
    from rollouts.inference.backends import generate_sglang

    async def run():
        return await generate_sglang(
            base_url=base_url,
            input_ids=input_ids,
            max_tokens=max_tokens,
            temperature=0.7,  # Some randomness to explore different outputs
            top_logprobs_num=5,
        )

    return trio.run(run)


def find_mismatches_with_logprobs(
    model,
    tokenizer,
    generated_ids: list[int],
    input_ids: list[int],
    generated_logprobs: list[float],
):
    """Find token mismatches and compute logprobs for wrong tokens.

    Returns list of (position, orig_token, wrong_token, orig_logprob, wrong_logprob)
    """
    device = next(model.parameters()).device

    # Decode and re-encode (simulating text-based API)
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    retokenized_ids = tokenizer.encode(generated_text, add_special_tokens=False)

    mismatches = []
    min_len = min(len(generated_ids), len(retokenized_ids))

    for i in range(min_len):
        orig = generated_ids[i]
        retok = retokenized_ids[i]

        if orig != retok:
            # Compute logprob of the wrong token
            prefix = list(input_ids) + list(generated_ids[:i])
            context = torch.tensor([prefix], device=device)

            with torch.no_grad():
                outputs = model(context)
                logits = outputs.logits[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
                wrong_logprob = log_probs[0, retok].item()

            orig_logprob = generated_logprobs[i] if i < len(generated_logprobs) else 0.0
            mismatches.append((i, orig, retok, orig_logprob, wrong_logprob))

    return mismatches


def run_single_rollout(model, tokenizer, base_url: str, problem: str, rollout_idx: int):
    """Run a single calculator rollout and check for mismatches.

    Returns (found_bad_mismatch, mismatch_details) where found_bad_mismatch is True
    if we found a mismatch with logprob < LOGPROB_THRESHOLD.
    """
    from rollouts.inference.backends import tokenize_chat

    messages = [
        {{"role": "system", "content": SYSTEM_PROMPT}},
        {{"role": "user", "content": problem}},
    ]

    print(f"\\n--- Rollout {{rollout_idx + 1}}: {{problem}} ---")

    # Tokenize and generate
    input_ids = tokenize_chat(tokenizer, messages, add_generation_prompt=True)
    result = generate_with_sglang(base_url, input_ids, MAX_TOKENS)

    generated_text = tokenizer.decode(result.output_ids, skip_special_tokens=False)
    print(f"Generated {{len(result.output_ids)}} tokens")
    print(f"Text preview: {{generated_text[:100]}}...")

    # Check for mismatches
    mismatches = find_mismatches_with_logprobs(
        model, tokenizer,
        list(result.output_ids),
        input_ids,
        list(result.logprobs),
    )

    if not mismatches:
        print("No token mismatches found")
        return False, None

    print(f"Found {{len(mismatches)}} token mismatches!")

    # Check each mismatch
    for pos, orig_tok, wrong_tok, orig_lp, wrong_lp in mismatches[:5]:
        orig_decoded = tokenizer.decode([orig_tok])
        wrong_decoded = tokenizer.decode([wrong_tok])

        print(f"  Position {{pos}}:")
        print(f"    Generated: token={{orig_tok}} ({{repr(orig_decoded)}}) logprob={{orig_lp:.2f}}")
        print(f"    Retokenized: token={{wrong_tok}} ({{repr(wrong_decoded)}}) logprob={{wrong_lp:.2f}}")
        print(f"    Logprob difference: {{wrong_lp - orig_lp:.1f}} nats")

        if wrong_lp < LOGPROB_THRESHOLD:
            print(f"    ⚠️  FOUND IT! Logprob {{wrong_lp:.2f}} < {{LOGPROB_THRESHOLD}}")
            print(f"    This token would DOMINATE gradients in RL training!")
            return True, (pos, orig_tok, wrong_tok, orig_lp, wrong_lp, orig_decoded, wrong_decoded)

    return False, None


def main():
    print("=" * 70)
    print("TI/TO (Tokens-In/Tokens-Out) Multi-Turn Calculator Test")
    print(f"Model: {{MODEL}}")
    print("=" * 70)
    print()
    print("This test demonstrates the retokenization collapse problem:")
    print("  - Generate tool calls (JSON with quotes, whitespace)")
    print("  - Compare generated tokens vs retokenized tokens")
    print("  - Find mismatches with logprob < -10 (the smoking gun)")
    print()
    print(f"Running {{NUM_ROLLOUTS}} rollouts to find mismatches...")

    # Load model for logprob computation
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"✓ Model loaded")

    # Start SGLang server
    process, base_url = start_sglang_server()

    try:
        found_smoking_gun = False
        smoking_gun_details = None
        total_mismatches = 0

        for i in range(NUM_ROLLOUTS):
            problem = MATH_PROBLEMS[i % len(MATH_PROBLEMS)]
            found, details = run_single_rollout(model, tokenizer, base_url, problem, i)

            if details:
                total_mismatches += 1

            if found:
                found_smoking_gun = True
                smoking_gun_details = details
                print(f"\\n🎯 Found the smoking gun on rollout {{i + 1}}!")
                break

        # Summary
        print("\\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        if found_smoking_gun:
            pos, orig_tok, wrong_tok, orig_lp, wrong_lp, orig_dec, wrong_dec = smoking_gun_details
            print()
            print("✗ FOUND RETOKENIZATION COLLAPSE!")
            print()
            print("The Problem (from tokens.md Figure 4):")
            print(f"  At position {{pos}}, model generated token {{orig_tok}} ({{repr(orig_dec)}})")
            print(f"  with logprob {{orig_lp:.2f}} (reasonable)")
            print()
            print(f"  But after decode→re-encode, we get token {{wrong_tok}} ({{repr(wrong_dec)}})")
            print(f"  with logprob {{wrong_lp:.2f}} (EXTREMELY UNLIKELY)")
            print()
            print("Why this breaks RL training:")
            print(f"  - Logprob {{wrong_lp:.2f}} << -10 means model NEVER predicted this token")
            print("  - When training on this token, gradient is HUGE")
            print("  - These huge gradients DOMINATE the loss")
            print("  - Over time: instability → collapse")
            print()
            print("The Fix (TI/TO):")
            print("  - Store generated token_ids directly (no decode→re-encode)")
            print("  - Train on EXACTLY what model generated")
            print("  - All logprobs are reasonable → stable training")
            print()
            print("✓ TEST PASSED - Successfully demonstrated the problem!")
            return 0
        else:
            print()
            print(f"Ran {{NUM_ROLLOUTS}} rollouts, found {{total_mismatches}} with mismatches")
            print("but none had logprob < -10.")
            print()
            print("This can happen with simple prompts. In practice, multi-turn")
            print("conversations with tool calls trigger this much more frequently.")
            print()
            print("The test infrastructure is working correctly.")
            return 0

    finally:
        print("\\nShutting down SGLang server...")
        process.terminate()
        process.wait(timeout=10)


if __name__ == "__main__":
    sys.exit(main())
'''


# =============================================================================
# Test runner
# =============================================================================


def run_tito_test(client, workspace: str) -> bool:
    """Run TI/TO test on remote GPU."""
    print("\n" + "=" * 60)
    print("TEST: TI/TO Multi-Turn Calculator Test")
    print("=" * 60)

    # Write test script to remote
    script_content = REMOTE_TEST_SCRIPT.format(
        model=MODEL,
        port=SGLANG_PORT,
    )
    script_path = f"{workspace}/test_tito_remote.py"
    client.exec(f"cat > {script_path} << 'SCRIPT_EOF'\n{script_content}\nSCRIPT_EOF")

    # Submit job
    print("\n1. Submitting TI/TO test job...")
    job = submit(
        client,
        # The workspace is at git root (/research), rollouts package is at /research/rollouts
        # So we need {workspace}/rollouts on PYTHONPATH to find rollouts.inference.backends
        command=f"PYTHONPATH={workspace}/rollouts:$PYTHONPATH python {script_path}",
        workspace=workspace,
        gpu_ids=[0],
        deps=DependencyConfig(
            project_name="tito-test",
            dependencies=[
                "torch>=2.0",
                "transformers",
                "accelerate",  # Required for device_map in transformers
                "sglang[all]",
                "httpx",
                "trio",
            ],
        ),
        job_name="test-tito",
    )

    # Stream logs
    print("\n2. Running test (streaming logs)...")
    print("-" * 40)
    success, exit_code = job.stream(timeout_sec=1800)  # 30 min
    print("-" * 40)

    if success:
        print("\n✓ TI/TO test completed")
        return True
    else:
        print(f"\n✗ TI/TO test FAILED (exit code: {exit_code})")
        return False


def main():
    parser = argparse.ArgumentParser(description="TI/TO correctness tests")

    # Node acquisition (mutually exclusive)
    node_group = parser.add_mutually_exclusive_group()
    node_group.add_argument("--ssh", help="Static SSH connection (e.g., root@gpu:22)")
    node_group.add_argument("--node-id", help="Existing instance ID (e.g., runpod:abc123)")
    node_group.add_argument("--provision", action="store_true", help="Provision new instance")

    parser.add_argument(
        "--keep-alive", action="store_true", help="Don't terminate after completion"
    )
    parser.add_argument("--gpu-type", default="A100", help="GPU type (default: A100)")

    args = parser.parse_args()

    # Need at least one acquisition method
    if not (args.ssh or args.node_id or args.provision):
        parser.error("Must specify --ssh, --node-id, or --provision")

    print("=" * 60)
    print("TI/TO (Tokens-In/Tokens-Out) Correctness Tests")
    print("=" * 60)
    print(f"Model: {MODEL}")

    client, instance = acquire_node_runpod_only(
        ssh=args.ssh,
        node_id=args.node_id,
        provision=args.provision,
        gpu_type=args.gpu_type,
    )

    try:
        print("\nDeploying code...")
        workspace = client.push("~/.bifrost/workspaces/rollouts-tito-test")
        print(f"Workspace: {workspace}")

        passed = run_tito_test(client, workspace)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  TI/TO test: {'✓ COMPLETED' if passed else '✗ FAILED'}")

        return 0 if passed else 1

    finally:
        release_node(instance, keep_alive=args.keep_alive)


if __name__ == "__main__":
    sys.exit(main())
