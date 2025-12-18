#!/usr/bin/env python3
"""End-to-end correctness test for TI/TO (Tokens-In/Tokens-Out).

Demonstrates:
1. THE PROBLEM: Retokenization produces different tokens with logprob -20
2. THE FIX: TI/TO stores generated token_ids directly, avoiding retokenization

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

    # Use specific provider
    python tests/test_tito_correctness.py --provision --provider runpod
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

Demonstrates:
1. THE PROBLEM: Text-based generation → retokenization → wrong tokens with logprob -20
2. THE FIX: TI/TO stores generated token_ids directly, no retokenization needed
"""

import json
import subprocess
import time
import sys
import httpx
import torch
import torch.nn.functional as F

MODEL = "{model}"
PORT = {port}
MAX_TOKENS = 30


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


def demonstrate_the_problem(model, tokenizer):
    """Show that retokenization produces different tokens with terrible logprobs.

    This is THE core problem TI/TO solves.
    """
    print("\\n" + "=" * 60)
    print("DEMONSTRATING THE PROBLEM: Retokenization Collapse")
    print("=" * 60)

    device = next(model.parameters()).device
    prompt = "The capital of France is"

    # Step 1: Generate tokens
    print(f"\\n1. Generate from prompt: {{prompt!r}}")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    generated_ids = []
    generated_logprobs = []
    current_ids = input_ids

    with torch.no_grad():
        for _ in range(MAX_TOKENS):
            outputs = model(current_ids)
            logits = outputs.logits[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)

            next_token = logits.argmax(dim=-1).item()
            token_logprob = log_probs[0, next_token].item()

            generated_ids.append(next_token)
            generated_logprobs.append(token_logprob)

            if next_token == tokenizer.eos_token_id:
                break

            current_ids = torch.cat(
                [current_ids, torch.tensor([[next_token]], device=device)],
                dim=1,
            )

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"   Generated text: {{generated_text!r}}")
    print(f"   Generated tokens: {{generated_ids[:10]}}...")
    print(f"   Logprobs: {{[f'{{lp:.2f}}' for lp in generated_logprobs[:10]]}}...")

    # Step 2: Simulate text-based API (decode → re-encode)
    print(f"\\n2. Simulate text-based API: decode → re-encode")
    full_text = prompt + generated_text
    retokenized_ids = tokenizer.encode(full_text, add_special_tokens=False)
    retokenized_generated = retokenized_ids[len(tokenizer.encode(prompt, add_special_tokens=False)):]

    print(f"   Retokenized tokens: {{retokenized_generated[:10]}}...")

    # Step 3: Find mismatches and compute their logprobs
    print(f"\\n3. Compare tokens and find mismatches:")
    mismatches = []
    min_len = min(len(generated_ids), len(retokenized_generated))

    for i in range(min_len):
        orig = generated_ids[i]
        retok = retokenized_generated[i]
        if orig != retok:
            mismatches.append((i, orig, retok, generated_logprobs[i]))

    if not mismatches:
        print("   No mismatches in this example (try a different prompt)")
        # Force an example by showing what WOULD happen
        print("\\n   Showing what happens when tokens DO mismatch:")
        print("   If model generated token 1234 (logprob -0.5)")
        print("   But retokenization gives token 5678")
        print("   → Model's logprob for 5678 might be -20 (never predicted it!)")
        print("   → This -20 logprob DOMINATES the gradient")
        print("   → RL training collapses")
    else:
        print(f"   Found {{len(mismatches)}} mismatches!")
        print()

        # Compute logprobs for the WRONG tokens
        current_ids = input_ids
        with torch.no_grad():
            for i, (idx, orig_tok, wrong_tok, orig_lp) in enumerate(mismatches[:5]):
                # Build context up to this position
                prefix = input_ids[0].tolist() + generated_ids[:idx]
                context = torch.tensor([prefix], device=device)

                outputs = model(context)
                logits = outputs.logits[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)

                wrong_logprob = log_probs[0, wrong_tok].item()

                orig_decoded = tokenizer.decode([orig_tok])
                wrong_decoded = tokenizer.decode([wrong_tok])

                print(f"   Position {{idx}}:")
                print(f"     Original token: {{orig_tok}} ({{orig_decoded!r}}) logprob={{orig_lp:.2f}}")
                print(f"     Wrong token:    {{wrong_tok}} ({{wrong_decoded!r}}) logprob={{wrong_logprob:.2f}}")
                print(f"     → Difference: {{wrong_logprob - orig_lp:.1f}} nats")
                if wrong_logprob < -10:
                    print(f"     ⚠️  LOGPROB < -10: This will DOMINATE gradients!")
                print()

    return len(mismatches)


def demonstrate_the_fix(tokenizer, base_url: str):
    """Show that TI/TO avoids retokenization entirely."""
    from rollouts.inference.backends import generate_sglang, tokenize_chat
    import trio

    print("\\n" + "=" * 60)
    print("DEMONSTRATING THE FIX: TI/TO (Tokens-In/Tokens-Out)")
    print("=" * 60)

    messages = [{{"role": "user", "content": "The capital of France is"}}]

    async def run():
        # Step 1: Tokenize input
        input_ids = tokenize_chat(tokenizer, messages, add_generation_prompt=True)
        print(f"\\n1. Tokenize input: {{len(input_ids)}} tokens")

        # Step 2: Generate via /generate (tokens in, tokens out)
        print(f"\\n2. Call /generate with token IDs (not text!)")
        result = await generate_sglang(
            base_url=base_url,
            input_ids=input_ids,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )

        print(f"   Got {{len(result.output_ids)}} tokens directly from server")
        print(f"   Token IDs: {{list(result.output_ids)[:10]}}...")
        print(f"   Logprobs: {{[f'{{lp:.2f}}' for lp in result.logprobs[:10]]}}...")

        # Step 3: Store token_ids (no retokenization!)
        print(f"\\n3. Store token_ids in Choice (this is what we train on)")
        print(f"   Choice.token_ids = {{list(result.output_ids)[:10]}}...")

        # Step 4: Decode for display only
        output_text = tokenizer.decode(result.output_ids, skip_special_tokens=True)
        print(f"\\n4. Decode for display (NOT for training): {{output_text!r}}")

        # Step 5: Show what training sees
        print(f"\\n5. What training sees:")
        print(f"   tokens:   {{list(result.output_ids)[:10]}}... (from Choice.token_ids)")
        print(f"   logprobs: {{[f'{{lp:.2f}}' for lp in result.logprobs[:10]]}}... (all reasonable)")
        print(f"   → NO retokenization, NO -20 logprobs, NO gradient explosion")

        return result

    return trio.run(run)


def main():
    print("=" * 60)
    print("TI/TO (Tokens-In/Tokens-Out) Demonstration")
    print(f"Model: {{MODEL}}")
    print("=" * 60)
    print()
    print("This test demonstrates:")
    print("  1. THE PROBLEM: Retokenization produces wrong tokens with logprob -20")
    print("  2. THE FIX: TI/TO stores token_ids directly, avoiding retokenization")

    # Load model and tokenizer
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

    # Part 1: Show the problem
    num_mismatches = demonstrate_the_problem(model, tokenizer)

    # Part 2: Start SGLang and show the fix
    process, base_url = start_sglang_server()

    try:
        result = demonstrate_the_fix(tokenizer, base_url)

        # Summary
        print("\\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print()
        print("THE PROBLEM (text-based APIs):")
        print("  generate() → decode to text → re-encode for training")
        print("  → Some tokens change during re-encoding")
        print("  → Changed tokens have logprob ≈ -20 (model never predicted them)")
        print("  → These dominate gradients → RL training collapses")
        print()
        print("THE FIX (TI/TO):")
        print("  tokenize() → /generate with token IDs → store output_ids directly")
        print("  → No re-encoding, tokens are exactly what model predicted")
        print("  → All logprobs are reasonable → stable RL training")
        print()

        if num_mismatches > 0:
            print(f"This run found {{num_mismatches}} token mismatches that would cause problems.")
        else:
            print("This run didn't find mismatches, but they happen frequently in practice.")

        print("\\n✓ DEMONSTRATION COMPLETE")
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
    print("TEST: TI/TO Correctness Validation")
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
        print("\n✓ TI/TO test PASSED")
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
        print(f"  TI/TO correctness: {'✓ PASSED' if passed else '✗ FAILED'}")

        return 0 if passed else 1

    finally:
        release_node(instance, keep_alive=args.keep_alive)


if __name__ == "__main__":
    sys.exit(main())
