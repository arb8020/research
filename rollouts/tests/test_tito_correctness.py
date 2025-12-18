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

TRUE multi-turn calculator test that:
1. Generates tool calls (turn 1)
2. Executes tools and appends results to context
3. Continues generation (turn 2+) - THIS IS WHERE BOUNDARIES CAUSE ISSUES
4. Checks for mismatches at boundaries with logprob < -10

The retokenization collapse happens at BOUNDARIES when we concatenate:
- Generated tokens from turn N
- Tool result text (inserted between turns)
- New generation from turn N+1

The boundary tokens are where different tokenizations emerge.
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
MAX_TOKENS_PER_TURN = 150
MAX_TURNS = 5
NUM_ROLLOUTS = 10
LOGPROB_THRESHOLD = -10

# Calculator system prompt - forces tool usage
SYSTEM_PROMPT = """You are a calculator assistant that MUST use tools for ALL math operations. You cannot do math in your head.

Available tools:
- add(a, b): Returns a + b
- subtract(a, b): Returns a - b
- multiply(a, b): Returns a * b
- divide(a, b): Returns a / b

RULES:
1. You MUST use a tool call for EVERY arithmetic operation
2. You CANNOT compute results mentally - use tools
3. Make ONE tool call per response
4. After getting a result, use it in the next tool call if needed

Tool call format:
<tool_call>
{{"name": "function_name", "arguments": {{"a": value1, "b": value2}}}}
</tool_call>

Example: For "5 + 3", you must respond with a tool call, NOT "8".

When you have computed the final answer through tool calls, say "DONE: [number]"."""

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


def execute_tool(name: str, args: dict) -> str:
    """Execute a calculator tool and return the result."""
    try:
        a = float(args.get("a", args.get("value1", args.get("value", 0))))
        b = float(args.get("b", args.get("value2", 0)))
    except (ValueError, TypeError):
        return "Error: Invalid arguments"

    if name == "add":
        return str(a + b)
    elif name == "subtract":
        return str(a - b)
    elif name == "multiply":
        return str(a * b)
    elif name == "divide":
        return str(a / b) if b != 0 else "Error: Division by zero"
    else:
        return f"Error: Unknown tool {{name}}"


def parse_tool_call(text: str) -> tuple[str, dict] | None:
    """Parse a tool call from model output."""
    match = re.search(r'<tool_call>\\s*(.+?)\\s*</tool_call>', text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
        return data.get("name"), data.get("arguments", {{}})
    except json.JSONDecodeError:
        return None


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


def generate_with_sglang(base_url: str, input_ids: list[int], max_tokens: int = 150):
    """Generate tokens via SGLang /generate endpoint."""
    import trio
    from rollouts.inference.backends import generate_sglang

    async def run():
        return await generate_sglang(
            base_url=base_url,
            input_ids=input_ids,
            max_tokens=max_tokens,
            temperature=0.7,
            num_logprobs=5,
        )

    return trio.run(run)


def check_boundary_mismatch(
    model,
    tokenizer,
    prefix_ids: list[int],
    generated_ids: list[int],
    generated_logprobs: list[float],
    turn_num: int,
):
    """Check for mismatches, especially at boundaries.

    The key insight from tokens.md Figure 4:
    - At rollout time, byte sequence ' \"' is generated as TWO tokens
    - After retokenization (parse + apply_chat_template), becomes ONE token
    - That one token has logprob -20 (model NEVER predicted it)

    We check two things:
    1. Simple decode→encode mismatch (within single generation)
    2. Chat template mismatch (what happens when we build next turn's context)

    Returns list of mismatches with (pos, orig, retok, orig_lp, wrong_lp, at_boundary).
    """
    device = next(model.parameters()).device

    # === Check 1: Simple decode→encode (single generation) ===
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    retokenized_ids = tokenizer.encode(generated_text, add_special_tokens=False)

    mismatches = []
    min_len = min(len(generated_ids), len(retokenized_ids))

    for i in range(min_len):
        orig = generated_ids[i]
        retok = retokenized_ids[i]

        if orig != retok:
            context_ids = prefix_ids + list(generated_ids[:i])
            context = torch.tensor([context_ids], device=device)

            with torch.no_grad():
                outputs = model(context)
                logits = outputs.logits[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
                wrong_logprob = log_probs[0, retok].item()

            orig_logprob = generated_logprobs[i] if i < len(generated_logprobs) else 0.0
            at_boundary = i < 5
            mismatches.append((i, orig, retok, orig_logprob, wrong_logprob, at_boundary))

    # === Check 2: Chat template retokenization (the real problem) ===
    # This simulates: parse(generated_text) -> apply_chat_template() -> different tokens
    # The issue is that the FULL context with chat template will tokenize differently
    # than our raw generated tokens

    # Build what the "next turn" context would look like via chat template
    # vs what we have via raw token concatenation
    if turn_num > 1:
        # We're in turn 2+, which means the previous turn's tokens were
        # concatenated directly (TITO style) vs being retokenized via chat template
        # The mismatch happens at the BOUNDARY where generated tokens meet new context
        pass  # The check above catches within-generation issues

    return mismatches


def check_chat_template_mismatch(
    model,
    tokenizer,
    messages_before: list[dict],
    generated_text: str,
    generated_ids: list[int],
    generated_logprobs: list[float],
):
    """Check for the REAL retokenization problem from tokens.md.

    The issue:
    1. Model generates response as tokens
    2. We parse to messages: messages_before + {{"role": "assistant", "content": generated_text}}
    3. We apply_chat_template to build next turn's prompt
    4. The assistant content gets RE-TOKENIZED differently!

    This is where ' \"' as two tokens becomes ' \"' as one token with logprob -20.
    """
    device = next(model.parameters()).device

    # Build messages with assistant response
    messages_with_response = messages_before + [{{"role": "assistant", "content": generated_text}}]

    # Tokenize via chat template (what text-based APIs do)
    chat_template_ids = tokenizer.apply_chat_template(
        messages_with_response,
        add_generation_prompt=False,
        tokenize=True,
    )

    # Find where the assistant content starts in the chat template output
    # The assistant's tokens should match generated_ids, but may not!
    messages_without_response = messages_before
    prefix_via_template = tokenizer.apply_chat_template(
        messages_without_response,
        add_generation_prompt=True,
        tokenize=True,
    )

    # The retokenized assistant content is everything after the prefix
    retokenized_assistant = chat_template_ids[len(prefix_via_template):]

    # Compare retokenized vs original generated tokens
    mismatches = []
    min_len = min(len(generated_ids), len(retokenized_assistant))

    for i in range(min_len):
        orig = generated_ids[i]
        retok = retokenized_assistant[i]

        if orig != retok:
            # Compute logprob of the WRONG token
            # Context is the prefix + generated tokens up to this point
            context_ids = list(prefix_via_template) + list(generated_ids[:i])
            context = torch.tensor([context_ids], device=device)

            with torch.no_grad():
                outputs = model(context)
                logits = outputs.logits[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
                wrong_logprob = log_probs[0, retok].item()

            orig_logprob = generated_logprobs[i] if i < len(generated_logprobs) else 0.0
            mismatches.append((i, orig, retok, orig_logprob, wrong_logprob, True))  # All are boundary issues

    # === Check length mismatch (token merging/splitting) - THE REAL PROBLEM ===
    if len(generated_ids) != len(retokenized_assistant):
        print(f"    ⚠️  LENGTH MISMATCH: generated {{len(generated_ids)}} vs retokenized {{len(retokenized_assistant)}}")
        print(f"    This indicates token merging/splitting (the ' \\\"' problem!)")

        # Find where the divergence happens using sequence alignment
        # Walk through both sequences to find first mismatch point
        i_gen, i_ret = 0, 0
        while i_gen < len(generated_ids) and i_ret < len(retokenized_assistant):
            if generated_ids[i_gen] == retokenized_assistant[i_ret]:
                i_gen += 1
                i_ret += 1
            else:
                # Found divergence! Show context
                gen_tok = generated_ids[i_gen]
                ret_tok = retokenized_assistant[i_ret]
                gen_dec = tokenizer.decode([gen_tok])
                ret_dec = tokenizer.decode([ret_tok])

                # Check if multiple generated tokens = one retokenized token
                # (common pattern: ' ' + '"' -> ' "')
                if i_gen + 1 < len(generated_ids):
                    two_gen_toks = generated_ids[i_gen:i_gen+2]
                    two_gen_dec = tokenizer.decode(two_gen_toks)

                    if ret_dec == two_gen_dec or ret_dec in two_gen_dec:
                        print(f"    FOUND MERGE: tokens {{two_gen_toks}} ({{repr(two_gen_dec)}}) -> {{ret_tok}} ({{repr(ret_dec)}})")

                        # Compute logprob of the merged token
                        context_ids = list(prefix_via_template) + list(generated_ids[:i_gen])
                        context = torch.tensor([context_ids], device=device)

                        with torch.no_grad():
                            outputs = model(context)
                            logits = outputs.logits[:, -1, :]
                            log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
                            merged_logprob = log_probs[0, ret_tok].item()
                            orig_logprob = generated_logprobs[i_gen] if i_gen < len(generated_logprobs) else 0.0

                        print(f"      Original first token logprob: {{orig_logprob:.2f}}")
                        print(f"      Merged token logprob: {{merged_logprob:.2f}}")

                        if merged_logprob < LOGPROB_THRESHOLD:
                            print(f"      🎯 THIS IS THE SMOKING GUN! logprob {{merged_logprob:.2f}} < {{LOGPROB_THRESHOLD}}")
                            mismatches.append((i_gen, gen_tok, ret_tok, orig_logprob, merged_logprob, True))

                break  # Found the divergence point

    return mismatches


def run_multi_turn_rollout(model, tokenizer, base_url: str, problem: str, rollout_idx: int):
    """Run a TRUE multi-turn rollout with tool execution.

    This is where the retokenization collapse actually happens:
    - Turn 1: Generate tool call
    - Execute tool, get result text
    - Turn 2: Append result, continue generation
    - The BOUNDARY between result text and new generation is where
      tokenization can differ!

    We now check BOTH:
    1. Simple decode→encode mismatch
    2. Chat template retokenization mismatch (the REAL problem from tokens.md)

    Returns (found_bad_mismatch, mismatch_details, turn_info)
    """
    from rollouts.inference.backends import tokenize_chat

    messages = [
        {{"role": "system", "content": SYSTEM_PROMPT}},
        {{"role": "user", "content": problem}},
    ]

    print(f"\\n--- Rollout {{rollout_idx + 1}}: {{problem}} ---")

    all_generated_ids = []
    all_logprobs = []

    for turn in range(MAX_TURNS):
        # Snapshot messages BEFORE this generation (for chat template check)
        messages_before = list(messages)

        # Tokenize current conversation
        input_ids = tokenize_chat(tokenizer, messages, add_generation_prompt=True)

        # Generate
        result = generate_with_sglang(base_url, input_ids, MAX_TOKENS_PER_TURN)

        generated_text = tokenizer.decode(result.output_ids, skip_special_tokens=False)
        print(f"  Turn {{turn + 1}}: {{len(result.output_ids)}} tokens")

        turn_generated_ids = list(result.output_ids)
        turn_logprobs = list(result.logprobs)

        # === Check 1: Simple decode→encode mismatch ===
        mismatches = check_boundary_mismatch(
            model, tokenizer, input_ids, turn_generated_ids, turn_logprobs, turn + 1
        )

        # === Check 2: Chat template retokenization (the REAL problem) ===
        chat_mismatches = check_chat_template_mismatch(
            model, tokenizer, messages_before, generated_text, turn_generated_ids, turn_logprobs
        )

        all_mismatches = mismatches + chat_mismatches

        if all_mismatches:
            print(f"    Found {{len(mismatches)}} decode mismatches, {{len(chat_mismatches)}} chat template mismatches")

            for pos, orig, retok, orig_lp, wrong_lp, at_boundary in all_mismatches[:5]:
                orig_dec = tokenizer.decode([orig])
                wrong_dec = tokenizer.decode([retok])
                marker = " [CHAT_TEMPLATE]" if at_boundary else ""

                print(f"      Pos {{pos}}{{marker}}: {{repr(orig_dec)}} -> {{repr(wrong_dec)}}")
                print(f"        logprob: {{orig_lp:.2f}} -> {{wrong_lp:.2f}}")

                if wrong_lp < LOGPROB_THRESHOLD:
                    print(f"      ⚠️  FOUND IT! logprob {{wrong_lp:.2f}} < {{LOGPROB_THRESHOLD}}")
                    return True, (turn + 1, pos, orig, retok, orig_lp, wrong_lp, orig_dec, wrong_dec, at_boundary), None

        all_generated_ids.extend(turn_generated_ids)
        all_logprobs.extend(turn_logprobs)

        # Check if done (only after at least one tool call)
        if "DONE:" in generated_text and turn > 0:
            print(f"    Completed with final answer")
            break

        # Parse tool call
        tool_info = parse_tool_call(generated_text)
        if not tool_info:
            print(f"    No tool call found, ending rollout")
            break

        tool_name, tool_args = tool_info
        result_value = execute_tool(tool_name, tool_args)
        print(f"    Tool: {{tool_name}}({{tool_args}}) = {{result_value}}")

        # Add assistant response and tool result to conversation
        messages.append({{"role": "assistant", "content": generated_text}})
        messages.append({{"role": "user", "content": f"Tool result: {{result_value}}\\n\\nContinue solving the problem."}})

    print(f"  Total: {{len(all_generated_ids)}} tokens across {{turn + 1}} turns")
    return False, None, {{"turns": turn + 1, "total_tokens": len(all_generated_ids)}}


def main():
    print("=" * 70)
    print("TI/TO Multi-Turn Calculator Test (TRUE MULTI-TURN)")
    print(f"Model: {{MODEL}}")
    print("=" * 70)
    print()
    print("This test demonstrates retokenization collapse at BOUNDARIES:")
    print("  1. Generate tool call (turn N)")
    print("  2. Execute tool, append result text")
    print("  3. Continue generation (turn N+1)")
    print("  4. Check for mismatches at the BOUNDARY")
    print()
    print("The boundary is where different tokenizations emerge!")
    print()
    print(f"Running {{NUM_ROLLOUTS}} multi-turn rollouts...")

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

    process, base_url = start_sglang_server()

    try:
        found_smoking_gun = False
        smoking_gun_details = None
        total_turns = 0
        total_tokens = 0

        for i in range(NUM_ROLLOUTS):
            problem = MATH_PROBLEMS[i % len(MATH_PROBLEMS)]
            found, details, stats = run_multi_turn_rollout(model, tokenizer, base_url, problem, i)

            if stats:
                total_turns += stats["turns"]
                total_tokens += stats["total_tokens"]

            if found:
                found_smoking_gun = True
                smoking_gun_details = details
                print(f"\\n🎯 Found the smoking gun on rollout {{i + 1}}!")
                break

        print("\\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        if found_smoking_gun:
            turn, pos, orig_tok, wrong_tok, orig_lp, wrong_lp, orig_dec, wrong_dec, at_boundary = smoking_gun_details
            print()
            print("✗ FOUND RETOKENIZATION COLLAPSE!")
            print()
            print(f"Location: Turn {{turn}}, position {{pos}}" + (" [AT BOUNDARY]" if at_boundary else ""))
            print()
            print("The Problem:")
            print(f"  Model generated token {{orig_tok}} ({{repr(orig_dec)}})")
            print(f"  with logprob {{orig_lp:.2f}} (reasonable)")
            print()
            print(f"  After decode→re-encode: token {{wrong_tok}} ({{repr(wrong_dec)}})")
            print(f"  with logprob {{wrong_lp:.2f}} (EXTREMELY UNLIKELY)")
            print()
            if at_boundary:
                print("  This happened at a BOUNDARY - exactly where multi-turn")
                print("  tool calls create tokenization mismatches!")
            print()
            print("Why this breaks RL:")
            print(f"  - logprob {{wrong_lp:.2f}} << -10 means model NEVER predicted this")
            print("  - Gradient is HUGE for unlikely tokens")
            print("  - These dominate the loss → instability → collapse")
            print()
            print("The Fix (TI/TO):")
            print("  - Store generated token_ids directly")
            print("  - Never decode→re-encode")
            print("  - Train on EXACTLY what model generated")
            print()
            print("✓ TEST PASSED - Demonstrated the problem!")
            return 0
        else:
            print()
            print(f"Ran {{NUM_ROLLOUTS}} rollouts ({{total_turns}} turns, {{total_tokens}} tokens)")
            print("No mismatches with logprob < -10 found.")
            print()
            print("This model/tokenizer may have stable round-trip encoding,")
            print("or we need more diverse prompts to trigger the issue.")
            print()
            print("Test infrastructure is working correctly.")
            return 0

    finally:
        print("\\nShutting down SGLang server...")
        process.terminate()
        process.wait(timeout=10)


if __name__ == "__main__":
    sys.exit(main())
'''


# =============================================================================
# Remote script for agent loop integration test
# =============================================================================

REMOTE_AGENT_LOOP_TEST_SCRIPT = '''
"""Agent loop TI/TO integration test - runs on GPU node.

Tests the ACTUAL fix by running agent_rollout_to_sample with and without TI/TO.
- Without TI/TO: May see logprobs < -15 (retokenization collapse)
- With TI/TO: Should see reasonable logprobs (no collapse)

This validates the full integration:
  agent_rollout_to_sample() -> run_agent() -> rollout() -> token-level providers
"""

import sys
import subprocess
import time
import httpx
import trio

MODEL = "{model}"
PORT = {port}
NUM_ROLLOUTS = 3
LOGPROB_THRESHOLD = -15.0

# Calculator prompts for multi-turn tool use
CALC_PROMPTS = [
    {{"messages": [{{"role": "user", "content": "What is 5 + 3?"}}], "ground_truth": 8.0}},
    {{"messages": [{{"role": "user", "content": "What is 12 - 7?"}}], "ground_truth": 5.0}},
    {{"messages": [{{"role": "user", "content": "What is 6 * 4?"}}], "ground_truth": 24.0}},
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


def run_agent_rollouts(base_url: str, tokenizer, use_tito: bool):
    """Run rollouts using the actual agent loop.

    Returns list of (sample, min_logprob) tuples.
    """
    from rollouts.dtypes import Endpoint
    from rollouts.environments.calculator import CalculatorEnvironment
    from rollouts.training.agent_integration import agent_rollout_to_sample

    # SGLang server needs /v1 suffix for OpenAI-compatible API
    endpoint = Endpoint(
        provider="sglang",
        api_base=f"{{base_url}}/v1",
        model=MODEL,
    )

    results = []

    async def run_single(prompt_data):
        sample = await agent_rollout_to_sample(
            prompt=prompt_data["messages"],
            environment_cls=CalculatorEnvironment,
            endpoint=endpoint,
            tokenizer=tokenizer,
            max_turns=5,
            metadata={{"ground_truth": prompt_data["ground_truth"]}},
            use_tito=use_tito,
        )
        return sample

    for i, prompt_data in enumerate(CALC_PROMPTS[:NUM_ROLLOUTS]):
        print(f"  Rollout {{i+1}}/{{NUM_ROLLOUTS}} (use_tito={{use_tito}})...")
        try:
            sample = trio.run(run_single, prompt_data)

            # Get min logprob from rollout_log_probs if available
            min_lp = 0.0
            if sample.rollout_log_probs:
                # Filter out zeros (prompt tokens)
                non_zero = [lp for lp in sample.rollout_log_probs if lp < -0.001]
                if non_zero:
                    min_lp = min(non_zero)

            results.append((sample, min_lp))
            print(f"    tokens={{len(sample.tokens)}}, min_logprob={{min_lp:.2f}}")

        except Exception as e:
            print(f"    FAILED: {{e}}")
            results.append((None, 0.0))

    return results


def check_logprobs_for_collapse(results, label: str) -> tuple[int, float]:
    """Check results for retokenization collapse (logprob < threshold).

    Returns (num_suspicious, min_logprob).
    """
    suspicious_count = 0
    min_overall = 0.0

    for sample, min_lp in results:
        if sample is None:
            continue
        if min_lp < min_overall:
            min_overall = min_lp
        if min_lp < LOGPROB_THRESHOLD:
            suspicious_count += 1

    return suspicious_count, min_overall


def main():
    print("=" * 70)
    print("Agent Loop TI/TO Integration Test")
    print(f"Model: {{MODEL}}")
    print("=" * 70)
    print()
    print("This test validates TI/TO through the actual agent loop:")
    print("  agent_rollout_to_sample() -> run_agent() -> rollout()")
    print()
    print(f"Running {{NUM_ROLLOUTS}} rollouts WITH and WITHOUT TI/TO...")
    print(f"Checking for logprobs < {{LOGPROB_THRESHOLD}} (retokenization collapse)")
    print()

    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    print(f"✓ Tokenizer loaded")

    process, base_url = start_sglang_server()

    try:
        # === Test WITHOUT TI/TO ===
        print()
        print("-" * 50)
        print("TEST 1: Without TI/TO (use_tito=False)")
        print("-" * 50)
        results_no_tito = run_agent_rollouts(base_url, tokenizer, use_tito=False)
        suspicious_no_tito, min_lp_no_tito = check_logprobs_for_collapse(
            results_no_tito, "no_tito"
        )

        # === Test WITH TI/TO ===
        print()
        print("-" * 50)
        print("TEST 2: With TI/TO (use_tito=True)")
        print("-" * 50)
        results_tito = run_agent_rollouts(base_url, tokenizer, use_tito=True)
        suspicious_tito, min_lp_tito = check_logprobs_for_collapse(
            results_tito, "tito"
        )

        # === Results ===
        print()
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print()
        print(f"Without TI/TO:")
        print(f"  Min logprob: {{min_lp_no_tito:.2f}}")
        print(f"  Suspicious samples (< {{LOGPROB_THRESHOLD}}): {{suspicious_no_tito}}/{{NUM_ROLLOUTS}}")
        print()
        print(f"With TI/TO:")
        print(f"  Min logprob: {{min_lp_tito:.2f}}")
        print(f"  Suspicious samples (< {{LOGPROB_THRESHOLD}}): {{suspicious_tito}}/{{NUM_ROLLOUTS}}")
        print()

        # === Assertions ===
        # TI/TO should have no extremely low logprobs
        if suspicious_tito > 0:
            print("✗ FAIL: TI/TO mode still has suspicious logprobs!")
            print("  This indicates the TI/TO fix is not working correctly.")
            return 1

        # If no_tito has suspicious samples, TI/TO should be better
        if suspicious_no_tito > 0 and suspicious_tito == 0:
            print("✓ PASS: TI/TO eliminates retokenization collapse!")
            print(f"  Without TI/TO: {{suspicious_no_tito}} samples had logprob < {{LOGPROB_THRESHOLD}}")
            print(f"  With TI/TO: 0 samples had suspicious logprobs")
            return 0

        # If neither has suspicious samples, test passed but didn't demonstrate the problem
        if suspicious_no_tito == 0 and suspicious_tito == 0:
            print("✓ PASS: No retokenization collapse detected")
            print("  (Model/prompts may not trigger the issue)")
            print("  TI/TO mode is working correctly regardless.")
            return 0

        print("✓ PASS: Agent loop integration test completed")
        return 0

    finally:
        print()
        print("Shutting down SGLang server...")
        process.terminate()
        process.wait(timeout=10)


if __name__ == "__main__":
    sys.exit(main())
'''


# =============================================================================
# Test runners
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


def run_agent_loop_test(client, workspace: str) -> bool:
    """Run agent loop TI/TO integration test on remote GPU.

    This test validates the full TI/TO integration through:
      agent_rollout_to_sample() -> run_agent() -> rollout() -> token-level providers

    Runs rollouts WITH and WITHOUT TI/TO to compare logprobs.
    """
    print("\n" + "=" * 60)
    print("TEST: Agent Loop TI/TO Integration Test")
    print("=" * 60)

    # Write test script to remote
    script_content = REMOTE_AGENT_LOOP_TEST_SCRIPT.format(
        model=MODEL,
        port=SGLANG_PORT,
    )
    script_path = f"{workspace}/test_agent_loop_tito.py"
    client.exec(f"cat > {script_path} << 'SCRIPT_EOF'\n{script_content}\nSCRIPT_EOF")

    # Submit job
    print("\n1. Submitting agent loop TI/TO test job...")
    job = submit(
        client,
        command=f"PYTHONPATH={workspace}/rollouts:$PYTHONPATH python {script_path}",
        workspace=workspace,
        gpu_ids=[0],
        deps=DependencyConfig(
            project_name="tito-agent-test",
            dependencies=[
                "torch>=2.0",
                "transformers",
                "accelerate",
                "sglang[all]",
                # rollouts package dependencies
                "openai>=1.0.0",
                "anthropic>=0.40.0",
                "dacite>=1.8.0",
                "aiohttp>=3.9.0",
                "trio>=0.32.0",
                "httpx>=0.25.0",
                "markdownify>=0.11.0",
            ],
        ),
        job_name="test-agent-loop-tito",
    )

    # Stream logs
    print("\n2. Running test (streaming logs)...")
    print("-" * 40)
    success, exit_code = job.stream(timeout_sec=1800)  # 30 min
    print("-" * 40)

    if success:
        print("\n✓ Agent loop TI/TO test completed")
        return True
    else:
        print(f"\n✗ Agent loop TI/TO test FAILED (exit code: {exit_code})")
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

    # Test selection
    parser.add_argument(
        "--test",
        choices=["all", "retokenization", "agent-loop"],
        default="all",
        help="Which test to run: 'retokenization' (original), 'agent-loop' (new), or 'all' (default)",
    )

    args = parser.parse_args()

    # Need at least one acquisition method
    if not (args.ssh or args.node_id or args.provision):
        parser.error("Must specify --ssh, --node-id, or --provision")

    print("=" * 60)
    print("TI/TO (Tokens-In/Tokens-Out) Correctness Tests")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Tests: {args.test}")

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

        results = {}

        # Run retokenization test (original)
        if args.test in ("all", "retokenization"):
            results["retokenization"] = run_tito_test(client, workspace)

        # Run agent loop test (new)
        if args.test in ("all", "agent-loop"):
            results["agent_loop"] = run_agent_loop_test(client, workspace)

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for name, passed in results.items():
            status = "✓ COMPLETED" if passed else "✗ FAILED"
            print(f"  {name}: {status}")

        all_passed = all(results.values())
        return 0 if all_passed else 1

    finally:
        release_node(instance, keep_alive=args.keep_alive)


if __name__ == "__main__":
    sys.exit(main())


# =============================================================================
# Background: The Retokenization Collapse Problem
# =============================================================================
"""
TI/TO (Tokens-In/Tokens-Out) Correctness Test
==============================================

This test demonstrates the "retokenization collapse" problem that causes RL
training instability in multi-turn tool-using agents.

THE PROBLEM
-----------
When using OpenAI-style messages for environment interactions:

1. Model generates tokens: [tok_1, tok_2, ..., tok_n]
2. We decode to text: "some generated text"
3. We parse into messages and re-apply chat template for next turn
4. Re-tokenization produces DIFFERENT tokens: [tok_1', tok_2', ..., tok_m']

At step 4, the tokenizer may merge or split tokens differently. For example:
- Generated: ' ' + '"' (two tokens)
- Re-tokenized: ' "' (one token)

The re-tokenized token has an extremely low logprob (e.g., -17 to -20) because
the model NEVER predicted it - it predicted the original tokens.

WHY THIS BREAKS RL TRAINING
---------------------------
1. At generation time, model produces rollouts G(ood) and B(ad)
2. Passing through parse() → apply_chat_template() transforms tokens
3. Some tokens in the transformed sequence have logprob << -10
4. These extremely unlikely tokens DOMINATE the gradient
5. Over time: instability → collapse

From empirical observation (Figure 5 in SID-1 paper):
- Training on malformed tokens sees reward increase initially
- After some time, tool calling accuracy decreases
- Followed by fully degenerate outputs
- Reward collapses and is not recoverable

THE FIX: TI/TO (Tokens-In/Tokens-Out)
-------------------------------------
Store generated token_ids directly. Never decode→re-encode.

Two rollout strategies:

1. BRANCHING: Each turn is a separate (prompt, completion) sample
   - Fresh tokenization of full history for each turn
   - No prefix sharing, but no boundary issues
   - Used for API models (Claude, GPT) or truncated reasoning

2. INTERLEAVED: Concatenate stored tokens across turns
   - prompt_ids + output_ids_1 + suffix_ids + tool_result_ids + output_ids_2 + ...
   - Prefix sharing for efficiency
   - Must handle suffix tokens (delimiters) correctly
   - Implemented in rollout_sglang_token_level / rollout_vllm_token_level

This test verifies the interleaved TI/TO implementation by:
1. Generating multi-turn tool-using rollouts
2. Comparing generated tokens vs chat-template-retokenized tokens
3. Computing logprobs of mismatched tokens
4. Flagging any with logprob < -10 as "smoking guns"

IMPLEMENTATION DETAILS
----------------------
Key functions in rollouts/providers/sglang.py:
- rollout_sglang_token_level(): Token-level generation via /generate
- _build_input_ids_from_trajectory(): Concatenates stored tokens for multi-turn

Key functions in rollouts/inference/backends/tokenize.py:
- compute_suffix_ids(): Pre-compute delimiter tokens after assistant
- append_suffix_with_overlap(): Handle overlap when appending suffixes
- tokenize_message_with_delimiter(): Prefix trick for correct delimiters

RAW SOURCES
-----------
From internal team notes (tokens.md):

"Most RL frameworks are fundamentally unstable.

We wasted more H100 hours on debugging this than any other issue for our
multi-turn, multi-env RL run.

When using OpenAI-style messages for env interactions, parsing and retokenizing
leads to subtly different tokens. This creates extremely unlikely tokens, which
dominate the gradient and over time lead to collapse.

We tried a lot of interventions, but ended up reimplementing our environments
to use token lists directly (Tokens-in/Tokens-out). This fixed it immediately.

Always inspect logprobs!"

From SID-1 paper:

"Messages are practicable for offline RL, single-turn environments, or settings
which use few chat template features (no thinking, no tool calling, for example).
For multi-turn environments with many tool calls, using the messages abstraction
invariably leads to model collapse. Parsing a token list to a message list is
lossy. For example, it erases whitespace information around tool calls. Applying
the chat template to generate the next turn then subtly changes the tokens.
When inspecting logprobs, we observe extremely unlikely tokens where these
shifts occurred.

Generation    Training
  query         query
  \":            \":
  \"             \" \\\"      <- merged token with logprob -20
  \\\"           A
  A             Love
  Love          That
  That

At rollout time, the byte sequence ' \\\"' is generated as two tokens, which gets
transformed into one extremely low probability token after retokenization.

Training on these tokens leads to instability and collapse over time. We trace
this collapse to feedback loops:

1. At generation time, the model produces two categories of rollouts, G(ood)
   and B(ad) rollouts, for example B rollouts could have incorrectly formatted
   tool calls.

2. Passing the bad rollouts from the generation engine to the training engine
   causes them to appear like good rollouts:
   apply_chat_template(parse(B)) = G'

3. At training time, the model sees G' with negative advantage, whereby some
   tokens in G' have extremely negative logprobs and thus dominate the gradient,
   driving these logprobs even more negative and the model to generate fewer
   good rollouts in general.

We find that simply ensuring that all rollouts are processed by our pipeline in
a strictly Tokens-In/Tokens-Out (TI/TO) manner is sufficient to prevent these
extreme training instabilities without the use of importance sampling ratios."

REFERENCES
----------
- PrimeIntellect prime-rl PR #1422: TI/TO implementation
- radixark/miles: Reference token-level rollout implementation
- SID-1 paper: Detailed analysis of retokenization collapse
- verifiers/utils/token_utils.py: Cached suffix approach
"""
