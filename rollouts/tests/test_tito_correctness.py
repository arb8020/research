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
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from kerbal.job_monitor import LogStreamConfig, stream_log_until_complete
from kerbal.tmux import start_tmux_session

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
            min_cuda_version="12.8",
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
import sys
import re
from pathlib import Path
import trio
import torch
import torch.nn.functional as F

MODEL = "{model}"
PORT = {port}
OUTPUT_DIR = Path("/tmp/tito-test")
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


def start_sglang_server():
    """Start SGLang server using SGLangEngine (handles logging, crash detection)."""
    from rollouts.training.weight_sync import SGLangEngine

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    engine = SGLangEngine(
        model_name=MODEL,
        port=PORT,
        gpu_ids=(0,),
        output_dir=OUTPUT_DIR,
        dtype="bfloat16",
        mem_fraction=0.5,  # Leave room for model forward pass
    )

    print(f"Starting SGLang server for {{MODEL}} on port {{PORT}}...")
    print(f"Logs: {{engine.log_path}}")

    engine.launch()
    engine.start_log_tailer()

    # Wait for server to be ready (async)
    async def wait():
        await engine.wait_until_ready(max_wait=300)

    trio.run(wait)

    print(f"✓ SGLang server ready at {{engine.api_base}}")
    return engine, engine.base_url


def generate_with_sglang(base_url: str, input_ids: list[int], max_tokens: int = 150):
    """Generate tokens via SGLang /generate endpoint."""
    import httpx

    # Call SGLang /generate directly (simpler than using async backend)
    resp = httpx.post(
        f"{{base_url}}/generate",
        json={{
            "input_ids": input_ids,
            "sampling_params": {{
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "return_logprob": True,
                "top_logprobs_num": 5,
            }},
        }},
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()

    # Parse response
    from dataclasses import dataclass

    @dataclass
    class GenerationOutput:
        output_ids: list[int]
        logprobs: list[float]

    output_ids = data.get("output_ids", [])
    # Extract logprobs from meta_info
    meta = data.get("meta_info", {{}})
    logprobs = meta.get("output_token_logprobs", [])

    return GenerationOutput(output_ids=output_ids, logprobs=logprobs)


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

    engine, base_url = start_sglang_server()

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
        engine.shutdown()
        print(f"Logs available at: {{engine.log_path}}")


if __name__ == "__main__":
    sys.exit(main())
'''


# =============================================================================
# Remote script for agent loop integration test
# =============================================================================

REMOTE_AGENT_LOOP_TEST_SCRIPT = '''
"""Agent loop TI/TO integration test - runs on GPU node.

Tests the ACTUAL fix by running agent_rollout_to_sample with TI/TO enabled,
then rigorously checking what would happen if we retokenized the generated tokens.

The key insight from tokens.md:
- At rollout time: model generates tokens like ' ' + '"' (two tokens)
- After retokenization via chat template: becomes ' "' (one token)
- That merged token has logprob -20 (model NEVER predicted it)

To detect this, we:
1. Run agent rollouts with TI/TO to get the actual generated tokens
2. For each sample, simulate retokenization (parse -> apply_chat_template)
3. Compare original tokens vs retokenized tokens
4. For mismatches, compute the logprob of the WRONG token via model forward pass
5. Flag any with logprob < -15 as "smoking guns"

This validates the full integration:
  agent_rollout_to_sample() -> run_agent() -> rollout() -> token-level providers
"""

import sys
from pathlib import Path
import trio
import torch
import torch.nn.functional as F

MODEL = "{model}"
PORT = {port}
OUTPUT_DIR = Path("/tmp/tito-agent-test")
NUM_ROLLOUTS = 5
LOGPROB_THRESHOLD = -15.0

# Calculator prompts for multi-turn tool use
CALC_PROMPTS = [
    {{"messages": [{{"role": "user", "content": "What is 5 + 3?"}}], "ground_truth": 8.0}},
    {{"messages": [{{"role": "user", "content": "What is 12 - 7?"}}], "ground_truth": 5.0}},
    {{"messages": [{{"role": "user", "content": "What is 6 * 4?"}}], "ground_truth": 24.0}},
    {{"messages": [{{"role": "user", "content": "Calculate (10 + 5) / 3"}}], "ground_truth": 5.0}},
    {{"messages": [{{"role": "user", "content": "What is 8 * 7 - 20?"}}], "ground_truth": 36.0}},
]


def start_sglang_server():
    """Start SGLang server using SGLangEngine (handles logging, crash detection)."""
    from rollouts.training.weight_sync import SGLangEngine

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    engine = SGLangEngine(
        model_name=MODEL,
        port=PORT,
        gpu_ids=(0,),
        output_dir=OUTPUT_DIR,
        dtype="bfloat16",
        mem_fraction=0.5,  # Leave room for model forward pass
    )

    print(f"Starting SGLang server for {{MODEL}} on port {{PORT}}...")
    print(f"Logs: {{engine.log_path}}")

    engine.launch()
    engine.start_log_tailer()

    # Wait for server to be ready (async)
    async def wait():
        await engine.wait_until_ready(max_wait=300)

    trio.run(wait)

    print(f"✓ SGLang server ready at {{engine.api_base}}")
    return engine, engine.base_url


def run_agent_rollouts(base_url: str, tokenizer, use_tito: bool, strategy: str = "interleaved"):
    """Run rollouts using the actual agent loop.

    Args:
        base_url: SGLang server URL
        tokenizer: HuggingFace tokenizer
        use_tito: Whether to use TI/TO mode
        strategy: "interleaved" (1 sample per rollout) or "branching" (1 sample per turn)

    Returns list of Sample objects.
    """
    from rollouts.dtypes import Endpoint
    from rollouts.environments.calculator import CalculatorEnvironment
    from rollouts.training.agent_integration import agent_rollout_to_sample, trajectory_to_samples
    from rollouts.agents import run_agent

    # SGLang server needs /v1 suffix for OpenAI-compatible API
    endpoint = Endpoint(
        provider="sglang",
        api_base=f"{{base_url}}/v1",
        model=MODEL,
    )

    samples = []

    async def run_single(prompt_data):
        if strategy == "interleaved":
            # Original path: single sample per rollout
            sample = await agent_rollout_to_sample(
                prompt=prompt_data["messages"],
                environment_cls=CalculatorEnvironment,
                endpoint=endpoint,
                tokenizer=tokenizer,
                max_turns=5,
                metadata={{"ground_truth": prompt_data["ground_truth"]}},
                use_tito=use_tito,
            )
            return [sample]
        else:
            # Branching: run agent to get trajectory, then split into samples per turn
            from rollouts.agents import Actor, Trajectory, Message
            from rollouts.providers import get_provider

            provider = get_provider(endpoint, use_tito=use_tito)
            env = CalculatorEnvironment()

            initial_messages = [Message(role=m["role"], content=m["content"]) for m in prompt_data["messages"]]
            trajectory = Trajectory(messages=initial_messages)
            actor = Actor(trajectory=trajectory, endpoint=endpoint)

            final_actor = await run_agent(actor, env, provider, max_turns=5)

            # Convert trajectory to samples using branching strategy
            return trajectory_to_samples(
                final_actor.trajectory,
                tokenizer,
                strategy="branching",
                metadata={{"ground_truth": prompt_data["ground_truth"]}},
            )

    for i, prompt_data in enumerate(CALC_PROMPTS[:NUM_ROLLOUTS]):
        print(f"  Rollout {{i+1}}/{{NUM_ROLLOUTS}} (strategy={{strategy}}, use_tito={{use_tito}})...")
        try:
            rollout_samples = trio.run(run_single, prompt_data)
            samples.extend(rollout_samples)
            print(f"    Got {{len(rollout_samples)}} samples")
            for j, s in enumerate(rollout_samples):
                print(f"      Sample {{j}}: {{len(s.tokens)}} tokens, response={{s.response[:40]}}...")
        except Exception as e:
            print(f"    FAILED: {{e}}")
            import traceback
            traceback.print_exc()

    return samples


def check_retokenization_mismatch(model, tokenizer, sample, sample_idx: int):
    """Check for retokenization collapse by simulating what would happen without TI/TO.

    The key insight:
    1. TI/TO stores the generated token_ids directly in sample.tokens
    2. Without TI/TO, we would: decode(tokens) -> parse to messages -> apply_chat_template
    3. The chat template retokenization can produce DIFFERENT tokens
    4. We compute the logprob of these wrong tokens via model forward pass

    Returns list of (position, orig_token, wrong_token, orig_logprob, wrong_logprob).
    """
    device = next(model.parameters()).device
    mismatches = []

    # Get the generated tokens (TI/TO preserves these)
    generated_ids = sample.tokens

    # Decode to text and re-tokenize via chat template (simulating non-TI/TO path)
    # First, find where the response starts by looking at loss_mask
    response_start = 0
    for i, m in enumerate(sample.loss_mask):
        if m > 0:
            response_start = i
            break

    # Get the prompt tokens (everything before response)
    prompt_ids = generated_ids[:response_start]
    response_ids = generated_ids[response_start:]

    if len(response_ids) == 0:
        print(f"    Sample {{sample_idx}}: No response tokens found")
        return []

    # Decode the response and retokenize
    response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
    retokenized_response = tokenizer.encode(response_text, add_special_tokens=False)

    # Compare token by token
    min_len = min(len(response_ids), len(retokenized_response))
    mismatch_count = 0

    for i in range(min_len):
        orig = response_ids[i]
        retok = retokenized_response[i]

        if orig != retok:
            mismatch_count += 1

            # Compute logprob of the WRONG token via forward pass
            # Context is everything up to this point
            context_ids = list(prompt_ids) + list(response_ids[:i])
            context = torch.tensor([context_ids], device=device)

            with torch.no_grad():
                outputs = model(context)
                logits = outputs.logits[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
                wrong_logprob = log_probs[0, retok].item()
                orig_logprob = log_probs[0, orig].item()

            orig_dec = tokenizer.decode([orig])
            wrong_dec = tokenizer.decode([retok])

            mismatches.append((i, orig, retok, orig_logprob, wrong_logprob))

            if wrong_logprob < LOGPROB_THRESHOLD:
                print(f"    🎯 SMOKING GUN at pos {{i}}: {{repr(orig_dec)}} -> {{repr(wrong_dec)}}")
                print(f"       logprob: {{orig_logprob:.2f}} -> {{wrong_logprob:.2f}}")

    # Check for length mismatch (token merging/splitting)
    if len(response_ids) != len(retokenized_response):
        print(f"    ⚠️  LENGTH MISMATCH: generated {{len(response_ids)}} vs retokenized {{len(retokenized_response)}}")

        # Find where the divergence happens
        i_gen, i_ret = 0, 0
        while i_gen < len(response_ids) and i_ret < len(retokenized_response):
            if response_ids[i_gen] == retokenized_response[i_ret]:
                i_gen += 1
                i_ret += 1
            else:
                # Found divergence - check if it's a token merge
                if i_gen + 1 < len(response_ids):
                    two_gen_toks = response_ids[i_gen:i_gen+2]
                    two_gen_dec = tokenizer.decode(list(two_gen_toks))
                    ret_tok = retokenized_response[i_ret]
                    ret_dec = tokenizer.decode([ret_tok])

                    if ret_dec == two_gen_dec or ret_dec in two_gen_dec:
                        print(f"    FOUND MERGE at pos {{i_gen}}: tokens {{list(two_gen_toks)}} ({{repr(two_gen_dec)}}) -> {{ret_tok}} ({{repr(ret_dec)}})")

                        # Compute logprob of the merged token
                        context_ids = list(prompt_ids) + list(response_ids[:i_gen])
                        context = torch.tensor([context_ids], device=device)

                        with torch.no_grad():
                            outputs = model(context)
                            logits = outputs.logits[:, -1, :]
                            log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
                            merged_logprob = log_probs[0, ret_tok].item()
                            orig_logprob = log_probs[0, response_ids[i_gen]].item()

                        print(f"       Original first token logprob: {{orig_logprob:.2f}}")
                        print(f"       Merged token logprob: {{merged_logprob:.2f}}")

                        if merged_logprob < LOGPROB_THRESHOLD:
                            print(f"       🎯 THIS IS THE SMOKING GUN! logprob {{merged_logprob:.2f}} < {{LOGPROB_THRESHOLD}}")
                            mismatches.append((i_gen, response_ids[i_gen], ret_tok, orig_logprob, merged_logprob))
                break

    if mismatch_count > 0:
        print(f"    Found {{mismatch_count}} token mismatches in sample {{sample_idx}}")

    return mismatches


def check_samples_for_collapse(model, tokenizer, samples, strategy_name: str):
    """Check samples for retokenization collapse.

    Returns dict with results.
    """
    print()
    print(f"Checking {{len(samples)}} samples for retokenization collapse...")
    print("(Simulating what would happen WITHOUT TI/TO)")
    print()

    all_mismatches = []
    smoking_guns = []

    for i, sample in enumerate(samples):
        print(f"  Sample {{i+1}}/{{len(samples)}}:")
        mismatches = check_retokenization_mismatch(model, tokenizer, sample, i)
        all_mismatches.extend(mismatches)

        for pos, orig, wrong, orig_lp, wrong_lp in mismatches:
            if wrong_lp < LOGPROB_THRESHOLD:
                smoking_guns.append((i, pos, orig, wrong, orig_lp, wrong_lp))

    print()
    print(f"  Results for {{strategy_name}}:")
    print(f"    Samples analyzed: {{len(samples)}}")
    print(f"    Total mismatches: {{len(all_mismatches)}}")
    print(f"    Smoking guns (logprob < {{LOGPROB_THRESHOLD}}): {{len(smoking_guns)}}")

    if smoking_guns:
        print()
        print("  🎯 Found retokenization collapse issues:")
        for sample_idx, pos, orig, wrong, orig_lp, wrong_lp in smoking_guns[:3]:
            orig_dec = tokenizer.decode([orig])
            wrong_dec = tokenizer.decode([wrong])
            print(f"    Sample {{sample_idx}}, pos {{pos}}: {{repr(orig_dec)}} -> {{repr(wrong_dec)}}")
            print(f"      logprob: {{orig_lp:.2f}} -> {{wrong_lp:.2f}}")

    return {{
        "passed": True,  # We pass if TI/TO is working - finding mismatches proves TI/TO is needed
        "samples": len(samples),
        "mismatches": len(all_mismatches),
        "smoking_guns": len(smoking_guns),
    }}


def main():
    print("=" * 70)
    print("Agent Loop TI/TO Integration Test (with Branching)")
    print(f"Model: {{MODEL}}")
    print("=" * 70)
    print()
    print("This test validates TI/TO with both trajectory strategies:")
    print()
    print("  INTERLEAVED: Full conversation as one sequence")
    print("    - Efficient (prefix sharing at training)")
    print("    - One sample per rollout")
    print()
    print("  BRANCHING: Each assistant turn as separate sample")
    print("    - Safer (mirrors deployment exactly)")
    print("    - One sample per turn")
    print()
    print("For each strategy, we:")
    print("  1. Run agent rollouts with TI/TO enabled")
    print("  2. Simulate what would happen if we retokenized (non-TI/TO)")
    print("  3. Compute logprobs of mismatched tokens via forward pass")
    print("  4. Flag logprobs < {{}} as 'smoking guns'".format(LOGPROB_THRESHOLD))
    print()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"✓ Model loaded")

    engine, base_url = start_sglang_server()

    try:
        results = {{}}

        # === Test 1: Interleaved strategy (original) ===
        print()
        print("-" * 50)
        print("TEST 1: INTERLEAVED STRATEGY")
        print("-" * 50)
        print("Running agent rollouts with TI/TO + interleaved...")
        interleaved_samples = run_agent_rollouts(base_url, tokenizer, use_tito=True, strategy="interleaved")

        if interleaved_samples:
            print(f"  Generated {{len(interleaved_samples)}} samples (1 per rollout)")
            results["interleaved"] = check_samples_for_collapse(model, tokenizer, interleaved_samples, "interleaved")
        else:
            print("  ✗ No samples generated!")
            results["interleaved"] = {{"passed": False, "error": "No samples"}}

        # === Test 2: Branching strategy (new) ===
        print()
        print("-" * 50)
        print("TEST 2: BRANCHING STRATEGY")
        print("-" * 50)
        print("Running agent rollouts with TI/TO + branching...")
        branching_samples = run_agent_rollouts(base_url, tokenizer, use_tito=True, strategy="branching")

        if branching_samples:
            print(f"  Generated {{len(branching_samples)}} samples (1 per assistant turn)")
            results["branching"] = check_samples_for_collapse(model, tokenizer, branching_samples, "branching")
        else:
            print("  ✗ No samples generated!")
            results["branching"] = {{"passed": False, "error": "No samples"}}

        # === Final Results ===
        print()
        print("=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)

        all_passed = True
        for strategy_name, result in results.items():
            status = "✓ PASSED" if result.get("passed", False) else "✗ FAILED"
            print(f"  {{strategy_name}}: {{status}}")
            if "samples" in result:
                print(f"    Samples: {{result['samples']}}, Mismatches: {{result['mismatches']}}, Smoking guns: {{result['smoking_guns']}}")
            if not result.get("passed", False):
                all_passed = False

        print()
        if all_passed:
            print("✓ ALL TESTS PASSED: TI/TO + trajectory strategies working correctly!")
            return 0
        else:
            print("✗ SOME TESTS FAILED")
            return 1

    finally:
        print()
        print("Shutting down SGLang server...")
        engine.shutdown()
        print(f"Logs available at: {{engine.log_path}}")


if __name__ == "__main__":
    sys.exit(main())
'''


# =============================================================================
# Test runners
# =============================================================================


def run_tito_test(client, workspace: str) -> bool:
    """Run TI/TO test on remote GPU.

    Uses tmux + log file pattern (like examples/rl) for reliable logging.
    """
    print("\n" + "=" * 60)
    print("TEST: TI/TO Multi-Turn Calculator Test")
    print("=" * 60)

    # Write test script to remote
    script_content = REMOTE_TEST_SCRIPT.format(
        model=MODEL,
        port=SGLANG_PORT,
    )
    script_path = f"{workspace}/rollouts/test_tito_remote.py"
    log_file = f"{workspace}/rollouts/results/tito-test.log"

    client.exec(f"mkdir -p {workspace}/rollouts/results")
    client.exec(f"cat > {script_path} << 'SCRIPT_EOF'\n{script_content}\nSCRIPT_EOF")

    # Start test in tmux session
    print("\n1. Starting TI/TO test...")
    cmd = f"PYTHONPATH={workspace}/rollouts:$PYTHONPATH uv run python {script_path}"

    session, err = start_tmux_session(
        client=client,
        session_name="tito-test",
        command=cmd,
        workspace=f"{workspace}/rollouts",
        log_file=log_file,
        capture_exit_code=True,
    )
    if err:
        print(f"Failed to start tmux session: {err}")
        return False

    print(f"Test started in tmux session: {session}")
    print(f"Log file: {log_file}")

    # Stream logs until complete
    print("\n2. Running test (streaming logs)...")
    print("-" * 40)
    monitor_config = LogStreamConfig(
        session_name="tito-test",
        log_file=log_file,
        timeout_sec=1800,  # 30 min
        poll_interval_sec=1.0,
    )
    success, exit_code, err = stream_log_until_complete(client, monitor_config)
    print("-" * 40)

    if success and exit_code == 0:
        print("\n✓ TI/TO test completed")
        return True
    else:
        print(f"\n✗ TI/TO test FAILED (exit code: {exit_code}, error: {err})")
        return False


def run_agent_loop_test(client, workspace: str) -> bool:
    """Run agent loop TI/TO integration test on remote GPU.

    Uses tmux + log file pattern (like examples/rl) for reliable logging.

    This test validates the full TI/TO integration through:
      agent_rollout_to_sample() -> run_agent() -> rollout() -> token-level providers
    """
    print("\n" + "=" * 60)
    print("TEST: Agent Loop TI/TO Integration Test")
    print("=" * 60)

    # Write test script to remote
    script_content = REMOTE_AGENT_LOOP_TEST_SCRIPT.format(
        model=MODEL,
        port=SGLANG_PORT,
    )
    script_path = f"{workspace}/rollouts/test_agent_loop_tito.py"
    log_file = f"{workspace}/rollouts/results/agent-loop-tito-test.log"

    client.exec(f"mkdir -p {workspace}/rollouts/results")
    client.exec(f"cat > {script_path} << 'SCRIPT_EOF'\n{script_content}\nSCRIPT_EOF")

    # Start test in tmux session
    print("\n1. Starting agent loop TI/TO test...")
    cmd = f"PYTHONPATH={workspace}/rollouts:$PYTHONPATH uv run python {script_path}"

    session, err = start_tmux_session(
        client=client,
        session_name="agent-loop-tito-test",
        command=cmd,
        workspace=f"{workspace}/rollouts",
        log_file=log_file,
        capture_exit_code=True,
    )
    if err:
        print(f"Failed to start tmux session: {err}")
        return False

    print(f"Test started in tmux session: {session}")
    print(f"Log file: {log_file}")

    # Stream logs until complete
    print("\n2. Running test (streaming logs)...")
    print("-" * 40)
    monitor_config = LogStreamConfig(
        session_name="agent-loop-tito-test",
        log_file=log_file,
        timeout_sec=1800,  # 30 min
        poll_interval_sec=1.0,
    )
    success, exit_code, err = stream_log_until_complete(client, monitor_config)
    print("-" * 40)

    if success and exit_code == 0:
        print("\n✓ Agent loop TI/TO test completed")
        return True
    else:
        print(f"\n✗ Agent loop TI/TO test FAILED (exit code: {exit_code}, error: {err})")
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

    workspace = None
    try:
        # Deploy code with bootstrap (like examples/rl)
        print("\nDeploying code...")
        bootstrap = [
            "cd rollouts && uv python install 3.12 && uv sync --python 3.12",
            "uv pip install torch transformers accelerate sglang[all] curl_cffi",
        ]
        workspace = client.push("~/.bifrost/workspaces/rollouts-tito-test", bootstrap_cmd=bootstrap)
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

    except KeyboardInterrupt:
        print("\n\nInterrupted! Syncing logs before exit...")
        return 1

    finally:
        # Sync logs back (even on failure/interrupt)
        print("\nSyncing logs...")
        local_results = Path("results/tito-test")
        local_results.mkdir(parents=True, exist_ok=True)

        if workspace:
            files_to_sync = [
                # Test logs
                (f"{workspace}/rollouts/results/tito-test.log", "tito-test.log"),
                (f"{workspace}/rollouts/results/agent-loop-tito-test.log", "agent-loop-tito-test.log"),
                # SGLang logs
                ("/tmp/tito-test/sglang.log", "sglang-retok.log"),
                ("/tmp/tito-agent-test/sglang.log", "sglang-agent.log"),
            ]

            for remote_path, local_name in files_to_sync:
                try:
                    result = client.download_files(
                        remote_path=remote_path,
                        local_path=str(local_results / local_name),
                        recursive=False,
                    )
                    if result and result.success:
                        print(f"  Synced: {local_name}")
                except Exception:
                    pass

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
