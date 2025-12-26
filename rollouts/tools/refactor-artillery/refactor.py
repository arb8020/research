#!/usr/bin/env python3
"""
Multi-file AI refactoring tool.

Usage:
    python -m tools.refactor.refactor <file> --model anthropic/claude-sonnet-4-5-20250929
    python -m tools.refactor.refactor <file> --model openai/gpt-5.1 --thinking high
    python -m tools.refactor.refactor <file> --dry-run

Example:
    python -m tools.refactor.refactor main.py --model anthropic/claude-sonnet-4-5-20250929
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from .blocks import BlockState, build_block_state, estimate_tokens, format_blocks
from .commands import apply_commands, parse_commands
from .imports import collect_context

# Default model
DEFAULT_MODEL = "anthropic/claude-sonnet-4-5-20250929"

# Compaction threshold (tokens)
COMPACTION_THRESHOLD = 32000


SYSTEM_PROMPT = """You are a code refactoring assistant. Analyze the provided files and perform the requested refactoring.

The files are split into blocks. Each block is marked with '!N' where N is the block ID.

Output your changes using these commands:

1. To create a new file:
<write file="path/to/file.py">
complete file contents here
</write>

2. To replace a specific block by ID:
<patch id=N>
new block contents
</patch>

3. To delete a file:
<delete file="path/to/file.py"/>

Rules:
- Output ONLY the commands, no explanations
- Use block IDs for patches (e.g., <patch id=5>)
- The block ID markers (!N) are NOT part of the actual file content
- Use relative paths from the workspace root
- Make minimal changes to accomplish the task
"""


COMPACTION_PROMPT = """You are a code context optimizer. Your job is to identify which blocks are NOT relevant to the task.

The files below are split into blocks marked with '!N'. The task is described at the end.

Output <omit>N</omit> for each block ID that is NOT needed to complete the task.
Be aggressive - omit anything that isn't directly relevant.
Do NOT omit blocks that:
- Contain the task description
- Define functions/classes that will be modified
- Are imported by relevant code

Output only <omit> commands, nothing else.
"""


def build_prompt(block_state: BlockState, task: str, omit_ids: set[int] | None = None) -> str:
    """Build the prompt from blocks and task."""
    lines = [
        "You're a code editor.",
        "",
        "Files are split into blocks. Each block is marked with '!id'.",
        "These markers are NOT part of the file; they identify blocks for patching.",
        "",
    ]

    lines.append(format_blocks(block_state, omit_ids))

    lines.append("")
    lines.append("TASK:")
    lines.append(task)
    lines.append("")
    lines.append("Output <write>, <patch id=N>, or <delete> commands to complete the task.")

    return "\n".join(lines)


def build_compaction_prompt(block_state: BlockState, task: str) -> str:
    """Build the compaction prompt."""
    lines = [format_blocks(block_state)]
    lines.append("")
    lines.append("TASK:")
    lines.append(task)
    lines.append("")
    lines.append("Output <omit>N</omit> for each irrelevant block ID.")

    return "\n".join(lines)


def extract_task_from_file(content: str) -> tuple[str, str]:
    """
    Extract task from trailing comments in file.
    Returns (body_without_task, task_text).
    """
    lines = content.rstrip().split("\n")

    # Find trailing comment block
    task_lines = []
    idx = len(lines) - 1

    while idx >= 0:
        line = lines[idx]
        stripped = line.strip()

        # Check if line is a comment
        is_comment = False
        comment_text = ""

        for prefix in ("#", "//", "--"):
            if stripped.startswith(prefix):
                is_comment = True
                comment_text = stripped[len(prefix) :].strip()
                break

        if is_comment:
            task_lines.append(comment_text)
            idx -= 1
        elif stripped == "":
            # Skip trailing blank lines
            idx -= 1
        else:
            break

    if not task_lines:
        raise ValueError("File must end with a comment block describing the task")

    task = "\n".join(reversed(task_lines))
    body = "\n".join(lines[: idx + 1])

    return body, task


def get_api_key(vendor: str) -> str:
    """Get API key from environment or config file."""
    # Try environment variable first
    env_vars = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    env_var = env_vars.get(vendor)
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]

    # Try config file
    config_path = Path.home() / ".config" / f"{vendor}.token"
    if config_path.exists():
        return config_path.read_text().strip()

    raise ValueError(
        f"No API key found for {vendor}. Set ${env_var} or create ~/.config/{vendor}.token"
    )


def call_anthropic(prompt: str, model: str, thinking: str, api_key: str) -> str:
    """Call Anthropic API."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    # Map thinking level to budget_tokens
    thinking_budgets = {
        "low": 2048,
        "medium": 4096,
        "high": 8192,
    }

    kwargs = {
        "model": model,
        "max_tokens": 16384,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }

    if thinking != "none":
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budgets.get(thinking, 4096),
        }

    response = client.messages.create(**kwargs)

    # Extract text from response
    text_parts = []
    for block in response.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)

    return "\n".join(text_parts)


def call_openai(prompt: str, model: str, thinking: str, api_key: str) -> str:
    """Call OpenAI API."""
    import openai

    client = openai.OpenAI(api_key=api_key)

    # Map thinking level to reasoning effort
    reasoning_efforts = {
        "low": "low",
        "medium": "medium",
        "high": "high",
    }

    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }

    if thinking != "none":
        kwargs["reasoning"] = {"effort": reasoning_efforts.get(thinking, "medium")}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def call_google(prompt: str, model: str, thinking: str, api_key: str) -> str:
    """Call Google Gemini API."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    generation_config = {}
    if thinking != "none":
        generation_config["thinking_config"] = {
            "thinking_level": "low" if thinking == "low" else "high",
            "include_thoughts": False,
        }

    model_instance = genai.GenerativeModel(
        model_name=model,
        system_instruction=SYSTEM_PROMPT,
        generation_config=generation_config if generation_config else None,
    )

    response = model_instance.generate_content(prompt)
    return response.text


def call_llm(prompt: str, vendor: str, model: str, thinking: str) -> str:
    """Call the appropriate LLM API."""
    api_key = get_api_key(vendor)

    if vendor == "anthropic":
        return call_anthropic(prompt, model, thinking, api_key)
    elif vendor == "openai":
        return call_openai(prompt, model, thinking, api_key)
    elif vendor == "google":
        return call_google(prompt, model, thinking, api_key)
    else:
        raise ValueError(f"Unknown vendor: {vendor}")


def save_log(
    log_dir: Path,
    prompt: str,
    response: str,
    file_path: str,
    model: str,
) -> None:
    """Save prompt and response to log files."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = Path(file_path).stem

    # Save latest
    (log_dir / "refactor-prompt.txt").write_text(prompt)
    (log_dir / "refactor-response.txt").write_text(response)

    # Save to history
    history_dir = log_dir / "refactor-history"
    history_dir.mkdir(exist_ok=True)
    (history_dir / f"{timestamp}-{base_name}-prompt.txt").write_text(prompt)
    (history_dir / f"{timestamp}-{base_name}-response.txt").write_text(response)


def parse_model_spec(model_spec: str) -> tuple[str, str]:
    """Parse 'provider/model' format into (vendor, model)."""
    if "/" not in model_spec:
        raise ValueError(f"Model must be in 'provider/model' format, got: {model_spec}")

    vendor, model = model_spec.split("/", 1)
    return vendor, model


def parse_omit_commands(response: str) -> set[int]:
    """Parse <omit>N</omit> commands from compaction response."""
    import re

    omit_ids = set()

    for match in re.finditer(r"<omit>(\d+)</omit>", response):
        omit_ids.add(int(match.group(1)))

    # Also support ranges: <omit>5-10</omit>
    for match in re.finditer(r"<omit>(\d+)-(\d+)</omit>", response):
        start, end = int(match.group(1)), int(match.group(2))
        omit_ids.update(range(start, end + 1))

    return omit_ids


def refactor(
    file_path: str,
    model: str = DEFAULT_MODEL,
    thinking: str = "medium",
    dry_run: bool = False,
    no_compact: bool = False,
) -> None:
    """
    Main refactor function.

    Args:
        file_path: Path to the file to refactor
        model: Model in "provider/model" format (e.g., "anthropic/claude-sonnet-4-5-20250929")
        thinking: Thinking level (none, low, medium, high)
        dry_run: If True, print prompt instead of calling API
        no_compact: If True, skip compaction even if over threshold
    """
    file_path = Path(file_path).resolve()
    workspace_root = Path.cwd().resolve()

    # Parse model spec
    try:
        vendor, model_name = parse_model_spec(model)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"model: {vendor}/{model_name}")
    print(f"thinking: {thinking}")

    # Read file and extract task
    content = file_path.read_text()
    try:
        body, task = extract_task_from_file(content)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"task: {task[:80]}{'...' if len(task) > 80 else ''}")

    # Collect context
    print("Collecting context...")
    context = collect_context(file_path, workspace_root)
    print(f"context: {len(context)} files")

    # Replace root file content with body (without task)
    rel_path = str(file_path.relative_to(workspace_root))
    context[rel_path] = body

    # Build block state
    block_state = build_block_state(context)
    print(f"blocks: {len(block_state.blocks)}")

    # Estimate tokens
    full_prompt = build_prompt(block_state, task)
    token_estimate = estimate_tokens(full_prompt)
    print(f"tokens: ~{token_estimate}")

    # Save log directory
    log_dir = Path.home() / ".ai"

    # Compaction pass (if needed)
    omit_ids: set[int] = set()
    has_imports = len(context) > 1

    if not no_compact and has_imports and token_estimate >= COMPACTION_THRESHOLD:
        print(f"\n** Compacting (>{COMPACTION_THRESHOLD} tokens)... **")

        compact_prompt = build_compaction_prompt(block_state, task)

        if not dry_run:
            # Use low thinking for compaction (fast)
            compact_response = call_llm(compact_prompt, vendor, model_name, "low")
            omit_ids = parse_omit_commands(compact_response)

            # Recalculate tokens
            compacted_prompt = build_prompt(block_state, task, omit_ids)
            new_token_estimate = estimate_tokens(compacted_prompt)
            savings = token_estimate - new_token_estimate
            pct = (savings / token_estimate * 100) if token_estimate > 0 else 0

            print(f"omitted: {len(omit_ids)} blocks")
            print(f"tokens: ~{new_token_estimate} (saved {savings}, {pct:.0f}%)")

            save_log(log_dir, compact_prompt, compact_response, str(file_path) + ".compact", model)
    else:
        reasons = []
        if no_compact:
            reasons.append("--no-compact")
        elif not has_imports:
            reasons.append("single file")
        elif token_estimate < COMPACTION_THRESHOLD:
            reasons.append(f"under {COMPACTION_THRESHOLD} tokens")
        print(f"compaction: skipped ({', '.join(reasons)})")

    # Build final prompt
    prompt = build_prompt(block_state, task, omit_ids)

    if dry_run:
        print("\n=== DRY RUN - PROMPT ===")
        print(prompt)
        save_log(log_dir, prompt, "[dry run]", str(file_path), model)
        return

    # Call LLM
    print("\n** Calling AI... **")
    response = call_llm(prompt, vendor, model_name, thinking)

    # Save logs
    save_log(log_dir, prompt, response, str(file_path), model)

    # Parse commands
    commands = parse_commands(response)
    if not commands:
        print("No commands found in response")
        print("\n=== RESPONSE ===")
        print(response)
        return

    print(f"\nApplying {len(commands)} commands...")

    # Apply commands (with block state for block-based patches)
    results = apply_commands(commands, workspace_root, block_state)

    for result in results:
        status = "✓" if result.success else "✗"
        print(f"  {status} {result.message}")

    print("\nDone!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-file AI refactoring tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m tools.refactor.refactor main.py
    python -m tools.refactor.refactor main.py --model openai/gpt-5.1
    python -m tools.refactor.refactor main.py --thinking high
    python -m tools.refactor.refactor main.py --dry-run
        """,
    )
    parser.add_argument(
        "file",
        help="File to refactor (must end with task comment)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f'Model in "provider/model" format (default: {DEFAULT_MODEL})',
    )
    parser.add_argument(
        "--thinking",
        "-t",
        type=str,
        choices=["none", "low", "medium", "high"],
        default="medium",
        help="Thinking level (default: medium)",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Print prompt without calling API",
    )
    parser.add_argument(
        "--no-compact",
        action="store_true",
        help="Skip compaction pass even if over token threshold",
    )

    args = parser.parse_args()

    refactor(args.file, args.model, args.thinking, args.dry_run, args.no_compact)


if __name__ == "__main__":
    main()
