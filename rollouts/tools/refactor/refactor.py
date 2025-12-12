#!/usr/bin/env python3
"""
Multi-file AI refactoring tool.

Usage:
    python -m tools.refactor.refactor <file> [model]

Models:
    s  = Claude Sonnet (medium thinking)
    S  = Claude Sonnet (high thinking)
    g  = GPT-5.1 (medium)
    G  = GPT-5.1 (high)
    i  = Gemini 3 (medium)
    I  = Gemini 3 (high)

Example:
    python -m tools.refactor.refactor main.py s
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from .imports import collect_context, detect_language
from .commands import parse_commands, apply_commands


# Model aliases (matching Victor's convention)
MODELS = {
    # Claude Sonnet
    "s": ("anthropic", "claude-sonnet-4-5-20250929", "medium"),
    "s-": ("anthropic", "claude-sonnet-4-5-20250929", "low"),
    "s+": ("anthropic", "claude-sonnet-4-5-20250929", "high"),
    "S": ("anthropic", "claude-sonnet-4-5-20250929", "high"),

    # Claude Opus
    "o": ("anthropic", "claude-opus-4-1-20250805", "medium"),
    "O": ("anthropic", "claude-opus-4-1-20250805", "high"),

    # GPT-5.1
    "g": ("openai", "gpt-5.1", "medium"),
    "g-": ("openai", "gpt-5.1", "low"),
    "g+": ("openai", "gpt-5.1", "high"),
    "G": ("openai", "gpt-5.1", "high"),

    # Gemini 3
    "i": ("google", "gemini-3-pro-preview", "medium"),
    "i-": ("google", "gemini-3-pro-preview", "low"),
    "i+": ("google", "gemini-3-pro-preview", "high"),
    "I": ("google", "gemini-3-pro-preview", "high"),
}


SYSTEM_PROMPT = """You are a code refactoring assistant. Analyze the provided files and perform the requested refactoring.

Output your changes using these commands:

1. To create or overwrite a file:
<write file="path/to/file.py">
complete file contents here
</write>

2. To modify part of a file (search and replace):
<patch file="path/to/file.py">
<<<<<<< SEARCH
exact text to find
=======
replacement text
>>>>>>>
</patch>

3. To delete a file:
<delete file="path/to/file.py"/>

Rules:
- Output ONLY the commands, no explanations
- Use relative paths from the workspace root
- For patches, the SEARCH text must match exactly (including whitespace)
- Make minimal changes to accomplish the task
"""


def build_prompt(context: dict[str, str], task: str) -> str:
    """Build the prompt from context and task."""
    lines = ["You're a code editor.", "", "Files:", ""]

    for file_path, content in context.items():
        lines.append(f"=== {file_path} ===")
        lines.append(content)
        lines.append("")

    lines.append("TASK:")
    lines.append(task)
    lines.append("")
    lines.append("Output <write>, <patch>, or <delete> commands to complete the task.")

    return "\n".join(lines)


def extract_task_from_file(content: str) -> tuple[str, str]:
    """
    Extract task from trailing comments in file.
    Returns (body_without_task, task_text).
    """
    lines = content.rstrip().split('\n')

    # Find trailing comment block
    task_lines = []
    idx = len(lines) - 1

    while idx >= 0:
        line = lines[idx]
        stripped = line.strip()

        # Check if line is a comment
        is_comment = False
        comment_text = ""

        for prefix in ('#', '//', '--'):
            if stripped.startswith(prefix):
                is_comment = True
                comment_text = stripped[len(prefix):].strip()
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
    body = "\n".join(lines[:idx + 1])

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

    raise ValueError(f"No API key found for {vendor}. Set ${env_var} or create ~/.config/{vendor}.token")


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
        if hasattr(block, 'text'):
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


def refactor(
    file_path: str,
    model_spec: str = "s",
    dry_run: bool = False,
) -> None:
    """
    Main refactor function.

    Args:
        file_path: Path to the file to refactor
        model_spec: Model specification (s, S, g, G, i, I, etc.)
        dry_run: If True, print prompt instead of calling API
    """
    file_path = Path(file_path).resolve()
    workspace_root = Path.cwd().resolve()

    # Resolve model
    if model_spec not in MODELS:
        print(f"Unknown model: {model_spec}")
        print(f"Available: {', '.join(MODELS.keys())}")
        sys.exit(1)

    vendor, model, thinking = MODELS[model_spec]
    print(f"model: {vendor}:{model}:{thinking}")

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

    # Build prompt
    prompt = build_prompt(context, task)
    token_estimate = len(prompt) // 4  # Rough estimate
    print(f"tokens: ~{token_estimate}")

    # Save log directory
    log_dir = Path.home() / ".ai"

    if dry_run:
        print("\n=== DRY RUN - PROMPT ===")
        print(prompt)
        save_log(log_dir, prompt, "[dry run]", str(file_path), model)
        return

    # Call LLM
    print("\nCalling AI...")
    response = call_llm(prompt, vendor, model, thinking)

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

    # Apply commands
    results = apply_commands(commands, workspace_root)

    for result in results:
        status = "✓" if result.success else "✗"
        print(f"  {status} {result.message}")

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Multi-file AI refactoring tool")
    parser.add_argument("file", help="File to refactor (must end with task comment)")
    parser.add_argument("model", nargs="?", default="s", help="Model spec (s, S, g, G, i, I)")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt without calling API")

    args = parser.parse_args()

    refactor(args.file, args.model, args.dry_run)


if __name__ == "__main__":
    main()
