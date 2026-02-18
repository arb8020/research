"""Modal sandbox for running warmup model locally.

Usage:
    python -m src.modal_sandbox create    # Create sandbox, print ID
    python -m src.modal_sandbox shell     # Interactive shell in sandbox
    python -m src.modal_sandbox run "python script.py"  # Run command
"""

from __future__ import annotations

import subprocess
import sys


# Modal app name for this project
APP_NAME = "dormant-llm-puzzle"

# Image with transformers + torch for Qwen2-8B
SANDBOX_SCRIPT = '''
import modal

app = modal.App.lookup("{app_name}", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        index_url="https://download.pytorch.org/whl/cu124",
        extra_index_url="https://pypi.org/simple",
    )
    .pip_install(
        "transformers>=4.50",
        "accelerate",
        "safetensors",
        "numpy",
        "bitsandbytes",  # For quantization if needed
    )
    .env({{
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
    }})
    .run_commands("mkdir -p /workspace")
)

sandbox = modal.Sandbox.create(
    app=app,
    image=image,
    gpu="{gpu_type}",
    timeout=3600,  # 1 hour
)

print(sandbox.object_id)
'''


def create_sandbox(gpu_type: str = "A10G") -> str:
    """Create a Modal sandbox and return its ID."""
    script = SANDBOX_SCRIPT.format(app_name=APP_NAME, gpu_type=gpu_type)

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error creating sandbox: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    sandbox_id = result.stdout.strip().split("\n")[-1]
    print(f"Created sandbox: {sandbox_id}")
    return sandbox_id


def run_in_sandbox(sandbox_id: str, command: str, timeout: int = 300) -> tuple[str, str, int]:
    """Run a command in an existing sandbox."""
    import modal

    sandbox = modal.Sandbox.from_id(sandbox_id)
    proc = sandbox.exec("bash", "-c", command, timeout=timeout)
    proc.wait()

    stdout = proc.stdout.read()
    stderr = proc.stderr.read()

    return stdout, stderr, proc.returncode


def interactive_shell(sandbox_id: str) -> None:
    """Start an interactive shell in the sandbox.

    Note: This is a simple REPL, not a full PTY.
    """
    import modal

    sandbox = modal.Sandbox.from_id(sandbox_id)
    print(f"Connected to sandbox {sandbox_id}")
    print("Type commands (Ctrl+D to exit):")
    print()

    while True:
        try:
            cmd = input("$ ")
        except EOFError:
            print("\nExiting...")
            break

        if not cmd.strip():
            continue

        stdout, stderr, code = run_in_sandbox(sandbox_id, cmd)
        if stdout:
            print(stdout, end="")
        if stderr:
            print(stderr, file=sys.stderr, end="")
        if code != 0:
            print(f"[exit code: {code}]")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Modal sandbox for dormant LLM puzzle")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create
    create_parser = subparsers.add_parser("create", help="Create a new sandbox")
    create_parser.add_argument("--gpu", default="A10G", help="GPU type (T4, A10G, A100, H100)")

    # shell
    shell_parser = subparsers.add_parser("shell", help="Interactive shell")
    shell_parser.add_argument("sandbox_id", help="Sandbox ID")

    # run
    run_parser = subparsers.add_parser("run", help="Run a command")
    run_parser.add_argument("sandbox_id", help="Sandbox ID")
    run_parser.add_argument("cmd", help="Command to run")
    run_parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")

    args = parser.parse_args()

    if args.command == "create":
        create_sandbox(args.gpu)
    elif args.command == "shell":
        interactive_shell(args.sandbox_id)
    elif args.command == "run":
        stdout, stderr, code = run_in_sandbox(args.sandbox_id, args.cmd, args.timeout)
        print(stdout, end="")
        if stderr:
            print(stderr, file=sys.stderr, end="")
        sys.exit(code)


if __name__ == "__main__":
    main()
