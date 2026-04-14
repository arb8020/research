"""Run a scratch script on the B200 node using bifrost.

Usage:
    python scratch/run.py scratch/explore_tokenizer.py
    python scratch/run.py scratch/explore_tokenizer.py --env MY_VAR=value HF_TOKEN

The script must have inline uv dependencies at the top:
    # /// script
    # dependencies = ["torch", "transformers"]
    # ///

HF_TOKEN is always forwarded from local env if set.
stdout/stderr stream back in real time with source labels. Exit code is forwarded.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

DEFAULT_SSH = "ubuntu@146.88.195.10:2222"
REMOTE_SCRATCH = "~/scratch"


def _repo_root() -> Path:
    """Walk up from cwd until we find a .git directory."""
    p = Path.cwd()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    raise RuntimeError("not inside a git repo")


def parse_env(env_args: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for arg in env_args:
        if "=" in arg:
            k, v = arg.split("=", 1)
            result[k] = v
        else:
            val = os.environ.get(arg)
            if val is None:
                print(f"warning: {arg} not set locally, skipping", file=sys.stderr)
            else:
                result[arg] = val
    return result


async def run(script: str, ssh: str, env_args: list[str]) -> int:
    from bifrost import ProcessSpec, acquire_node

    env_vars = parse_env(env_args)
    if "HF_TOKEN" not in env_vars and "HF_TOKEN" in os.environ:
        env_vars["HF_TOKEN"] = os.environ["HF_TOKEN"]

    script_path = Path(script)
    if not script_path.is_absolute() and not script_path.exists():
        script_path = _repo_root() / script_path
    assert script_path.exists(), f"script not found: {script_path}"

    print(f"connecting to {ssh} ...", file=sys.stderr)
    bifrost, _ = await acquire_node(ssh=ssh)

    remote_dir = bifrost.expand_path(REMOTE_SCRATCH)
    bifrost.exec(f"mkdir -p {remote_dir}")

    remote_script = f"{remote_dir}/{script_path.name}"
    print(f"uploading {script_path.name} -> {remote_script}", file=sys.stderr)
    bifrost.upload_files(str(script_path), remote_script)

    print(f"running: uv run {script_path.name}\n", file=sys.stderr)
    spec = ProcessSpec(
        command="/home/ubuntu/.local/bin/uv",
        args=("run", remote_script),
        cwd=remote_dir,
        env=env_vars or None,
    )
    handle = bifrost.start_process(spec, name=script_path.stem)

    async for line in handle.stream_output():
        out = sys.stdout if line.stream == "stdout" else sys.stderr
        print(line.text, file=out)
        out.flush()

    result = await handle.wait()
    return result.exit_code


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("script", help="Local script path to run remotely")
    parser.add_argument("--ssh", default=DEFAULT_SSH, help="SSH target (user@host:port)")
    parser.add_argument("--env", nargs="+", default=[], metavar="KEY[=VALUE]")
    args = parser.parse_args()

    exit_code = asyncio.run(run(args.script, args.ssh, args.env))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
