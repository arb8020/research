#!/usr/bin/env python3
"""Download a HuggingFace model snapshot into a remote HF cache directory."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path

repo_root = Path(__file__).parent.parent
for subpkg in ["rollouts", "bifrost", "broker", "shared"]:
    pkg_path = repo_root / subpkg
    if pkg_path.exists():
        sys.path.insert(0, str(pkg_path))
sys.path.insert(0, str(repo_root))

env_file = repo_root / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

import trio  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Download model weights to HF cache on remote node"
    )

    # Node acquisition (mutually exclusive)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ssh", type=str, help="Static SSH connection string (user@host:port)")
    group.add_argument(
        "--node-id",
        type=str,
        help="Existing broker instance ID (provider:instance_id)",
    )

    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="HF model ID")
    parser.add_argument("--revision", type=str, help="Model revision/commit (default: latest)")
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default="/workspace/.cache/huggingface",
        help="HF cache directory on remote node",
    )

    parser.add_argument("--gpu-type", type=str, default="4090", help="GPU type to provision")
    parser.add_argument("--gpu-count", type=int, default=1, help="GPU count")
    parser.add_argument("--provider", type=str, default="runpod", help="Provider to provision on")
    parser.add_argument("--community", action="store_true", help="Use community cloud")
    parser.add_argument("--network-volume-id", type=str, help="RunPod network volume ID")
    parser.add_argument(
        "--datacenter-id",
        type=str,
        help="RunPod datacenter ID (required with --network-volume-id)",
    )
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout seconds for download")
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep provisioned instance alive after download",
    )

    return parser.parse_args()


def build_download_command(
    model: str, revision: str | None, hf_cache_dir: str, hf_token: str | None
) -> str:
    """Build shell command that downloads a model snapshot."""
    quote = shlex.quote
    token_line = f"export HF_TOKEN={quote(hf_token)}\n" if hf_token else ""
    revision_arg = f",\\n    revision={revision!r}" if revision else ""

    return f"""
set -euo pipefail

mkdir -p {quote(hf_cache_dir)}
export HF_HOME={quote(hf_cache_dir)}
export HF_HUB_ENABLE_HF_TRANSFER=1
{token_line}

python - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("huggingface_hub") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])

from huggingface_hub import snapshot_download

result = snapshot_download(
    repo_id={quote(model)!r},
    cache_dir={quote(hf_cache_dir)!r},{revision_arg}
    resume_download=True,
)
print(result)
PY
"""


async def main() -> int:
    """Download model snapshot on provisioned or existing node."""
    args = parse_args()

    if args.network_volume_id and not args.datacenter_id:
        print("Error: --datacenter-id is required when --network-volume-id is set.")
        return 1
    if args.network_volume_id and args.provider != "runpod":
        print("Error: --network-volume-id is only supported with --provider runpod.")
        return 1

    try:
        from bifrost import GPUQuery, acquire_node
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install -e bifrost broker")
        return 1

    provisioned = args.ssh is None and args.node_id is None
    client = None
    instance = None

    if args.ssh:
        print(f"Using static SSH: {args.ssh}")
        client, instance = await acquire_node(ssh=args.ssh)
    elif args.node_id:
        print(f"Using existing instance: {args.node_id}")
        client, instance = await acquire_node(node_id=args.node_id)
        assert instance is not None
    else:
        cloud_type = "community" if args.community else "secure"
        print(f"Provisioning new instance: {args.gpu_count}x {args.gpu_type} ({cloud_type})")
        client, instance = await acquire_node(
            provision=GPUQuery(
                type=args.gpu_type,
                count=args.gpu_count,
                provider=args.provider,
                cloud_type=cloud_type,
                network_volume_id=args.network_volume_id,
                datacenter_id=args.datacenter_id,
            )
        )
        assert instance is not None
        print(f"Instance ID: {instance.provider}:{instance.id}")

    assert client is not None
    assert instance is not None or args.ssh

    if args.node_id:
        print(f"Target: {args.node_id}")

    print("Starting model download...")
    cmd = build_download_command(
        args.model,
        args.revision,
        args.hf_cache_dir,
        os.environ.get("HF_TOKEN"),
    )
    start = datetime.now()

    result = client.exec(cmd, timeout=args.timeout)
    duration = (datetime.now() - start).total_seconds()
    print(f"Download command exit_code: {result.exit_code}")
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.exit_code != 0:
        print(f"Download failed after {duration:.1f}s")
        return 1

    print(f"Download completed in {duration:.1f}s")
    print(f"HF cache: {args.hf_cache_dir}")

    if provisioned and not args.keep_alive and instance is not None:
        print("Terminating instance (download-only workflow)")
        await instance.terminate()

    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(trio.run(main))
