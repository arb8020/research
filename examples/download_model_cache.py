#!/usr/bin/env python3
"""Download a HuggingFace model snapshot into a remote HF cache directory."""

from __future__ import annotations

import argparse
import logging
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
    parser.add_argument(
        "--persistent-volume-id",
        type=str,
        help="Persistent volume ID to attach (RunPod network volumes supported today)",
    )
    parser.add_argument(
        "--persistent-volume-location",
        type=str,
        help="Provider-specific placement hint for the volume (for RunPod: datacenter ID)",
    )
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout seconds for download")
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep provisioned instance alive after download",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Directory to write structured JSONL events and streamed output logs",
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
set -uo pipefail

__download_exit_code=0

{{
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
}} || __download_exit_code=$?

echo "__DOWNLOAD_EXIT_CODE__=${{__download_exit_code}}"
exit "${{__download_exit_code}}"
"""


def _sanitize_sample_id(value: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in value)


async def main() -> int:
    """Download model snapshot on provisioned or existing node."""
    args = parse_args()

    if args.persistent_volume_id and not args.persistent_volume_location:
        print("Error: --persistent-volume-location is required when --persistent-volume-id is set.")
        return 1
    if args.persistent_volume_id and args.provider != "runpod":
        print("Error: persistent volumes are currently supported only with --provider runpod.")
        return 1

    try:
        from bifrost import GPUQuery, acquire_node
        from rollouts._logging import setup_eval_logging
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install -e bifrost broker rollouts")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else repo_root / "results" / f"model_cache_download_{timestamp}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    output_log = log_dir / "download_output.log"
    sample_id = _sanitize_sample_id(args.model)

    log_ctx = setup_eval_logging(log_dir, logger_name="download_model_cache.events")
    event_logger = logging.getLogger("download_model_cache.events")
    event_logger.info(
        "download_start",
        extra={
            "sample_id": sample_id,
            "model": args.model,
            "revision": args.revision,
            "provider": args.provider,
            "gpu_type": args.gpu_type,
            "gpu_count": args.gpu_count,
            "persistent_volume_id": args.persistent_volume_id,
            "persistent_volume_location": args.persistent_volume_location,
            "hf_cache_dir": args.hf_cache_dir,
            "log_dir": str(log_dir),
        },
    )

    print(f"Structured events: {log_dir / 'events.jsonl'}")
    print(f"Streamed output:   {output_log}")
    print(f"Tail live:         tail -f {output_log}")

    provisioned = args.ssh is None and args.node_id is None
    client = None
    instance = None
    sample_ended = False
    try:
        if args.ssh:
            print(f"Using static SSH: {args.ssh}")
            event_logger.info("acquire_node_start", extra={"sample_id": sample_id, "mode": "ssh"})
            client, instance = await acquire_node(ssh=args.ssh)
        elif args.node_id:
            print(f"Using existing instance: {args.node_id}")
            event_logger.info(
                "acquire_node_start",
                extra={"sample_id": sample_id, "mode": "node_id", "node_id": args.node_id},
            )
            client, instance = await acquire_node(node_id=args.node_id)
            assert instance is not None
        else:
            cloud_type = "community" if args.community else "secure"
            print(f"Provisioning new instance: {args.gpu_count}x {args.gpu_type} ({cloud_type})")
            event_logger.info(
                "acquire_node_start",
                extra={
                    "sample_id": sample_id,
                    "mode": "provision",
                    "cloud_type": cloud_type,
                },
            )
            client, instance = await acquire_node(
                provision=GPUQuery(
                    type=args.gpu_type,
                    count=args.gpu_count,
                    provider=args.provider,
                    cloud_type=cloud_type,
                    persistent_volume_id=args.persistent_volume_id,
                    persistent_volume_location=args.persistent_volume_location,
                )
            )
            assert instance is not None
            print(f"Instance ID: {instance.provider}:{instance.id}")
            event_logger.info(
                "instance_provisioned",
                extra={
                    "sample_id": sample_id,
                    "instance_id": instance.id,
                    "provider": instance.provider,
                },
            )

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

        event_logger.info(
            "remote_download_start",
            extra={
                "sample_id": sample_id,
                "timeout_seconds": args.timeout,
            },
        )

        stream_lines = 0
        download_exit_code: int | None = None
        try:
            with output_log.open("a") as f:
                for line in client.exec_stream(cmd, timeout=args.timeout):
                    stream_lines += 1
                    f.write(line + "\n")
                    f.flush()
                    if line.startswith("__DOWNLOAD_EXIT_CODE__="):
                        try:
                            download_exit_code = int(line.split("=", 1)[1].strip())
                        except ValueError:
                            download_exit_code = 1
                    elif stream_lines % 25 == 0:
                        event_logger.info(
                            "download_progress",
                            extra={
                                "sample_id": sample_id,
                                "stream_lines": stream_lines,
                                "last_line": line[:500],
                            },
                        )
        except Exception as e:
            duration = (datetime.now() - start).total_seconds()
            event_logger.exception(
                "remote_download_stream_error",
                extra={
                    "sample_id": sample_id,
                    "duration_seconds": round(duration, 2),
                    "error": str(e),
                },
            )
            print(f"Download failed after {duration:.1f}s (stream error)")
            return 1

        duration = (datetime.now() - start).total_seconds()
        if download_exit_code is None:
            download_exit_code = 1

        print(f"Download command exit_code: {download_exit_code}")

        if download_exit_code != 0:
            event_logger.error(
                "download_failed",
                extra={
                    "sample_id": sample_id,
                    "duration_seconds": round(duration, 2),
                    "stream_lines": stream_lines,
                    "exit_code": download_exit_code,
                },
            )
            print(f"Download failed after {duration:.1f}s")
            return 1

        event_logger.info(
            "download_succeeded",
            extra={
                "sample_id": sample_id,
                "duration_seconds": round(duration, 2),
                "stream_lines": stream_lines,
                "exit_code": download_exit_code,
            },
        )
        print(f"Download completed in {duration:.1f}s")
        print(f"HF cache: {args.hf_cache_dir}")

        if provisioned and not args.keep_alive and instance is not None:
            print("Terminating instance (download-only workflow)")
            terminated = await instance.terminate()
            event_logger.info(
                "instance_terminated",
                extra={
                    "sample_id": sample_id,
                    "instance_id": instance.id,
                    "terminated": terminated,
                },
            )

        event_logger.info("sample_end", extra={"sample_id": sample_id})
        sample_ended = True
        print("Done")
        return 0
    finally:
        if not sample_ended:
            event_logger.info("sample_end", extra={"sample_id": sample_id})
        log_ctx.teardown()


if __name__ == "__main__":
    sys.exit(trio.run(main))
