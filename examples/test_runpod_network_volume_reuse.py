#!/usr/bin/env python3
"""Validate RunPod network-volume model cache reuse across pod restarts.

Test flow:
1. Provision pod with attached network volume and download model to HF cache.
2. Terminate pod.
3. Provision fresh pod with same volume/cache path.
4. Verify cache-only load succeeds with HF_HUB_OFFLINE=1 and local_files_only=True.
5. Terminate pod.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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
    parser = argparse.ArgumentParser(description="Test RunPod network volume cache reuse")
    parser.add_argument(
        "--persistent-volume-id",
        required=True,
        help="Persistent volume ID to attach (RunPod network volume)",
    )
    parser.add_argument(
        "--persistent-volume-location",
        required=True,
        help="Provider-specific placement hint for the volume (for RunPod: datacenter ID)",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF model ID")
    parser.add_argument("--gpu-type", default="RTX PRO 6000", help="GPU type filter")
    parser.add_argument("--gpu-count", type=int, default=1, help="GPU count")
    parser.add_argument("--provider", default="runpod", help="Provider (must be runpod)")
    parser.add_argument(
        "--hf-cache-dir", default="/workspace/.cache/huggingface", help="Remote HF cache dir"
    )
    parser.add_argument(
        "--timeout", type=int, default=3600, help="Per remote command timeout in seconds"
    )
    parser.add_argument("--log-dir", type=str, help="Directory for events/output logs")
    parser.add_argument(
        "--skip-preload",
        action="store_true",
        help="Skip initial preload and only run cache-only verify on fresh pod",
    )
    return parser.parse_args()


def _sanitize_sample_id(value: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in value)


def _build_snapshot_cmd(model: str, hf_cache_dir: str, hf_token: str | None, offline: bool) -> str:
    quote = shlex.quote
    token_line = f"export HF_TOKEN={quote(hf_token)}\n" if hf_token else ""
    offline_line = "export HF_HUB_OFFLINE=1\nexport TRANSFORMERS_OFFLINE=1\n" if offline else ""
    local_only = "True" if offline else "False"
    mode = "verify_offline" if offline else "preload"
    return f"""
set -uo pipefail
__cmd_exit_code=0
{{
mkdir -p {quote(hf_cache_dir)}
export HF_HOME={quote(hf_cache_dir)}
export HF_HUB_ENABLE_HF_TRANSFER=1
{offline_line}{token_line}
python - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("huggingface_hub") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])

from huggingface_hub import snapshot_download

result = snapshot_download(
    repo_id={quote(model)!r},
    cache_dir={quote(hf_cache_dir)!r},
    local_files_only={local_only},
)
print("{mode}_snapshot_path", result)
PY
}} || __cmd_exit_code=$?
echo "__STEP_EXIT_CODE__=${{__cmd_exit_code}}"
exit "${{__cmd_exit_code}}"
"""


def _run_streamed_command(
    *,
    client: Any,
    cmd: str,
    timeout: int,
    output_log: Path,
    event_logger: logging.Logger,
    sample_id: str,
    phase: str,
) -> tuple[int, int]:
    lines = 0
    exit_code: int | None = None
    with output_log.open("a") as f:
        for line in client.exec_stream(cmd, timeout=timeout):
            lines += 1
            f.write(f"[{phase}] {line}\n")
            f.flush()
            if line.startswith("__STEP_EXIT_CODE__="):
                try:
                    exit_code = int(line.split("=", 1)[1].strip())
                except ValueError:
                    exit_code = 1
            elif lines % 25 == 0:
                event_logger.info(
                    "phase_progress",
                    extra={
                        "sample_id": sample_id,
                        "phase": phase,
                        "stream_lines": lines,
                        "last_line": line[:500],
                    },
                )
    if exit_code is None:
        exit_code = 1
    return exit_code, lines


async def _provision_single_pod(
    args: argparse.Namespace, event_logger: logging.Logger, sample_id: str
) -> tuple[Any, Any]:
    from bifrost import GPUQuery, acquire_node

    event_logger.info(
        "acquire_node_start",
        extra={
            "sample_id": sample_id,
            "mode": "provision",
            "provider": args.provider,
            "gpu_type": args.gpu_type,
            "gpu_count": args.gpu_count,
            "persistent_volume_id": args.persistent_volume_id,
            "persistent_volume_location": args.persistent_volume_location,
        },
    )
    client, instance = await acquire_node(
        provision=GPUQuery(
            type=args.gpu_type,
            count=args.gpu_count,
            provider=args.provider,
            cloud_type="secure",
            persistent_volume_id=args.persistent_volume_id,
            persistent_volume_location=args.persistent_volume_location,
        )
    )
    assert instance is not None
    event_logger.info(
        "instance_provisioned",
        extra={
            "sample_id": sample_id,
            "instance_id": instance.id,
            "provider": instance.provider,
        },
    )
    return client, instance


async def main() -> int:
    args = parse_args()
    if args.provider != "runpod":
        print("Error: this script currently supports --provider runpod only")
        return 1

    try:
        from rollouts._logging import setup_eval_logging
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install -e rollouts bifrost broker")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else repo_root / "results" / f"network_volume_reuse_{timestamp}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    output_log = log_dir / "download_output.log"
    sample_id = _sanitize_sample_id(args.model)

    log_ctx = setup_eval_logging(log_dir, logger_name="network_volume_reuse.events")
    event_logger = logging.getLogger("network_volume_reuse.events")
    event_logger.info(
        "test_start",
        extra={
            "sample_id": sample_id,
            "model": args.model,
            "persistent_volume_id": args.persistent_volume_id,
            "persistent_volume_location": args.persistent_volume_location,
            "hf_cache_dir": args.hf_cache_dir,
            "skip_preload": args.skip_preload,
            "log_dir": str(log_dir),
        },
    )
    print(f"Structured events: {log_dir / 'events.jsonl'}")
    print(f"Streamed output:   {output_log}")
    print(f"Tail live:         tail -f {output_log}")

    hf_token = os.environ.get("HF_TOKEN")
    current_instance = None
    sample_ended = False
    try:
        if not args.skip_preload:
            # Phase 1: preload cache on pod A
            client, current_instance = await _provision_single_pod(args, event_logger, sample_id)
            preload_cmd = _build_snapshot_cmd(
                model=args.model,
                hf_cache_dir=args.hf_cache_dir,
                hf_token=hf_token,
                offline=False,
            )
            event_logger.info("phase_start", extra={"sample_id": sample_id, "phase": "preload"})
            t0 = datetime.now()
            preload_exit, preload_lines = _run_streamed_command(
                client=client,
                cmd=preload_cmd,
                timeout=args.timeout,
                output_log=output_log,
                event_logger=event_logger,
                sample_id=sample_id,
                phase="preload",
            )
            preload_dur = (datetime.now() - t0).total_seconds()
            event_logger.info(
                "phase_end",
                extra={
                    "sample_id": sample_id,
                    "phase": "preload",
                    "duration_seconds": round(preload_dur, 2),
                    "stream_lines": preload_lines,
                    "exit_code": preload_exit,
                },
            )
            if preload_exit != 0:
                print("Preload phase failed.")
                return 1

            terminated = await current_instance.terminate()
            event_logger.info(
                "instance_terminated",
                extra={
                    "sample_id": sample_id,
                    "instance_id": current_instance.id,
                    "phase": "preload",
                    "terminated": terminated,
                },
            )
            current_instance = None

        # Phase 2: fresh pod, verify offline cache-only load
        client2, current_instance = await _provision_single_pod(args, event_logger, sample_id)
        verify_cmd = _build_snapshot_cmd(
            model=args.model,
            hf_cache_dir=args.hf_cache_dir,
            hf_token=hf_token,
            offline=True,
        )
        event_logger.info("phase_start", extra={"sample_id": sample_id, "phase": "verify_offline"})
        t1 = datetime.now()
        verify_exit, verify_lines = _run_streamed_command(
            client=client2,
            cmd=verify_cmd,
            timeout=args.timeout,
            output_log=output_log,
            event_logger=event_logger,
            sample_id=sample_id,
            phase="verify_offline",
        )
        verify_dur = (datetime.now() - t1).total_seconds()
        event_logger.info(
            "phase_end",
            extra={
                "sample_id": sample_id,
                "phase": "verify_offline",
                "duration_seconds": round(verify_dur, 2),
                "stream_lines": verify_lines,
                "exit_code": verify_exit,
            },
        )
        if verify_exit != 0:
            print("Offline verify phase failed.")
            return 1

        terminated2 = await current_instance.terminate()
        event_logger.info(
            "instance_terminated",
            extra={
                "sample_id": sample_id,
                "instance_id": current_instance.id,
                "phase": "verify_offline",
                "terminated": terminated2,
            },
        )
        current_instance = None

        event_logger.info("test_passed", extra={"sample_id": sample_id})
        event_logger.info("sample_end", extra={"sample_id": sample_id})
        sample_ended = True
        print("PASS: network volume cache reuse verified (offline/local-only on fresh pod).")
        return 0
    finally:
        if current_instance is not None:
            try:
                terminated = await current_instance.terminate()
                event_logger.info(
                    "instance_terminated",
                    extra={
                        "sample_id": sample_id,
                        "instance_id": current_instance.id,
                        "phase": "cleanup",
                        "terminated": terminated,
                    },
                )
            except Exception as e:
                event_logger.exception(
                    "instance_terminate_error",
                    extra={"sample_id": sample_id, "error": str(e)},
                )
        if not sample_ended:
            event_logger.info("sample_end", extra={"sample_id": sample_id})
        log_ctx.teardown()


if __name__ == "__main__":
    sys.exit(trio.run(main))
