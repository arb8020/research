"""Driver for the mini-swe-agent CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from functools import partial
from pathlib import Path
from typing import Any

import trio

from ..core import Message, Trajectory
from ..drivers.runner import _make_external_progress_emitter, _make_raw_driver_line_handler
from ..environments.local_workspace_resource import LocalWorkspaceResource
from ..environments.resources import SandboxWorkspaceResource
from ..training.types import Status
from .remote_runtime import _sample_id_slug
from .types import ExternalAttemptArtifact


def _mini_swe_agent_output_to_trajectory(payload: dict[str, Any]) -> Trajectory:
    messages: list[Message] = []
    for raw_msg in payload.get("messages", []):
        if not isinstance(raw_msg, dict):
            continue
        raw_role = str(raw_msg.get("role") or "").lower()
        if raw_role == "exit":
            continue
        role = raw_role if raw_role in {"system", "user", "assistant", "tool"} else "assistant"
        raw_content = raw_msg.get("content")
        if isinstance(raw_content, str) or raw_content is None:
            content = raw_content
        else:
            content = json.dumps(raw_content, ensure_ascii=False)
        details = {k: v for k, v in raw_msg.items() if k not in {"role", "content", "tool_call_id"}}
        messages.append(
            Message(
                role=role,
                content=content,
                tool_call_id=raw_msg.get("tool_call_id"),
                details=details or None,
            )
        )

    if not messages:
        raise RuntimeError("mini-swe-agent produced no importable messages")
    return Trajectory(messages=messages)


async def trajectory_from_mini_swe_agent(
    prompt: str,
    sample_id: str,
    sample_data: dict[str, Any],
    *,
    workspace: SandboxWorkspaceResource | LocalWorkspaceResource | None = None,
    run_config: Any | None = None,
    model: str | None = None,
    timeout_seconds: float = 600.0,
    config_spec: list[str] | None = None,
    agent_class: str | None = None,
    environment_class: str | None = None,
    yolo: bool = True,
) -> ExternalAttemptArtifact:
    del sample_data

    workdir = (
        Path(workspace.working_dir).resolve() if workspace is not None else Path.cwd().resolve()
    )
    output_path = workdir / f".mini-swe-agent-{_sample_id_slug(sample_id)}.traj.json"
    if output_path.exists():
        output_path.unlink()

    cli = shutil.which("mini")
    if cli is not None:
        cmd = [cli]
    elif shutil.which("uvx") is not None:
        cmd = ["uvx", "--from", "mini-swe-agent", "mini"]
    else:
        raise RuntimeError("mini-swe-agent CLI not found. Install `mini` or make `uvx` available.")

    cmd.extend(["--task", prompt, "--output", str(output_path)])
    if model is not None:
        cmd.extend(["--model", model])
    if yolo:
        cmd.append("--yolo")
    cmd.append("--exit-immediately")
    for spec in config_spec or ():
        cmd.extend(["--config", spec])
    if agent_class:
        cmd.extend(["--agent-class", agent_class])
    if environment_class:
        cmd.extend(["--environment-class", environment_class])

    env = os.environ.copy()
    env.setdefault("MSWEA_CONFIGURED", "true")
    if model is not None:
        env["MSWEA_MODEL_NAME"] = model
    raw_line_handler = _make_raw_driver_line_handler(run_config, driver="mini_swe_agent")
    progress_emitter = _make_external_progress_emitter(run_config, driver="mini_swe_agent")
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    latest_payload: dict[str, Any] | None = None
    latest_messages = 0

    async def _consume_stream(
        stream: trio.abc.ReceiveStream | None,
        *,
        sink: list[str],
    ) -> None:
        if stream is None:
            return
        buffer = ""
        while True:
            chunk = await stream.receive_some(4096)
            if not chunk:
                break
            buffer += chunk.decode("utf-8", errors="replace")
            while True:
                newline = buffer.find("\n")
                if newline < 0:
                    break
                line = buffer[:newline]
                buffer = buffer[newline + 1 :]
                sink.append(line)
                if raw_line_handler is not None:
                    await raw_line_handler(line)
        if buffer:
            sink.append(buffer)
            if raw_line_handler is not None:
                await raw_line_handler(buffer)

    async def _poll_output_file() -> None:
        nonlocal latest_payload, latest_messages
        last_mtime_ns: int | None = None
        while True:
            await trio.sleep(1.0)
            if not output_path.is_file():
                continue
            stat = output_path.stat()
            if stat.st_mtime_ns == last_mtime_ns:
                continue
            last_mtime_ns = stat.st_mtime_ns
            try:
                payload = json.loads(output_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(payload, dict):
                continue
            latest_payload = payload
            latest_messages = len(payload.get("messages", []))
            if progress_emitter is not None:
                await progress_emitter(
                    "external_artifact_progress",
                    {
                        "trajectory_path": str(output_path),
                        "messages": latest_messages,
                        "bytes": stat.st_size,
                    },
                )

    proc = await trio.lowlevel.open_process(
        cmd,
        cwd=str(workdir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    with trio.move_on_after(timeout_seconds) as cancel_scope:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(partial(_consume_stream, proc.stdout, sink=stdout_lines))
            nursery.start_soon(partial(_consume_stream, proc.stderr, sink=stderr_lines))
            nursery.start_soon(_poll_output_file)
            await proc.wait()
            nursery.cancel_scope.cancel()

    if cancel_scope.cancelled_caught:
        proc.kill()
        await proc.wait()
        raise TimeoutError(f"mini-swe-agent run timed out after {timeout_seconds:.1f}s")

    stdout_text = "\n".join(stdout_lines).strip()
    stderr_text = "\n".join(stderr_lines).strip()
    combined_output = "\n".join(part for part in (stdout_text, stderr_text) if part).strip()

    if not output_path.is_file():
        raise RuntimeError(
            "mini-swe-agent did not write a trajectory file.\n"
            f"stdout:\n{stdout_text}\n\nstderr:\n{stderr_text}"
        )

    if latest_payload is None:
        payload = json.loads(output_path.read_text())
    else:
        payload = latest_payload
    trajectory = _mini_swe_agent_output_to_trajectory(payload)
    info = payload.get("info", {})
    metadata: dict[str, Any] = {
        "runtime": "mini_swe_agent",
        "driver": "mini_swe_agent",
        "cwd": str(workdir),
        "trajectory_path": str(output_path),
        "exit_status": info.get("exit_status"),
        "submission": info.get("submission"),
    }
    if model is not None:
        metadata["model"] = model
    metadata["returncode"] = proc.returncode
    metadata["message_count"] = latest_messages or len(payload.get("messages", []))
    if combined_output:
        metadata["mini_swe_agent_output"] = combined_output[-8000:]

    status = Status.COMPLETED if proc.returncode == 0 else Status.ABORTED
    return ExternalAttemptArtifact(
        trajectory=trajectory,
        metadata=metadata,
        status=status,
    )
