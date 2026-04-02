"""Modal image/materialization helpers owned by the broker boundary.

These helpers lower the shared runtime/image contract into Modal image builder
calls. They are provider concerns, not workload execution concerns.

This still imports Rollouts-side image/runtime contracts. That dependency is
acceptable for now because the denotation is still shared there; the important
ownership move is that provider-specific lowering no longer lives in
``rollouts.modal_runner``.
"""

from __future__ import annotations

import logging
import shlex
from pathlib import Path
from typing import Any

import trio
from rollouts.image_spec import (
    ImageManifest,
    image_manifest_for_spec,
    infer_cuda_version,
    manifest_write_command,
    resolve_image_for_provisioning,
)
from rollouts.install_probes import (
    apt_install_probe_command,
    command_looks_like_install,
    python_install_probe_command,
    python_runtime_contract_snapshot_command,
    python_runtime_contract_verify_command,
)

logger = logging.getLogger(__name__)

HF_CACHE_DIR = "/root/.cache/huggingface"
UV_BIN = "/root/.local/bin/uv"
IMAGE_VENV_DIR = "/opt/venvs/rollouts"
IMAGE_VENV_PYTHON = f"{IMAGE_VENV_DIR}/bin/python"
# Inference venv for split_env: trainer and inference server use separate venvs.
# When split_env is implemented in argus/run.py, the launcher must:
#   1. Build a second venv at this path from InferenceConfig.deps
#   2. Set ROLLOUTS_INFERENCE_PYTHON=INFERENCE_VENV_PYTHON in the trainer process env
# weight_sync._python_module_launch reads ROLLOUTS_INFERENCE_PYTHON to find the
# inference interpreter. See the TODO there for the proper long-term fix.
INFERENCE_VENV_DIR = "/opt/venvs/inference"
INFERENCE_VENV_PYTHON = f"{INFERENCE_VENV_DIR}/bin/python"
MODAL_IMAGE_BUILD_HEARTBEAT_S = 15.0
MODAL_IMAGE_BUILD_LOG_LINE_LIMIT = 200
MODAL_IMAGE_BUILD_LOG_CHAR_LIMIT = 1000
MODAL_IMAGE_BUILD_LOG_FETCH_TIMEOUT_S = 30.0
MODAL_IMAGE_BUILD_FAILURE_TAIL_LINES = 40


def build_modal_image(modal: Any, deps: Any, gpu_type: str) -> Any:
    """Build a Modal image from the shared dependency contract."""

    spec = deps.resolved_image(gpu_type)
    overlay = deps.resolved_runtime_overlay()
    cuda_version = infer_cuda_version(gpu_type, spec.pip_index_url)
    if spec.python_runtime == "image_owned":
        image_python = spec.python_executable
        image_path_prefix = (
            "/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        )
    else:
        image_python = IMAGE_VENV_PYTHON
        image_path_prefix = (
            f"{IMAGE_VENV_DIR}/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:"
            "/usr/sbin:/usr/bin:/sbin:/bin"
        )

    if spec.source_type == "registry":
        if spec.python_runtime == "image_owned":
            image = modal.Image.from_registry(spec.source_ref)
        else:
            image = modal.Image.from_registry(spec.source_ref, add_python=spec.python_version)
    elif spec.source_type == "dockerfile_path":
        dockerfile_path = Path(spec.source_ref)
        dockerfile_kwargs = {
            "context_dir": spec.context_dir or str(dockerfile_path.parent),
            "build_args": spec.build_args,
        }
        if spec.python_runtime == "image_owned":
            image = modal.Image.from_dockerfile(dockerfile_path, **dockerfile_kwargs)
        else:
            image = modal.Image.from_dockerfile(
                dockerfile_path,
                add_python=spec.python_version,
                **dockerfile_kwargs,
            )
    else:
        raise ValueError(
            f"Modal provider does not know how to build image source_type={spec.source_type!r}"
        )

    if spec.system_packages:
        image = image.apt_install(*spec.system_packages)
        image = image.run_commands(
            apt_install_probe_command("image-system-packages", packages=spec.system_packages)
        )

    image = image.run_commands(
        "which curl >/dev/null 2>&1 || (apt-get update && apt-get install -y curl)",
        "if [ ! -x /root/.local/bin/uv ]; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi",
    )

    if spec.python_runtime == "managed_venv":
        image = image.run_commands(
            f"{UV_BIN} python install {spec.python_version}",
            f"{UV_BIN} venv {IMAGE_VENV_DIR} --python {spec.python_version}",
        )
    else:
        image = image.run_commands(f"{spec.python_executable} -c 'import sys; print(sys.version)'")

    def _uv_install_command(
        packages: tuple[str, ...],
        *,
        index_url: str | None,
        extra_index_url: str | None,
        pre: bool,
    ) -> str:
        parts = [UV_BIN, "pip", "install", "--compile-bytecode"]
        if spec.python_runtime == "image_owned":
            parts.append("--system")
        else:
            parts.extend(["--python", image_python])
        if index_url:
            parts.extend(["--index-url", index_url])
        if extra_index_url:
            parts.extend(["--extra-index-url", extra_index_url])
        if pre:
            parts.extend(["--prerelease", "allow"])
        parts.extend(packages)
        return shlex.join(parts)

    if spec.pip_packages:
        image = image.run_commands(
            _uv_install_command(
                spec.pip_packages,
                index_url=spec.pip_index_url,
                extra_index_url=spec.pip_extra_index_url,
                pre=spec.pip_prerelease,
            ),
            f"{python_install_probe_command('image-pip-packages', python_bin=image_python, uv_bin=UV_BIN)} && "
            f"{python_runtime_contract_snapshot_command('image-pip-packages', python_bin=image_python)}",
        )

    for cmd in spec.build_commands:
        image = image.run_commands(cmd)
        if command_looks_like_install(cmd):
            image = image.run_commands(
                f"{python_install_probe_command('image-build-command-post-install', python_bin=image_python, uv_bin=UV_BIN)} && "
                f"{python_runtime_contract_verify_command('image-build-command-post-install', python_bin=image_python)}"
            )

    if overlay.system_packages:
        image = image.apt_install(*overlay.system_packages)
        image = image.run_commands(
            apt_install_probe_command("overlay-system-packages", packages=overlay.system_packages)
        )

    if overlay.pip_packages:
        image = image.run_commands(
            _uv_install_command(
                overlay.pip_packages,
                index_url=overlay.pip_index_url or spec.pip_index_url,
                extra_index_url=overlay.pip_extra_index_url or spec.pip_extra_index_url,
                pre=overlay.pip_prerelease or spec.pip_prerelease,
            ),
            f"{python_install_probe_command('overlay-pip-packages', python_bin=image_python, uv_bin=UV_BIN)} && "
            f"{python_runtime_contract_snapshot_command('overlay-pip-packages', python_bin=image_python)}",
        )

    for cmd in overlay.commands:
        image = image.run_commands(cmd)
        if command_looks_like_install(cmd):
            image = image.run_commands(
                f"{python_install_probe_command('overlay-command-post-install', python_bin=image_python, uv_bin=UV_BIN)} && "
                f"{python_runtime_contract_verify_command('overlay-command-post-install', python_bin=image_python)}"
            )

    image = image.run_commands("echo 'rollouts-build-v4-uv'")

    env_vars = {
        "HF_HOME": HF_CACHE_DIR,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PATH": image_path_prefix,
        "PYTHONPATH": "/root/Megatron-LM:/root",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        **spec.env,
        **overlay.env,
    }
    image = image.env(env_vars)

    manifest: ImageManifest = image_manifest_for_spec(
        spec,
        image_name=f"modal-{gpu_type.lower()}",
        cuda_version=cuda_version,
        resolved_image_ref=(
            resolve_image_for_provisioning(spec).resolved_ref
            if spec.source_type == "registry"
            else None
        ),
        features=overlay.features,
        installed_groups=overlay.installed_groups,
        env=env_vars,
        paths={"megatron_root": "/root/Megatron-LM"},
    )
    image = image.run_commands(manifest_write_command(manifest, spec.manifest_path))
    return image


def add_inference_venv_to_image(modal: Any, image: Any, inference_deps: Any, gpu_type: str) -> Any:
    """Layer an inference venv onto an existing trainer image (split_env, same base image).

    Installs inference_deps into INFERENCE_VENV_DIR on top of the trainer image.
    The trainer image is unchanged; inference processes use INFERENCE_VENV_PYTHON.

    Precondition: inference_deps.base_image == trainer_deps.base_image.
    For different base images (e.g. TRT-LLM), use two separate sandboxes instead.
    """
    spec = inference_deps.resolved_image(gpu_type)
    assert spec.source_type not in ("registry", "dockerfile_path") or spec.source_ref == getattr(
        image, "_spec_base_image", None
    ), (
        "add_inference_venv_to_image requires the same base image as the trainer. "
        "For a different base image use two separate sandboxes."
    )

    def _uv_install_command(packages: tuple[str, ...], *, index_url: str | None, extra_index_url: str | None, pre: bool) -> str:
        parts = [UV_BIN, "pip", "install", "--compile-bytecode", "--python", INFERENCE_VENV_PYTHON]
        if index_url:
            parts.extend(["--index-url", index_url])
        if extra_index_url:
            parts.extend(["--extra-index-url", extra_index_url])
        if pre:
            parts.extend(["--prerelease", "allow"])
        parts.extend(packages)
        return shlex.join(parts)

    image = image.run_commands(
        f"{UV_BIN} venv {INFERENCE_VENV_DIR} --python {spec.python_version or '3.11'}",
    )

    if spec.pip_packages:
        image = image.run_commands(
            _uv_install_command(
                spec.pip_packages,
                index_url=spec.pip_index_url,
                extra_index_url=spec.pip_extra_index_url,
                pre=spec.pip_prerelease,
            ),
        )

    for cmd in spec.build_commands:
        image = image.run_commands(cmd)

    overlay = inference_deps.resolved_runtime_overlay()
    if overlay.pip_packages:
        image = image.run_commands(
            _uv_install_command(
                overlay.pip_packages,
                index_url=overlay.pip_index_url or spec.pip_index_url,
                extra_index_url=overlay.pip_extra_index_url or spec.pip_extra_index_url,
                pre=overlay.pip_prerelease or spec.pip_prerelease,
            ),
        )
    for cmd in overlay.commands:
        image = image.run_commands(cmd)

    return image


def _trim_modal_build_log_line(line: str) -> str:
    trimmed = line.rstrip()
    if len(trimmed) <= MODAL_IMAGE_BUILD_LOG_CHAR_LIMIT:
        return trimmed
    return trimmed[: MODAL_IMAGE_BUILD_LOG_CHAR_LIMIT - 3] + "..."


async def _emit_private_modal_image_logs(
    image: Any,
    emit: Any,
) -> list[str]:
    """Best-effort image build event capture via Modal ImageJoinStreaming."""

    image_id = getattr(image, "object_id", None)
    if not image_id:
        emit("modal_image_build_logs_unavailable", reason="missing_image_id")
        return []

    client = getattr(image, "client", None)
    stub = getattr(client, "stub", None)
    join_stream = getattr(stub, "ImageJoinStreaming", None)
    if client is None or join_stream is None:
        emit(
            "modal_image_build_logs_unavailable",
            image_id=image_id,
            reason="missing_image_join_stream",
        )
        return []

    emit("modal_image_build_logs_fetch_start", image_id=image_id)
    lines_emitted = 0
    truncated = False
    progress_updates = 0
    last_entry_id = ""
    emitted_lines: list[str] = []

    try:
        import trio_asyncio
        from modal_proto import api_pb2

        terminal_status: str | None = None

        async def _consume_stream() -> list[str]:
            nonlocal lines_emitted, truncated, progress_updates, last_entry_id, terminal_status
            request = api_pb2.ImageJoinStreamingRequest(
                image_id=image_id,
                timeout=55,
                last_entry_id=last_entry_id,
                include_logs_for_finished=True,
            )

            async for response in join_stream.unary_stream(request):
                if response.entry_id:
                    last_entry_id = response.entry_id
                if response.result.status:
                    terminal_status = api_pb2.GenericResult.GenericStatus.Name(
                        response.result.status
                    )
                for task_log in response.task_logs:
                    progress = task_log.task_progress
                    if progress.pos or progress.len:
                        progress_updates += 1
                        emit(
                            "modal_image_build_progress",
                            image_id=image_id,
                            progress_type=api_pb2.ProgressType.Name(progress.progress_type),
                            pos=int(progress.pos),
                            total=int(progress.len),
                        )
                    elif task_log.data:
                        if lines_emitted >= MODAL_IMAGE_BUILD_LOG_LINE_LIMIT:
                            truncated = True
                            continue
                        line = _trim_modal_build_log_line(task_log.data)
                        if not line:
                            continue
                        logger.info("[modal image] %s", line)
                        emit("modal_image_build_log", image_id=image_id, line=line)
                        emitted_lines.append(line)
                        lines_emitted += 1
                if terminal_status is not None:
                    return emitted_lines

            return emitted_lines

        async with trio_asyncio.open_loop():
            with trio.move_on_after(MODAL_IMAGE_BUILD_LOG_FETCH_TIMEOUT_S) as scope:
                emitted_lines = await trio_asyncio.aio_as_trio(_consume_stream())
        if scope.cancelled_caught:
            emit(
                "modal_image_build_logs_fetch_timeout",
                image_id=image_id,
                timeout_sec=MODAL_IMAGE_BUILD_LOG_FETCH_TIMEOUT_S,
                line_count=lines_emitted,
                progress_updates=progress_updates,
                truncated=truncated,
            )
            return emitted_lines
    except Exception as exc:
        emit(
            "modal_image_build_logs_fetch_failed",
            image_id=image_id,
            error=f"{type(exc).__name__}: {exc}",
            line_count=lines_emitted,
            progress_updates=progress_updates,
            truncated=truncated,
        )
        logger.warning("Failed to fetch Modal image build logs for %s: %s", image_id, exc)
        return emitted_lines

    emit(
        "modal_image_build_logs_fetch_finished",
        image_id=image_id,
        line_count=lines_emitted,
        progress_updates=progress_updates,
        truncated=truncated,
        terminal_status=terminal_status,
    )
    return emitted_lines


def _raise_modal_image_build_failure(exc: Exception, image: Any, log_lines: list[str]) -> None:
    image_id = getattr(image, "object_id", None)
    if not log_lines:
        raise exc

    tail = "\n".join(log_lines[-MODAL_IMAGE_BUILD_FAILURE_TAIL_LINES:])
    message = (
        f"Modal image build failed for {image_id or '<unknown-image>'}: {exc}\n\n"
        "Recent Modal image build logs:\n"
        f"{tail}"
    )
    raise RuntimeError(message) from exc


async def eager_build_modal_image(
    image: Any,
    app: Any,
    emit: Any,
) -> Any:
    """Build the Modal image explicitly before sandbox creation."""

    result: dict[str, Any] = {}
    start = trio.current_time()
    image_id = getattr(image, "object_id", None)
    emit("modal_image_build_start", image_id=image_id)

    async def _build_task() -> None:
        try:
            import trio_asyncio

            result["image"] = await trio_asyncio.aio_as_trio(image.build.aio(app))
        except Exception as exc:
            result["error"] = exc

    async with trio.open_nursery() as nursery:
        nursery.start_soon(_build_task)
        while "image" not in result and "error" not in result:
            elapsed = trio.current_time() - start
            emit(
                "modal_image_build_heartbeat",
                image_id=getattr(image, "object_id", None),
                elapsed_sec=round(elapsed, 3),
            )
            await trio.sleep(MODAL_IMAGE_BUILD_HEARTBEAT_S)
        nursery.cancel_scope.cancel()

    if "error" in result:
        exc = result["error"]
        elapsed = trio.current_time() - start
        emit(
            "modal_image_build_failed",
            image_id=getattr(image, "object_id", None),
            elapsed_sec=round(elapsed, 3),
            error=f"{type(exc).__name__}: {exc}",
        )
        log_lines = await _emit_private_modal_image_logs(image, emit)
        _raise_modal_image_build_failure(exc, image, log_lines)

    built_image = result["image"]
    elapsed = trio.current_time() - start
    emit(
        "modal_image_build_finished",
        image_id=getattr(built_image, "object_id", None),
        elapsed_sec=round(elapsed, 3),
    )

    await _emit_private_modal_image_logs(built_image, emit)
    return built_image
