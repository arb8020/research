"""Shared utilities for RL example remote execution."""

from __future__ import annotations


async def run_remote(
    script_path: str,
    keep_alive: bool = False,
    node_id: str | None = None,
    use_tui: bool = False,
    tui_debug: bool = False,
    fire_and_forget: bool = False,
    gpu_count: int = 1,
    gpu_type: str = "A100",
    container_disk_gb: int = 100,
) -> None:
    """Delegate remote execution to `rollouts.run`."""
    from rollouts.run import run_remote as run_remote_impl

    await run_remote_impl(
        script_path=script_path,
        keep_alive=keep_alive,
        node_id=node_id,
        tui=use_tui,
        tail=not use_tui and not tui_debug,
        block=not fire_and_forget,
        gpu_count=gpu_count,
        gpu_type=gpu_type,
        allow_dirty=True,
        skip_hf_token_check=True,
        container_disk_gb=container_disk_gb,
        raw_script=True,
    )
