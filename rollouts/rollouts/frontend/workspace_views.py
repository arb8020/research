from __future__ import annotations

import importlib.util as _ilu
import json
import sys as _sys
from pathlib import Path
from typing import Any

from .artifacts import load_sample_payload_from_trace_dir


def load_workspace_response(trace_dir: Path, sample_id: str) -> dict[str, Any]:
    """Build the workspace API payload for one sample."""
    _mod_name = "_rollouts_workspace_snapshot"
    if _mod_name not in _sys.modules:
        _ws_path = Path(__file__).parent.parent / "eval" / "workspace_snapshot.py"
        _spec = _ilu.spec_from_file_location(_mod_name, _ws_path)
        assert _spec is not None and _spec.loader is not None
        _ws_mod = _ilu.module_from_spec(_spec)
        _sys.modules[_mod_name] = _ws_mod
        _spec.loader.exec_module(_ws_mod)
    _ws_mod = _sys.modules[_mod_name]
    reconstruct_workspace_snapshots = _ws_mod.reconstruct_workspace_snapshots
    snapshots_to_api_response = _ws_mod.snapshots_to_api_response

    sample_data = load_sample_payload_from_trace_dir(trace_dir, sample_id)
    trajectory = sample_data.get("trajectory", {})
    messages = list(trajectory.get("messages", []))

    events_path = trace_dir / "events.jsonl"
    live_snapshots = []
    if events_path.exists():
        with events_path.open() as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    evt = json.loads(stripped)
                    if (
                        evt.get("type") == "workspace_snapshot"
                        and evt.get("sample_id") == sample_id
                    ):
                        live_snapshots.append(evt)
                except json.JSONDecodeError:
                    pass

    if live_snapshots:
        # TODO: implement live snapshot deserialization
        pass

    initial_files: dict[str, str] = {}
    cwd = "/workspace"
    try:
        meta = sample_data.get("metadata", {})
        challenge_root = meta.get("challenge_root") or meta.get("workspace_dir")
        cwd = meta.get("cwd") or meta.get("workspace_dir") or "/workspace"
        if challenge_root:
            cr = Path(challenge_root)
            if cr.exists():
                for p in cr.rglob("*.py"):
                    if p.stat().st_size < 200_000:
                        rel = str(p.relative_to(cr))
                        try:
                            initial_files[rel] = p.read_text(errors="replace")
                        except OSError:
                            pass
    except OSError:
        pass

    snapshots, line_history = reconstruct_workspace_snapshots(
        messages,
        initial_files=initial_files or None,
        cwd=cwd,
    )

    def _strip_cwd(path: str) -> str:
        if cwd and path.startswith(cwd + "/"):
            return path[len(cwd) + 1 :]
        return path

    for snap in snapshots:
        snap.files = {_strip_cwd(k): v for k, v in snap.files.items()}

    stripped_line_history = {_strip_cwd(k): v for k, v in line_history.items()}
    return snapshots_to_api_response(snapshots, stripped_line_history, source="reconstructed")
