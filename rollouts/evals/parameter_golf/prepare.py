from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.parameter_golf.common import (
    build_rollouts_sdk_command,
    default_sample,
    materialize_workspace,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a Parameter Golf interactive workspace")
    parser.add_argument("--sample", default="baseline", help="Sample id (currently only 'baseline')")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of prose")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    sample = default_sample()
    if args.sample != sample["id"]:
        raise ValueError(f"Unknown sample {args.sample!r}. Available: {sample['id']}")

    workspace = materialize_workspace(sample)
    payload = {
        "sample_id": workspace.sample_id,
        "source_dir": str(workspace.source_dir),
        "workspace_dir": str(workspace.workspace_dir),
        "prompt_path": str(workspace.prompt_path),
        "metadata_path": str(workspace.metadata_path),
        "rollouts_sdk_command": build_rollouts_sdk_command(workspace),
        "claude_code_launch_config": str(
            Path(__file__).resolve().parent / "interactive_eval.py"
        ),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Workspace: {workspace.workspace_dir}")
        print(f"Prompt: {workspace.prompt_path}")
        print("Run with rollouts SDK:")
        print(f"  {payload['rollouts_sdk_command']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
