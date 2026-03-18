#!/usr/bin/env python3
"""HTTP server for agent dev loop tool.

Provides:
- Static file serving (index.html)
- Config generation API
- Trace viewing API
- Live streaming of agent runs

Usage:
    python -m rollouts.frontend.server
    python -m rollouts.frontend.server --port 8080
    python -m rollouts.frontend.server --project ~/myproject
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import threading
import time
import webbrowser
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Global process registry for tracking running agent evaluations
_active_runs = {}  # run_id -> {process, config_name, start_time, status, output_lines, exit_code}
_run_counter = 0
_run_lock = threading.Lock()

# Semaphore for limiting concurrent runs (default: 2 concurrent runs)
_max_concurrent_runs = 2
_run_semaphore = threading.Semaphore(_max_concurrent_runs)


class DevLoopServer(SimpleHTTPRequestHandler):
    """HTTP server for agent dev loop tool.

    Serves static files and provides API endpoints for:
    - /api/configs - List available configs
    - /api/traces - List/load evaluation traces
    - /api/generate - Generate new config files
    """

    # Class variables (set by main())
    project_root: Path = Path.cwd()
    results_dir: Path = Path.cwd() / "results"
    known_results_dirs: list[Path] = []

    def log_message(self, format: str, *args: object) -> None:
        """Override to use our logger instead of stderr."""
        logger.info(f"{self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        """Handle GET requests."""
        try:
            self._do_GET_inner()
        except Exception:
            logger.exception(f"Unhandled error in GET {self.path}")
            try:
                self.send_error(500, "Internal server error")
            except Exception:
                pass

    def _do_GET_inner(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._serve_ui_index()
        elif path.startswith("/assets/"):
            self._serve_ui_static(path)
        elif path == "/api/configs":
            self._list_configs()
        elif path == "/api/traces":
            self._list_traces()
        elif path.startswith("/api/trace/") and "/sample/" in path and path.endswith("/workspace"):
            # /api/trace/{id}/sample/{sampleId}/workspace — environment state snapshots
            rest = path.split("/api/trace/")[1]
            trace_id, _, rest2 = rest.partition("/sample/")
            sample_id = rest2.removesuffix("/workspace")
            self._get_workspace(trace_id, sample_id)
        elif path.startswith("/api/trace/") and "/sample/" in path:
            # Extract trace ID and sample ID from /api/trace/{id}/sample/{sampleId}
            rest = path.split("/api/trace/")[1]
            trace_id, _, sample_id = rest.partition("/sample/")
            self._get_sample(trace_id, sample_id)
        elif path.startswith("/api/trace/"):
            # Extract trace ID from path like /api/trace/02_agent_multiturn_20231114_143022
            trace_id = path.split("/api/trace/")[1]
            self._get_trace(trace_id)
        elif path.startswith("/api/load-config/"):
            # Extract config name from path like /api/load-config/02_agent_multiturn
            config_name = path.split("/api/load-config/")[1]
            self._load_config(config_name)
        elif path.startswith("/api/dataset-preview/"):
            # Extract config name from path like /api/dataset-preview/01_agent_eval
            config_name = path.split("/api/dataset-preview/")[1]
            self._get_dataset_preview(config_name)
        elif path.startswith("/api/parse-messages/"):
            # Extract config name from path like /api/parse-messages/01_agent_eval
            config_name = path.split("/api/parse-messages/")[1]
            self._parse_messages(config_name)
        elif path.startswith("/api/parse-tools/"):
            # Extract config name from path like /api/parse-tools/01_agent_eval
            config_name = path.split("/api/parse-tools/")[1]
            self._parse_tools(config_name)
        elif path.startswith("/api/view-hook/"):
            # Extract config name from path like /api/view-hook/01_agent_eval
            config_name = path.split("/api/view-hook/")[1]
            self._view_hook(config_name)
        elif path.startswith("/api/view-environment/"):
            # Extract config name from path like /api/view-environment/01_agent_eval
            config_name = path.split("/api/view-environment/")[1]
            self._view_environment(config_name)
        elif path == "/api/models":
            self._list_models()
        elif path == "/api/list-datasets":
            self._list_datasets()
        elif path == "/api/runs":
            self._list_active_runs()
        elif path == "/api/results-dirs":
            self._list_results_dirs()
        elif path.startswith("/api/stream/"):
            # Extract run_id from path like /api/stream/run_1_1234567890
            run_id = path.split("/api/stream/")[1]
            self._stream_run_output(run_id)
        elif path.startswith("/api/watch/"):
            # Attach to an externally-launched run by results dir name
            run_id = path.split("/api/watch/")[1]
            self._watch_run(run_id)
        else:
            # Default behavior for other files
            super().do_GET()

    def do_POST(self) -> None:
        """Handle POST requests."""
        try:
            self._do_POST_inner()
        except Exception:
            logger.exception(f"Unhandled error in POST {self.path}")
            try:
                self.send_error(500, "Internal server error")
            except Exception:
                pass

    def _do_POST_inner(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/generate":
            self._generate_config()
        elif path == "/api/launch":
            self._launch_config()
        elif path == "/api/log":
            self._log_from_frontend()
        elif path == "/api/preview-dataset":
            self._preview_dataset_direct()
        elif path == "/api/set-results-dir":
            self._set_results_dir()
        elif path.startswith("/api/kill/"):
            # Extract run_id from path like /api/kill/run_1_1234567890
            run_id = path.split("/api/kill/")[1]
            self._kill_run(run_id)
        elif path.startswith("/api/delete-run/"):
            # Extract run_id from path like /api/delete-run/run_1_1234567890
            run_id = path.split("/api/delete-run/")[1]
            self._delete_run(run_id)
        else:
            self.send_error(404, "Not found")

    def _serve_ui_index(self) -> None:
        """Serve the React UI from ui/dist/index.html (falls back to legacy index.html)."""
        ui_dist = Path(__file__).parent / "ui" / "dist" / "index.html"
        legacy = Path(__file__).parent / "index.html"

        index_path = ui_dist if ui_dist.exists() else legacy

        if not index_path.exists():
            self.send_error(404, "index.html not found")
            return

        content = index_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_ui_static(self, path: str) -> None:
        """Serve static assets from ui/dist/assets/."""
        import mimetypes

        asset_path = Path(__file__).parent / "ui" / "dist" / path.lstrip("/")

        if not asset_path.exists() or not asset_path.is_file():
            self.send_error(404, f"Asset not found: {path}")
            return

        content = asset_path.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(asset_path))
        self.send_response(200)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        self.end_headers()
        self.wfile.write(content)

    def _serve_index(self) -> None:
        """Serve the main HTML file (legacy config builder)."""
        index_path = Path(__file__).parent / "index.html"

        if not index_path.exists():
            self.send_error(404, "index.html not found")
            return

        content = index_path.read_bytes()

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _list_configs(self) -> None:
        """List available config files in project."""
        configs_dir = self.project_root / "configs"

        if not configs_dir.exists():
            self._json_response([])
            return

        configs = []
        for config_file in sorted(configs_dir.glob("*.py")):
            # Skip __init__.py and other special files
            if config_file.stem.startswith("_"):
                continue

            configs.append({
                "name": config_file.stem,
                "path": str(config_file.relative_to(self.project_root)),
                "modified": config_file.stat().st_mtime,
            })

        self._json_response(configs)

    def _list_traces(self) -> None:
        """List available evaluation traces in results_dir."""
        results_dir = self.results_dir

        if not results_dir.exists():
            self._json_response([])
            return

        traces = []
        for trace_dir in sorted(results_dir.iterdir(), reverse=True):
            if not trace_dir.is_dir():
                continue

            # Check if it has a report.json (indicates it's a valid trace)
            report_path = trace_dir / "report.json"
            if not report_path.exists():
                continue

            # Load report to get summary info
            report = json.loads(report_path.read_text())

            traces.append({
                "id": trace_dir.name,
                "name": trace_dir.name,
                "timestamp": trace_dir.stat().st_mtime,
                "total_samples": report.get("total_samples", 0),
                "mean_reward": report.get("summary_metrics", {}).get("mean_reward", 0),
            })

        self._json_response(traces)

    def _get_trace(self, trace_id: str) -> None:
        """Load a specific evaluation trace."""
        trace_dir = self.results_dir / trace_id

        if not trace_dir.exists():
            self.send_error(404, f"Trace not found: {trace_id}")
            return

        report_path = trace_dir / "report.json"
        if not report_path.exists():
            self.send_error(404, f"No report.json in trace: {trace_id}")
            return

        # Load report
        report = json.loads(report_path.read_text())

        # Load trajectories (if they exist) - these are JSONL files with one event per line
        trajectories_dir = trace_dir / "trajectories"
        samples = []

        if trajectories_dir.exists():
            for traj_file in sorted(trajectories_dir.glob("*.jsonl")):
                # Parse JSONL - each line is a JSON object representing one turn/event
                messages = []
                rewards = []
                metadata = {}

                for line in traj_file.read_text().splitlines():
                    if not line.strip():
                        continue

                    event = json.loads(line)

                    # Extract messages from completions
                    if "messages" in event:
                        messages = event["messages"]

                    # Extract reward
                    if "rewards" in event:
                        rewards.append(event["rewards"])
                    elif "reward" in event:
                        rewards.append(event["reward"])

                    # Extract metadata
                    if "metadata" in event:
                        metadata = event["metadata"]

                samples.append({
                    "name": traj_file.stem,
                    "messages": messages,
                    "rewards": rewards[-1] if rewards else 0,
                    "metadata": metadata,
                })

        trace_data = {
            "id": trace_id,
            "name": report.get("config_name", trace_id),
            "total_samples": report.get("total_samples", len(samples)),
            "mean_reward": report.get("summary_metrics", {}).get("mean_reward", 0),
            "samples": samples,
            "report": report,
        }

        self._json_response(trace_data)

    def _get_sample(self, trace_id: str, sample_id: str) -> None:
        """Load a sample JSON, inlining trajectory from trajectory_path if needed.

        Some eval frameworks (e.g. charisma) store trajectory separately and
        only put a trajectory_path pointer in the sample JSON. We resolve it
        here so the frontend always receives a consistent shape with a
        trajectory: {completions, messages, rewards} field present.
        """
        sample_path = self.results_dir / trace_id / "samples" / f"{sample_id}.json"

        if not sample_path.exists():
            # Also try .jsonl — some evals only write the JSONL form
            jsonl_path = self.results_dir / trace_id / "samples" / f"{sample_id}.jsonl"
            if jsonl_path.exists():
                first_line = jsonl_path.read_text().splitlines()[0]
                sample = json.loads(first_line)
            else:
                self.send_error(404, f"Sample not found: {trace_id}/{sample_id}")
                return
        else:
            sample = json.loads(sample_path.read_text())

        # If trajectory is missing but trajectory_path is present, inline it
        if "trajectory" not in sample and "trajectory_path" in sample:
            traj_rel = sample["trajectory_path"]
            traj_file = self.results_dir / trace_id / traj_rel
            if traj_file.exists():
                lines = [l for l in traj_file.read_text().splitlines() if l.strip()]
                if lines:
                    last = json.loads(lines[-1])
                    if last.get("role"):
                        # New format: one message dict per line
                        sample["trajectory"] = {
                            "completions": [],
                            "messages": [json.loads(l) for l in lines],
                        }
                    else:
                        # Old format: last line is a {"messages": [...]} snapshot
                        sample["trajectory"] = last
                else:
                    sample["trajectory"] = {"completions": [], "messages": []}
            else:
                sample["trajectory"] = {"completions": [], "messages": []}

        # Normalise: ensure trajectory always has completions and messages keys
        traj = sample.get("trajectory", {})
        if "completions" not in traj:
            traj["completions"] = []
        if "messages" not in traj:
            traj["messages"] = []
        sample["trajectory"] = traj

        content = json.dumps(sample).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _get_workspace(self, trace_id: str, sample_id: str) -> None:
        """Return workspace snapshots and line history for a sample.

        Attempts to read live workspace_snapshot events from events.jsonl.
        Falls back to reconstructing from trajectory tool calls when none exist.
        """
        # Import workspace_snapshot directly by file path to avoid triggering
        # the full rollouts package import chain (which requires trio, etc.).
        # Must register in sys.modules before exec so @dataclass works in Python 3.13.
        import importlib.util as _ilu
        import sys as _sys

        _mod_name = "_rollouts_workspace_snapshot"
        if _mod_name not in _sys.modules:
            _ws_path = Path(__file__).parent.parent / "eval" / "workspace_snapshot.py"
            _spec = _ilu.spec_from_file_location(_mod_name, _ws_path)
            _ws_mod = _ilu.module_from_spec(_spec)
            _sys.modules[_mod_name] = _ws_mod
            _spec.loader.exec_module(_ws_mod)
        _ws_mod = _sys.modules[_mod_name]
        reconstruct_workspace_snapshots = _ws_mod.reconstruct_workspace_snapshots
        snapshots_to_api_response = _ws_mod.snapshots_to_api_response

        # Search primary and all known results dirs for the trace
        run_dir = None
        for search_dir in [self.results_dir] + list(self.__class__.known_results_dirs):
            candidate = search_dir / trace_id
            if candidate.is_dir():
                run_dir = candidate
                break
        if run_dir is None:
            self.send_error(404, f"Trace not found: {trace_id}")
            return

        trajectory_path = run_dir / "trajectories" / f"{sample_id}.jsonl"
        if not trajectory_path.exists():
            self.send_error(404, f"Trajectory not found: {sample_id}")
            return

        # Load trajectory messages
        messages = []
        with trajectory_path.open() as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    try:
                        messages.append(json.loads(stripped))
                    except json.JSONDecodeError:
                        pass

        # Check for live workspace_snapshot events in events.jsonl
        events_path = run_dir / "events.jsonl"
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
            # Live path: events were emitted during the run
            # Convert to WorkspaceSnapshot format
            # TODO: implement live snapshot deserialization
            # For now fall through to reconstruction
            pass

        # Reconstructed path: derive from trajectory tool calls
        # Load initial file state from sample metadata if available
        initial_files: dict[str, str] = {}
        sample_path = run_dir / "samples" / f"{sample_id}.json"
        cwd = "/workspace"
        if sample_path.exists():
            try:
                sample_meta = json.loads(sample_path.read_text())
                meta = sample_meta.get("metadata", {})
                challenge_root = meta.get("challenge_root") or meta.get("workspace_dir")
                cwd = meta.get("cwd") or meta.get("workspace_dir") or "/workspace"
                if challenge_root:
                    from pathlib import Path as _Path

                    cr = _Path(challenge_root)
                    if cr.exists():
                        for p in cr.rglob("*.py"):
                            if p.stat().st_size < 200_000:  # skip large files
                                rel = str(p.relative_to(cr))
                                try:
                                    initial_files[rel] = p.read_text(errors="replace")
                                except OSError:
                                    pass
            except (json.JSONDecodeError, OSError):
                pass

        snapshots, line_history = reconstruct_workspace_snapshots(
            messages,
            initial_files=initial_files or None,
            cwd=cwd,
        )

        # Strip cwd prefix from file keys for cleaner display
        def _strip_cwd(path: str) -> str:
            if cwd and path.startswith(cwd + "/"):
                return path[len(cwd) + 1 :]
            return path

        for snap in snapshots:
            snap.files = {_strip_cwd(k): v for k, v in snap.files.items()}

        stripped_line_history = {_strip_cwd(k): v for k, v in line_history.items()}

        response = snapshots_to_api_response(
            snapshots, stripped_line_history, source="reconstructed"
        )
        content = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def _load_config(self, config_name: str) -> None:
        """Load and parse an existing config file."""
        config_path = self.project_root / "configs" / f"{config_name}.py"

        if not config_path.exists():
            self.send_error(404, f"Config not found: {config_name}")
            return

        # Read the config file
        config_source = config_path.read_text()

        # Parse key settings from the config using simple regex/string matching
        # This is intentionally simple - just extracting common patterns
        import re

        config_data = {
            "name": config_name,
            "source": config_source,
        }

        # Extract model name
        model_match = re.search(r'model_name\s*[:=]\s*["\']([^"\']+)["\']', config_source)
        if model_match:
            config_data["model"] = model_match.group(1)

        # Extract temperature
        temp_match = re.search(r"temperature\s*[:=]\s*([0-9.]+)", config_source)
        if temp_match:
            config_data["temperature"] = float(temp_match.group(1))

        # Extract prepare_messages method (full function)
        prepare_msg_match = re.search(
            r"(def prepare_messages\(self.*?^    def \w+|def prepare_messages\(self.*?^class \w+|def prepare_messages\(self.*?$)",
            config_source,
            re.DOTALL | re.MULTILINE,
        )
        if prepare_msg_match:
            # Clean up and dedent the function
            func_text = prepare_msg_match.group(1)
            # Remove trailing class/def if captured
            func_text = re.sub(r"\n    (def |class )\w+.*$", "", func_text, flags=re.DOTALL)
            config_data["prepareMessages"] = func_text.strip()

        # Also extract just system_prompt for backward compatibility
        prompt_match = re.search(r'system_prompt\s*=\s*"""([^"]+)"""', config_source, re.DOTALL)
        if prompt_match:
            config_data["systemPrompt"] = prompt_match.group(1).strip()

        # Extract max turns
        turns_match = re.search(r"max_turns\s*[:=]\s*(\d+)", config_source)
        if turns_match:
            config_data["maxTurns"] = int(turns_match.group(1))

        # Extract num samples
        samples_match = re.search(r"num_samples\s*[:=]\s*(\d+)", config_source)
        if samples_match:
            config_data["numSamples"] = int(samples_match.group(1))

        # Extract seed
        seed_match = re.search(r"seed\s*[:=]\s*(\d+)", config_source)
        if seed_match:
            config_data["seed"] = int(seed_match.group(1))

        # Extract start_idx and end_idx
        start_match = re.search(r"start_idx\s*[:=]\s*(\d+)", config_source)
        if start_match:
            config_data["startIdx"] = int(start_match.group(1))

        end_match = re.search(r"end_idx\s*[:=]\s*(\d+)", config_source)
        if end_match:
            config_data["endIdx"] = int(end_match.group(1))

        # Extract environment-specific fields from both direct assignment and environment_config dict
        ssh_match = re.search(r'["\']ssh_target["\']\s*:\s*["\']([^"\']+)["\']', config_source)
        if not ssh_match:
            ssh_match = re.search(r'ssh_target\s*[:=]\s*["\']([^"\']+)["\']', config_source)
        if ssh_match:
            config_data["sshTarget"] = ssh_match.group(1)

        # Try to match cuda_device_ids as a list first
        gpu_list_match = re.search(r'["\']cuda_device_ids["\']\s*:\s*\[([^\]]+)\]', config_source)
        if not gpu_list_match:
            gpu_list_match = re.search(r"cuda_device_ids\s*[:=]\s*\[([^\]]+)\]", config_source)

        if gpu_list_match:
            # Parse list of GPU IDs
            cuda_device_ids_str = gpu_list_match.group(1)
            config_data["gpuIds"] = [
                int(x.strip()) for x in cuda_device_ids_str.split(",") if x.strip().isdigit()
            ]
        else:
            # Fallback to single gpu_id for backwards compatibility
            gpu_match = re.search(r'["\']gpu_id["\']\s*:\s*(\d+)', config_source)
            if not gpu_match:
                gpu_match = re.search(r"gpu_id\s*[:=]\s*(\d+)", config_source)
            if gpu_match:
                config_data["gpuIds"] = [int(gpu_match.group(1))]

        dataset_match = re.search(
            r'["\']dataset_path["\']\s*:\s*Path\(["\']([^"\']+)["\']\)', config_source
        )
        if not dataset_match:
            dataset_match = re.search(
                r'dataset_path\s*[:=]\s*Path\(["\']([^"\']+)["\']\)', config_source
            )
        if dataset_match:
            config_data["datasetPath"] = dataset_match.group(1)

        env_name_match = re.search(r'env_name\s*[:=]\s*["\']([^"\']+)["\']', config_source)
        if env_name_match:
            config_data["envName"] = env_name_match.group(1)

        # Parse tools from get_tools() method
        tools_section = re.search(
            r"def get_tools\(self\).*?return \[(.*?)\]", config_source, re.DOTALL
        )
        if tools_section:
            # This is a simplified parser - just check if it returns empty list or has tools
            tools_content = tools_section.group(1).strip()
            config_data["hasTools"] = len(tools_content) > 0 and "Tool(" in tools_content
        else:
            config_data["hasTools"] = False

        self._json_response(config_data)

    def _get_dataset_preview(self, config_name: str) -> None:
        """Get preview of dataset for a config - shows first sample with all fields."""
        import re

        config_path = self.project_root / "configs" / f"{config_name}.py"

        if not config_path.exists():
            self._json_response({"error": f"Config not found: {config_name}"})
            return

        # Read config to extract dataset path
        config_source = config_path.read_text()

        # Try to find dataset_path in the source
        dataset_match = re.search(
            r'["\']dataset_path["\']\s*:\s*Path\(["\']([^"\']+)["\']\)', config_source
        )
        if not dataset_match:
            dataset_match = re.search(
                r'dataset_path\s*[:=]\s*Path\(["\']([^"\']+)["\']\)', config_source
            )

        if not dataset_match:
            self._json_response({"error": "Could not find dataset_path in config"})
            return

        dataset_path_str = dataset_match.group(1)
        dataset_path = self.project_root / dataset_path_str

        if not dataset_path.exists():
            self._json_response({"error": f"Dataset not found: {dataset_path_str}"})
            return

        # Read first sample from dataset and count total
        try:
            dataset_size = 0
            if dataset_path.suffix == ".jsonl":
                # JSONL format - read first line and count total
                with dataset_path.open() as f:
                    first_line = f.readline()
                    sample = json.loads(first_line)
                    # Count total lines
                    dataset_size = 1 + sum(1 for _ in f)
            else:
                # JSON array format
                data = json.loads(dataset_path.read_text())
                if isinstance(data, list) and len(data) > 0:
                    sample = data[0]
                    dataset_size = len(data)
                else:
                    self._json_response({"error": "Dataset is empty or not a list"})
                    return

            # Extract fields and truncate long values for preview
            fields = list(sample.keys())
            preview_sample = {}
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 100:
                    preview_sample[key] = value[:100] + "..."
                else:
                    preview_sample[key] = value

            self._json_response({
                "datasetPath": dataset_path_str,
                "fields": fields,
                "sample": preview_sample,
                "datasetSize": dataset_size,
                "error": None,
            })

        except Exception as e:
            self._json_response({"error": f"Error reading dataset: {str(e)}"})

    def _parse_messages(self, config_name: str) -> None:
        """Parse prepare_messages() method from config to extract message list."""
        import re

        config_path = self.project_root / "configs" / f"{config_name}.py"

        if not config_path.exists():
            self._json_response({"error": f"Config not found: {config_name}"})
            return

        config_source = config_path.read_text()

        # Find prepare_messages function (can be standalone or method)
        method_match = re.search(
            r"def prepare_messages\(.*?\).*?:\s*\n(.*?)(?=\ndef |\nclass |\Z)",
            config_source,
            re.DOTALL,
        )

        if not method_match:
            self._json_response({"error": "Could not find prepare_messages() function"})
            return

        method_body = method_match.group(1)

        # Extract variable assignments for content
        variable_values = {}

        # Match triple-quoted strings (including multi-line)
        triple_quote_pattern = r'(\w+)\s*=\s*"""(.*?)"""'
        for match in re.finditer(triple_quote_pattern, method_body, re.DOTALL):
            var_name = match.group(1)
            var_value = match.group(2).strip()
            variable_values[var_name] = var_value

        # Match f-strings with triple quotes
        f_triple_quote_pattern = r'(\w+)\s*=\s*f"""(.*?)"""'
        for match in re.finditer(f_triple_quote_pattern, method_body, re.DOTALL):
            var_name = match.group(1)
            var_value = match.group(2).strip()
            variable_values[var_name] = var_value

        # Match sample_data field access
        sample_data_pattern = r'(\w+)\s*=\s*sample_data\[["\'](\w+)["\']\]'
        for match in re.finditer(sample_data_pattern, method_body):
            var_name = match.group(1)
            field_name = match.group(2)
            variable_values[var_name] = f"{{{field_name}}}"

        # Match simple string assignments
        simple_string_pattern = r'(\w+)\s*=\s*"([^"]+)"'
        for match in re.finditer(simple_string_pattern, method_body):
            var_name = match.group(1)
            var_value = match.group(2)
            # Don't override if already captured by triple-quote pattern
            if var_name not in variable_values:
                variable_values[var_name] = var_value

        # Find return statement with Message list
        return_match = re.search(r"return\s*\[(.*?)\]", method_body, re.DOTALL)

        if not return_match:
            self._json_response({"error": "Could not parse messages from return statement"})
            return

        messages_str = return_match.group(1)

        # Extract Message() calls
        messages = []
        message_pattern = (
            r'Message\(\s*role\s*=\s*["\'](\w+)["\']\s*,\s*content\s*=\s*(.*?)\s*\)(?=\s*(?:,|\]))'
        )

        for match in re.finditer(message_pattern, messages_str, re.DOTALL):
            role = match.group(1)
            content_expr = match.group(2).strip()

            # Handle different content formats
            if content_expr.startswith('f"""') or content_expr.startswith("f'''"):
                # f-string with triple quotes
                content = (
                    re.search(r'f["\']{{3}}(.*?)["\']{{3}}', content_expr, re.DOTALL)
                    .group(1)
                    .strip()
                )
            elif content_expr.startswith('"""') or content_expr.startswith("'''"):
                # Regular triple-quoted string
                content = (
                    re.search(r'["\']{{3}}(.*?)["\']{{3}}', content_expr, re.DOTALL)
                    .group(1)
                    .strip()
                )
            elif content_expr.startswith('f"') or content_expr.startswith("f'"):
                # f-string with single quotes
                quote_char = content_expr[1]
                content = content_expr[2 : content_expr.rfind(quote_char)]
            elif content_expr.startswith('"') or content_expr.startswith("'"):
                # Regular string
                quote_char = content_expr[0]
                content = content_expr[1 : content_expr.rfind(quote_char)]
            elif content_expr in variable_values:
                # Variable reference - look up the value
                content = variable_values[content_expr]
            else:
                # Complex expression - use as-is
                content = content_expr

            messages.append({"role": role, "content": content})

        self._json_response({"messages": messages, "error": None})

    def _parse_tools(self, config_name: str) -> None:
        """Parse get_tools() method from config to extract tool definitions."""
        import re

        config_path = self.project_root / "configs" / f"{config_name}.py"

        if not config_path.exists():
            self._json_response({"error": f"Config not found: {config_name}"})
            return

        config_source = config_path.read_text()

        # Find get_tools method
        method_match = re.search(
            r"def get_tools\(self\).*?:\s*\n(.*?)(?=\n    def |\nclass |\Z)",
            config_source,
            re.DOTALL,
        )

        if not method_match:
            self._json_response({"tools": [], "hasTools": False, "error": None})
            return

        method_body = method_match.group(1)

        # Find return statement
        return_match = re.search(r"return\s*\[(.*?)\]", method_body, re.DOTALL)

        if not return_match:
            self._json_response({"tools": [], "hasTools": False, "error": None})
            return

        tools_str = return_match.group(1)

        if not tools_str.strip() or "Tool(" not in tools_str:
            self._json_response({"tools": [], "hasTools": False, "error": None})
            return

        # Parse each Tool() definition
        tools = []
        tool_pattern = r'Tool\(\s*name\s*=\s*["\'](\w+)["\']\s*,\s*description\s*=\s*["\']{{3}}(.*?)["\']{{3}}\s*,\s*parameters\s*=\s*\{(.*?)\}\s*\)'

        for match in re.finditer(tool_pattern, tools_str, re.DOTALL):
            tool_name = match.group(1)
            tool_desc = match.group(2).strip()
            params_str = match.group(3)

            # Parse parameters
            parameters = []
            param_pattern = r'["\'](\w+)["\']\s*:\s*ToolParam\(\s*type\s*=\s*["\'](\w+)["\']\s*,\s*description\s*=\s*["\']([^"\']+)["\']\s*(?:,\s*required\s*=\s*(True|False))?\s*\)'

            for param_match in re.finditer(param_pattern, params_str):
                param_name = param_match.group(1)
                param_type = param_match.group(2)
                param_desc = param_match.group(3)
                param_required = param_match.group(4) != "False" if param_match.group(4) else True

                parameters.append({
                    "name": param_name,
                    "type": param_type,
                    "description": param_desc,
                    "required": param_required,
                })

            tools.append({"name": tool_name, "description": tool_desc, "parameters": parameters})

        self._json_response({"tools": tools, "hasTools": len(tools) > 0, "error": None})

    def _view_hook(self, config_name: str) -> None:
        """View on_assistant_message() implementation from environment."""
        import re

        config_path = self.project_root / "configs" / f"{config_name}.py"

        if not config_path.exists():
            self._json_response({"error": f"Config not found: {config_name}"})
            return

        config_source = config_path.read_text()

        # Try to find environment class name or environment file
        env_match = re.search(r"from\s+([\w.]+)\s+import\s+(\w+Environment)", config_source)

        if not env_match:
            self._json_response({"error": "Could not find environment import"})
            return

        env_module = env_match.group(1)
        _env_class = env_match.group(2)  # Available for class-specific handling

        # Convert module path to file path
        module_parts = env_module.split(".")
        env_file = self.project_root / Path(*module_parts[:-1]) / f"{module_parts[-1]}.py"

        if not env_file.exists():
            # Try without the last part
            env_file = self.project_root / f"{module_parts[0]}.py"

        if not env_file.exists():
            self._json_response({"error": f"Could not find environment file: {env_module}"})
            return

        env_source = env_file.read_text()

        # Find on_assistant_message method
        method_match = re.search(
            r"(async def on_assistant_message\(.*?\).*?:\s*\n.*?)(?=\n    async def |\n    def |\nclass |\Z)",
            env_source,
            re.DOTALL,
        )

        if not method_match:
            self._json_response({"error": "Could not find on_assistant_message() method"})
            return

        method_source = method_match.group(1).strip()

        self._json_response({"source": method_source, "error": None})

    def _view_environment(self, config_name: str) -> None:
        """View full environment source code."""
        import re

        config_path = self.project_root / "configs" / f"{config_name}.py"

        if not config_path.exists():
            self._json_response({"error": f"Config not found: {config_name}"})
            return

        config_source = config_path.read_text()

        # Try to find environment class name or environment file
        env_match = re.search(r"from\s+([\w.]+)\s+import\s+(\w+Environment)", config_source)

        if not env_match:
            self._json_response({"error": "Could not find environment import"})
            return

        env_module = env_match.group(1)
        _env_class = env_match.group(2)  # Available for class-specific handling

        # Convert module path to file path
        module_parts = env_module.split(".")
        env_file = self.project_root / Path(*module_parts[:-1]) / f"{module_parts[-1]}.py"

        if not env_file.exists():
            # Try without the last part
            env_file = self.project_root / f"{module_parts[0]}.py"

        if not env_file.exists():
            self._json_response({"error": f"Could not find environment file: {env_module}"})
            return

        # Read entire environment file
        env_source = env_file.read_text()

        self._json_response({
            "source": env_source,
            "file_path": str(env_file.relative_to(self.project_root)),
            "class_name": _env_class,
            "error": None,
        })

    def _list_models(self) -> None:
        """Fetch available models from OpenAI and Anthropic APIs."""
        import os
        import urllib.request

        models = []
        errors = []

        # Fetch OpenAI models
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            try:
                req = urllib.request.Request(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {openai_key}"},
                )
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    # Filter to relevant models (gpt-4*, o1*, o3*)
                    for model in data.get("data", []):
                        model_id = model.get("id", "")
                        if any(model_id.startswith(prefix) for prefix in ["gpt-4", "o1", "o3"]):
                            models.append({"id": model_id, "provider": "openai", "name": model_id})
            except Exception as e:
                errors.append(f"OpenAI: {str(e)}")

        # Fetch Anthropic models
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                req = urllib.request.Request(
                    "https://api.anthropic.com/v1/models",
                    headers={"x-api-key": anthropic_key, "anthropic-version": "2023-06-01"},
                )
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    for model in data.get("data", []):
                        model_id = model.get("id", "")
                        models.append({
                            "id": model_id,
                            "provider": "anthropic",
                            "name": model.get("display_name", model_id),
                        })
            except Exception as e:
                errors.append(f"Anthropic: {str(e)}")

        self._json_response({"models": models, "errors": errors if errors else None})

    def _list_datasets(self) -> None:
        """List available dataset files in the data directory."""
        datasets_dir = self.project_root / "data"

        if not datasets_dir.exists():
            self._json_response({"datasets": [], "error": "data/ directory not found"})
            return

        try:
            # Find all .json and .jsonl files
            datasets = []
            for file_path in datasets_dir.rglob("*"):
                if file_path.suffix in [".json", ".jsonl"] and file_path.is_file():
                    # Get relative path from project root
                    relative_path = file_path.relative_to(self.project_root)
                    datasets.append({
                        "path": str(relative_path),
                        "name": file_path.name,
                        "size": file_path.stat().st_size,
                    })

            # Sort by name
            datasets.sort(key=lambda x: x["name"])

            self._json_response({"datasets": datasets, "error": None})

        except Exception as e:
            self._json_response({"datasets": [], "error": f"Error listing datasets: {str(e)}"})

    def _preview_dataset_direct(self) -> None:
        """Preview a dataset by path directly (not via config)."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
            dataset_path_str = data.get("datasetPath", "")
        except json.JSONDecodeError:
            self._json_response({"error": "Invalid JSON in request body"})
            return

        if not dataset_path_str:
            self._json_response({"error": "Missing datasetPath in request"})
            return

        dataset_path = self.project_root / dataset_path_str

        if not dataset_path.exists():
            self._json_response({"error": f"Dataset not found: {dataset_path_str}"})
            return

        # Read first sample from dataset and count total
        try:
            dataset_size = 0
            if dataset_path.suffix == ".jsonl":
                # JSONL format - read first line and count total
                with dataset_path.open() as f:
                    first_line = f.readline()
                    if not first_line:
                        self._json_response({"error": "Dataset is empty"})
                        return
                    sample = json.loads(first_line)
                    # Count total lines
                    dataset_size = 1 + sum(1 for _ in f)
            else:
                # JSON array format
                data = json.loads(dataset_path.read_text())
                if isinstance(data, list) and len(data) > 0:
                    sample = data[0]
                    dataset_size = len(data)
                else:
                    self._json_response({"error": "Dataset is empty or not a list"})
                    return

            # Extract fields and truncate long values for preview
            fields = list(sample.keys())
            preview_sample = {}
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 100:
                    preview_sample[key] = value[:100] + "..."
                else:
                    preview_sample[key] = value

            self._json_response({
                "datasetPath": dataset_path_str,
                "fields": fields,
                "sample": preview_sample,
                "datasetSize": dataset_size,
            })

        except json.JSONDecodeError as e:
            self._json_response({"error": f"Invalid JSON in dataset: {str(e)}"})
        except Exception as e:
            self._json_response({"error": f"Error reading dataset: {str(e)}"})

    def _generate_config(self) -> None:
        """Generate a new config file from JSON payload."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            config_data = json.loads(body)
        except json.JSONDecodeError as e:
            self.send_error(400, f"Invalid JSON: {e}")
            return

        # Generate config file content
        config_text = self._build_config_file(config_data)

        # Save to file
        config_name = config_data.get("configName", "untitled_config")
        config_path = self.project_root / "configs" / f"{config_name}.py"

        # Ensure configs directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Write config file
        config_path.write_text(config_text)

        # Return JSON response with file path
        self._json_response({
            "success": True,
            "file_path": str(config_path.relative_to(self.project_root)),
            "config_name": config_name,
        })

    def _build_config_file(self, data: dict[str, Any]) -> str:
        """Build config file content from data.

        Args:
            data: Config parameters from frontend

        Returns:
            Python source code for config file
        """
        # Normalize cuda_device_ids: convert to list if needed
        if "cuda_device_ids" in data:
            cuda_device_ids = data["cuda_device_ids"]
            if not isinstance(cuda_device_ids, list):
                data["cuda_device_ids"] = [cuda_device_ids]

        # Check if we're building from a base config
        base_name = data.get("baseName")

        logger.debug(f"_build_config: baseName={base_name}, configName={data.get('configName')}")

        if base_name:
            # Load base config and modify it
            logger.debug(f"Building from base config: {base_name}")
            return self._build_from_base_config(data, base_name)
        else:
            # Build from scratch
            logger.debug("Building new config from template")
            return self._build_new_config(data)

    def _build_from_base_config(self, data: dict[str, Any], base_name: str) -> str:
        """Build config by copying and modifying base config."""
        base_path = self.project_root / "configs" / f"{base_name}.py"

        if not base_path.exists():
            # Fallback to new config
            logger.warning(f"Base config not found: {base_path}, falling back to template")
            return self._build_new_config(data)

        # Read base config
        config_source = base_path.read_text()

        # Check what environment class is in the base config
        import re

        env_import = re.search(r"from\s+([\w.]+)\s+import\s+(\w+Environment)", config_source)
        env_class = re.search(r"environment_class:\s*type\s*=\s*(\w+)", config_source)
        logger.debug(
            f"Base config has environment import: {env_import.group(0) if env_import else 'None'}"
        )
        logger.debug(
            f"Base config has environment_class: {env_class.group(1) if env_class else 'None'}"
        )

        # Replace specific values
        import re

        # Update model if changed
        if "model" in data:
            model_name = data["model"]
            config_source = re.sub(
                r'(model_name\s*[:=]\s*)["\']([^"\']+)["\']', f'\\1"{model_name}"', config_source
            )

            # Also update provider and api_key_env_var based on new model
            if "claude" in model_name.lower() or "anthropic" in model_name.lower():
                provider = "anthropic"
                api_key_env_var = "ANTHROPIC_API_KEY"
                api_base = "https://api.anthropic.com"  # SDK adds /v1 automatically
            elif (
                "gpt" in model_name.lower()
                or "o1" in model_name.lower()
                or "o3" in model_name.lower()
            ):
                provider = "openai"
                api_key_env_var = "OPENAI_API_KEY"
                api_base = "https://api.openai.com/v1"
            else:
                provider = "openai"
                api_key_env_var = "OPENAI_API_KEY"
                api_base = "https://api.openai.com/v1"

            # Update provider
            config_source = re.sub(
                r'(provider\s*[:=]\s*)["\']([^"\']+)["\']', f'\\1"{provider}"', config_source
            )

            # Update api_key_env_var
            config_source = re.sub(
                r'(api_key_env_var\s*[:=]\s*)["\']([^"\']+)["\']',
                f'\\1"{api_key_env_var}"',
                config_source,
            )

            # Update api_base
            config_source = re.sub(
                r'(api_base\s*[:=]\s*)["\']([^"\']+)["\']', f'\\1"{api_base}"', config_source
            )

        # Update temperature if changed
        if "temperature" in data:
            config_source = re.sub(
                r"(temperature\s*[:=]\s*)([0-9.]+)", f"\\g<1>{data['temperature']}", config_source
            )

        # Update system prompt if changed
        if "systemPrompt" in data:
            prompt = data["systemPrompt"]
            config_source = re.sub(
                r'(system_prompt\s*=\s*""")([^"]+)(""")',
                f"\\1{prompt}\\3",
                config_source,
                flags=re.DOTALL,
            )

        # Update max turns if changed
        if "maxTurns" in data:
            config_source = re.sub(
                r"(max_turns\s*[:=]\s*)(\d+)", f"\\g<1>{data['maxTurns']}", config_source
            )

        # Update num samples if changed
        if "numSamples" in data:
            config_source = re.sub(
                r"(num_samples\s*[:=]\s*)(\d+)", f"\\g<1>{data['numSamples']}", config_source
            )

        # Update environment fields if present
        env_fields = data.get("envFields", {})

        if "sshTarget" in env_fields:
            config_source = re.sub(
                r'(ssh_target\s*[:=]\s*)["\']([^"\']+)["\']',
                f'\\1"{env_fields["sshTarget"]}"',
                config_source,
            )

        # Handle both cuda_device_ids (list) and gpuId (single, backwards compat)
        if "cuda_device_ids" in data:
            cuda_device_ids = data["cuda_device_ids"]
            cuda_device_ids_str = str(cuda_device_ids)  # Convert list to string like [0, 1, 2]

            # Try to replace cuda_device_ids list first
            config_source = re.sub(
                r"(cuda_device_ids\s*[:=]\s*)\[[^\]]*\]",
                f"\\g<1>{cuda_device_ids_str}",
                config_source,
            )
            # Also try in dict format
            config_source = re.sub(
                r'(["\']cuda_device_ids["\']\s*:\s*)\[[^\]]*\]',
                f"\\g<1>{cuda_device_ids_str}",
                config_source,
            )
            # Fallback: replace old gpu_id with first GPU from list
            if cuda_device_ids:
                config_source = re.sub(
                    r"(gpu_id\s*[:=]\s*)(\d+)", f"\\g<1>{cuda_device_ids[0]}", config_source
                )
                config_source = re.sub(
                    r'(["\']gpu_id["\']\s*:\s*)(\d+)', f"\\g<1>{cuda_device_ids[0]}", config_source
                )

        if "datasetPath" in env_fields:
            config_source = re.sub(
                r'(dataset_path\s*[:=]\s*Path\()["\']([^"\']+)(["\'])',
                f'\\1"{env_fields["datasetPath"]}"\\3',
                config_source,
            )

        if "envName" in env_fields:
            config_source = re.sub(
                r'(env_name\s*[:=]\s*)["\']([^"\']+)["\']',
                f'\\1"{env_fields["envName"]}"',
                config_source,
            )

        # Update messages if provided
        if "messages" in data and data["messages"]:
            # Detect if prepare_messages is standalone or a method
            existing_match = re.search(r"def prepare_messages\((.*?)\)", config_source)
            is_standalone = existing_match and "self" not in existing_match.group(1)

            messages_code = self._generate_prepare_messages_method(
                data["messages"], is_standalone=is_standalone
            )
            # Replace the existing prepare_messages method
            config_source = re.sub(
                r"def prepare_messages\(.*?\).*?:\s*\n.*?(?=\ndef |\nclass |\Z)",
                messages_code,
                config_source,
                flags=re.DOTALL,
            )

        # Update tool descriptions if provided
        if "tools" in data and data["tools"]:
            config_source = self._update_tool_descriptions(config_source, data["tools"])

        # Add generation comment at top
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        config_name = data.get("configName", "custom_config")
        header = f'"""Agent configuration - {config_name}\n\nGenerated: {timestamp}\nBased on: {base_name}\n"""\n'

        # Replace existing docstring or prepend
        config_source = re.sub(r'^"""[^"]*"""\n', header, config_source)

        return config_source

    def _generate_prepare_messages_method(
        self, messages: list[dict], is_standalone: bool = True
    ) -> str:
        """Generate prepare_messages() code from message list.

        Args:
            messages: List of message dicts
            is_standalone: If True, generates standalone function. If False, generates class method.
        """
        lines = []
        if is_standalone:
            lines.append("def prepare_messages(sample_data: Dict[str, Any]) -> List[Message]:")
            lines.append('    """Prepare initial messages from dataset sample."""')
            indent = "    "
        else:
            lines.append(
                "    def prepare_messages(self, sample_data: dict[str, Any]) -> list[Message]:"
            )
            lines.append('        """Prepare initial messages for the agent."""')
            indent = "        "

        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]

            # Check if content has f-string placeholders like {field_name}
            has_placeholders = "{" in content and "}" in content

            # Escape any existing triple quotes in content
            content_escaped = content.replace('"""', r"\"\"\"")

            if has_placeholders:
                # Use f-string - convert {field} to {sample_data.get('field', '')}
                import re

                def replace_placeholder(match: re.Match[str]) -> str:
                    field = match.group(1)
                    return f"{{sample_data.get('{field}', '')}}"

                content_escaped = re.sub(r"\{(\w+)\}", replace_placeholder, content_escaped)
                lines.append(f'{indent}msg{i}_content = f"""')
                lines.append(content_escaped)
                lines.append(f'{indent}"""')
            else:
                # Use regular string
                lines.append(f'{indent}msg{i}_content = """')
                lines.append(content_escaped)
                lines.append(f'{indent}"""')

        # Build return statement
        lines.append(f"{indent}return [")
        for i, msg in enumerate(messages):
            role = msg["role"]
            lines.append(f'{indent}    Message(role="{role}", content=msg{i}_content),')
        lines.append(f"{indent}]")

        return "\n".join(lines)

    def _update_tool_descriptions(self, config_source: str, tools: list[dict]) -> str:
        """Update tool and parameter descriptions in config source."""
        import re

        for tool in tools:
            tool_name = tool["name"]
            tool_desc = tool["description"]

            # Update tool description
            pattern = rf'(Tool\(\s*name\s*=\s*["\']){tool_name}(["\']\\s*,\s*description\s*=\s*["\']{{3}}).*?(["\']{{3}})'
            replacement = rf"\1{tool_name}\2{tool_desc}\3"
            config_source = re.sub(pattern, replacement, config_source, flags=re.DOTALL)

            # Update parameter descriptions
            for param in tool.get("parameters", []):
                param_name = param["name"]
                param_desc = param["description"]

                # Find and replace parameter description
                param_pattern = rf'(["\']){param_name}\1\s*:\s*ToolParam\(\s*type\s*=\s*["\'](\w+)["\']\\s*,\s*description\s*=\s*["\']([^"\']*)["\']'
                param_replacement = (
                    rf'\1{param_name}\1: ToolParam(type="\2", description="{param_desc}"'
                )
                config_source = re.sub(param_pattern, param_replacement, config_source)

        return config_source

    def _build_new_config(self, data: dict[str, Any]) -> str:
        """Build a new config from scratch."""
        # Extract config params
        model_name = data.get("model", "gpt-4-turbo")
        system_prompt = data.get("systemPrompt", "You are an expert assistant.")
        _max_turns = data.get("maxTurns", 5)  # TODO: Use this in eval config
        num_samples = data.get("numSamples", 10)
        temperature = data.get("temperature", 0.1)
        stream_tokens = data.get("stream_tokens", True)  # Default to True for dev loop

        # Extract environment-specific fields
        ssh_target = data.get("ssh_target", "")
        cuda_device_ids = data.get("cuda_device_ids", [0])
        dataset_path = data.get("dataset_path", "data/default.json")

        # Infer provider and API key env var from model name
        if "claude" in model_name.lower() or "anthropic" in model_name.lower():
            provider = "anthropic"
            api_key_env_var = "ANTHROPIC_API_KEY"
            api_base = "https://api.anthropic.com"  # SDK adds /v1 automatically
        elif (
            "gpt" in model_name.lower() or "o1" in model_name.lower() or "o3" in model_name.lower()
        ):
            provider = "openai"
            api_key_env_var = "OPENAI_API_KEY"
            api_base = "https://api.openai.com/v1"
        else:
            # Default to OpenAI
            provider = "openai"
            api_key_env_var = "OPENAI_API_KEY"
            api_base = "https://api.openai.com/v1"

        # Build config file
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        config = f'''"""Agent configuration - Generated by dev loop tool

Generated: {timestamp}
Model: {model_name}
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

from ..dtypes import Message, Tool
from ..config import BaseModelConfig, BaseEvaluationConfig


@dataclass
class CustomEnvironment:
    """Custom agent environment."""

    # Environment identification
    env_name: str = "custom-environment"

    # Infrastructure settings (passed from environment_config)
    ssh_target: str = ""
    cuda_device_ids: List[int] = field(default_factory=lambda: [0])
    dataset_path: Path = field(default_factory=lambda: Path("data/default.json"))

    def get_tools(self) -> List[Tool]:
        """Return tools available to agent."""
        # TODO: Add custom tools here
        return []

    def prepare_messages(self, sample_data: Dict[str, Any]) -> List[Message]:
        """Prepare initial messages for task.

        Args:
            sample_data: Sample from dataset

        Returns:
            List of messages to initialize conversation
        """
        system_prompt = """{system_prompt}"""

        # Try common dataset field names
        user_prompt = (
            sample_data.get("problem_description") or
            sample_data.get("prompt") or
            sample_data.get("question") or
            sample_data.get("input") or
            str(sample_data)
        )

        return [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

    async def on_assistant_message(self, message: Message, state):
        """Handle assistant messages.

        This is where you process agent outputs, extract actions, etc.
        """
        # TODO: Implement environment reaction to agent
        return state


@dataclass(frozen=True)
class Config:
    """Main configuration."""

    # Model configuration
    model: BaseModelConfig = field(
        default_factory=lambda: BaseModelConfig(
            model_name="{model_name}",
            provider="{provider}",
            api_base="{api_base}",
            api_key_env_var="{api_key_env_var}",
            temperature={temperature},
            max_tokens=16384,
        )
    )

    # Environment configuration
    environment_class: type = CustomEnvironment
    environment_config: dict = field(default_factory=lambda: {{
        'ssh_target': '{ssh_target}',
        'cuda_device_ids': {cuda_device_ids},
        'dataset_path': Path('{dataset_path}'),
    }})

    # Evaluation settings
    evaluation: BaseEvaluationConfig = field(
        default_factory=lambda: BaseEvaluationConfig(
            environment=None,  # Will be set by factory
            eval_name="custom_eval",
            num_samples={num_samples},
            output_dir=Path("results/custom"),
            verbose=True,
            show_progress=True,
            stream_tokens={stream_tokens},  # Enable/disable token streaming to stdout
        )
    )

    experiment_name: str = "custom_experiment"

    async def create_environment(self, sample_data: dict):
        """Create environment instance for a sample."""
        return self.environment_class(**self.environment_config)

    # Backward compatibility methods for entrypoint
    def to_endpoint(self):
        """Convert to rollouts Endpoint (delegates to model config)."""
        return self.model.to_endpoint()

    def to_eval_config(self, score_fn):
        """Convert to rollouts EvalConfig (delegates to evaluation config)."""
        return self.evaluation.to_eval_config(score_fn)

    @property
    def dataset_path(self):
        """Access dataset path from environment config."""
        return self.environment_config.get('dataset_path')

    @property
    def ssh_target(self) -> str:
        return self.environment_config.get('ssh_target')

    @property
    def gpu_id(self) -> int:
        # Return first GPU from cuda_device_ids list, or fallback to gpu_id
        cuda_device_ids = self.environment_config.get('cuda_device_ids')
        if cuda_device_ids and isinstance(cuda_device_ids, list) and len(cuda_device_ids) > 0:
            return cuda_device_ids[0]
        return self.environment_config.get('gpu_id', 0)

    @property
    def max_turns(self) -> int:
        return self.evaluation.max_turns

    @property
    def num_samples(self) -> int:
        return self.evaluation.num_samples

    @property
    def model_name(self) -> str:
        return self.model.model_name


# Export config instance
config = Config()


# Export prepare_messages for backward compatibility
def prepare_messages(sample_data: Dict[str, Any]) -> List[Message]:
    """Prepare initial messages from dataset sample.

    Note: You can use f-string placeholders to reference dataset fields.
    For example: "Solve this problem: {{problem_description}}"
    """
    system_prompt = """{system_prompt}"""

    # Use f-string with dataset fields (customize based on your dataset)
    user_prompt = f"""{{sample_data.get("problem_description") or sample_data.get("prompt") or sample_data.get("question") or sample_data.get("input") or str(sample_data)}}"""

    return [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ]
'''

        return config

    def _launch_config(self) -> None:
        """Launch a config in the background with live streaming support."""
        global _run_counter, _active_runs

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
            config_name = data.get("configName")

            logger.debug(f"🚀 Launch request received for config: {config_name}")

            if not config_name:
                logger.error("Launch failed: Missing configName")
                self.send_error(400, "Missing configName")
                return

            config_path = self.project_root / "configs" / f"{config_name}.py"

            if not config_path.exists():
                logger.error(f"Launch failed: Config not found: {config_name}")
                self.send_error(404, f"Config not found: {config_name}")
                return

            logger.debug(f"Config file found: {config_path}")

            # GPU preflight check (if config has cuda_device_ids)
            # Read config to check for GPU requirements
            config_source = config_path.read_text()
            import re

            cuda_device_ids = []
            # Try to match cuda_device_ids as a list first
            gpu_list_match = re.search(
                r'["\']cuda_device_ids["\']\s*:\s*\[([^\]]+)\]', config_source
            )
            if not gpu_list_match:
                gpu_list_match = re.search(r"cuda_device_ids\s*[:=]\s*\[([^\]]+)\]", config_source)

            if gpu_list_match:
                # Parse list of GPU IDs
                cuda_device_ids_str = gpu_list_match.group(1)
                cuda_device_ids = [
                    int(x.strip()) for x in cuda_device_ids_str.split(",") if x.strip().isdigit()
                ]
            else:
                # Fallback to single gpu_id
                gpu_match = re.search(r'["\']gpu_id["\']\s*:\s*(\d+)', config_source)
                if not gpu_match:
                    gpu_match = re.search(r"gpu_id\s*[:=]\s*(\d+)", config_source)
                if gpu_match:
                    cuda_device_ids = [int(gpu_match.group(1))]

            # Check if GPUs are available (simple check using nvidia-smi)
            if cuda_device_ids:
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )

                    if result.returncode == 0:
                        # Parse nvidia-smi output
                        gpu_stats = {}
                        for line in result.stdout.strip().splitlines():
                            parts = [p.strip() for p in line.split(",")]
                            if len(parts) == 3:
                                try:
                                    gpu_id = int(parts[0])
                                    memory_mb = int(parts[1])
                                    util_pct = int(parts[2])
                                    gpu_stats[gpu_id] = {
                                        "memory_mb": memory_mb,
                                        "util_pct": util_pct,
                                    }
                                except ValueError:
                                    continue

                        # Check if requested GPUs are free (>1GB used or >5% util = busy)
                        for gpu_id in cuda_device_ids:
                            if gpu_id in gpu_stats:
                                stats = gpu_stats[gpu_id]
                                if stats["memory_mb"] > 1000 or stats["util_pct"] > 5:
                                    self._json_response({
                                        "success": False,
                                        "error": f"GPU {gpu_id} is busy ({stats['memory_mb']}MB used, {stats['util_pct']}% util)",
                                        "preflight_failed": True,
                                    })
                                    return
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    # nvidia-smi not available or timed out, proceed anyway
                    pass

            # Acquire semaphore (blocks if max concurrent runs reached)
            if not _run_semaphore.acquire(blocking=False):
                self._json_response({
                    "success": False,
                    "error": f"Maximum concurrent runs ({_max_concurrent_runs}) reached. Please wait for a run to complete.",
                    "queue_full": True,
                })
                return

            # Generate unique run ID
            with _run_lock:
                _run_counter += 1
                run_id = f"run_{_run_counter}_{int(time.time())}"

            logger.debug(f"Generated run_id: {run_id}")

            # Build command to launch config
            command = ["python", "entrypoint.py", str(config_path)]
            logger.debug(f"Command: {' '.join(command)}")

            # Launch with tracked pipes
            try:
                # Set PYTHONUNBUFFERED to disable Python's stdout buffering for token streaming
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"

                process = subprocess.Popen(
                    command,
                    cwd=self.project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Merge stderr into stdout
                    text=True,
                    bufsize=0,  # Unbuffered - changed from 1 (line buffered) for token streaming
                    start_new_session=True,  # Detach from parent
                    env=env,
                )
                logger.debug(f"Process started with PID: {process.pid}")
            except Exception as e:
                logger.exception(f"Failed to start process: {str(e)}")
                _run_semaphore.release()  # Release semaphore on failure
                self._json_response({
                    "success": False,
                    "error": f"Failed to start process: {str(e)}",
                })
                return

            # Store in registry
            with _run_lock:
                _active_runs[run_id] = {
                    "process": process,
                    "config_name": config_name,
                    "start_time": time.time(),
                    "status": "running",
                    "output_lines": [],
                    "exit_code": None,
                    "cuda_device_ids": cuda_device_ids,
                }

            logger.debug(f"Run registered in _active_runs. Total active: {len(_active_runs)}")

            self._json_response({
                "success": True,
                "run_id": run_id,
                "command": " ".join(command),
                "config_name": config_name,
            })

        except json.JSONDecodeError as e:
            self.send_error(400, f"Invalid JSON: {e}")
        except Exception as e:
            self.send_error(500, f"Launch failed: {e}")

    def _stream_run_output(self, run_id: str) -> None:
        """Stream output from events.jsonl and stdout via SSE.

        Dual-stream approach:
        1. Parse stdout to find result_dir path
        2. Once found, tail events.jsonl for structured events (sample/turn/token)
        3. Continue streaming stdout for logs (for debugging/CLI compatibility)

        TODO: Consider semantic compression of event stream to reduce bandwidth.
        See ~/research/docs/code_style/ryolu_design_frontend.md for compression strategies.
        Current approach sends every token/turn event separately - could batch or delta-encode.
        """
        global _active_runs

        logger.debug(f"📡 Stream connection opened for run_id: {run_id}")

        if run_id not in _active_runs:
            logger.error(f"Stream failed: Run not found: {run_id}")
            logger.debug(f"Available run_ids: {list(_active_runs.keys())}")
            self.send_error(404, f"Run not found: {run_id}")
            return

        run_data = _active_runs[run_id]
        process = run_data["process"]
        logger.debug(f"Streaming from PID: {process.pid}, status: {run_data['status']}")

        # Set up SSE headers
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")  # Disable nginx buffering
        self.end_headers()

        try:
            import re
            from pathlib import Path

            logger.debug(f"🔍 Starting dual-stream (stdout + events.jsonl) for {run_id}")

            # Track events.jsonl state
            events_file = None
            events_file_handle = None
            result_dir_found = False

            # Buffer for stdout line-by-line reading
            stdout_buffer = ""

            while True:
                # Read from stdout (line-buffered for parsing)
                char = process.stdout.read(1)
                if not char:
                    # Process ended
                    logger.debug(f"🔍 Stream ended (no more data) for {run_id}")
                    break

                stdout_buffer += char

                # When we hit newline, parse the line and check for events
                if char == "\n":
                    line = stdout_buffer.rstrip()

                    # Parse stdout to find result_dir
                    if not result_dir_found:
                        match = re.search(r"📂 Results directory: (.+)", line)
                        if match:
                            result_dir = Path(match.group(1))
                            events_file = result_dir / "events.jsonl"
                            result_dir_found = True
                            logger.debug(f"📂 Found result_dir: {result_dir}")
                            logger.debug(f"📄 Will tail events from: {events_file}")

                    # Send stdout line as log event (for debugging)
                    with _run_lock:
                        run_data["output_lines"].append(line)

                    event_data = json.dumps({"line": line, "type": "stdout"})
                    self.wfile.write(f"data: {event_data}\n\n".encode())
                    self.wfile.flush()

                    stdout_buffer = ""  # Clear buffer

                # If we found events file, tail it for structured events
                if result_dir_found and events_file and events_file.exists():
                    if events_file_handle is None:
                        # Open events file for reading
                        events_file_handle = open(events_file)
                        events_file_handle.seek(0, 2)  # Seek to end
                        logger.debug("📄 Opened events.jsonl for tailing")

                    # Read new lines from events.jsonl
                    event_line = events_file_handle.readline()
                    while event_line:
                        # Tiger Style: events.jsonl is written by our own code - malformed JSON is a programmer error
                        # Fail-fast instead of silently logging. Assertions catch bugs during development.
                        line = event_line.strip()
                        if not line:  # Skip empty lines
                            event_line = events_file_handle.readline()
                            continue

                        # Parse JSONL event - crash loudly if malformed (programmer error, not operational error)
                        event_obj = json.loads(line)
                        assert isinstance(event_obj, dict), (
                            f"Event must be dict, got {type(event_obj)}: {event_obj}"
                        )
                        event_type = event_obj.get("message")
                        assert event_type is not None, f"Event missing 'message' field: {event_obj}"
                        assert "timestamp" in event_obj, (
                            f"Event missing 'timestamp' field: {event_obj}"
                        )

                        # Forward event as-is
                        self.wfile.write(f"data: {json.dumps(event_obj)}\n\n".encode())
                        self.wfile.flush()
                        logger.debug(f"📤 Forwarded event: {event_type}")

                        event_line = events_file_handle.readline()

            # Send any remaining buffered stdout
            if stdout_buffer:
                event_data = json.dumps({"line": stdout_buffer, "type": "stdout"})
                self.wfile.write(f"data: {event_data}\n\n".encode())
                self.wfile.flush()

            # Close events file if opened
            if events_file_handle:
                events_file_handle.close()

            # Wait for process to complete
            exit_code = process.wait()

            # Release semaphore
            _run_semaphore.release()

            # Send completion event
            completion_data = json.dumps({
                "type": "complete",
                "exit_code": exit_code,
                "status": "success" if exit_code == 0 else "failed",
            })
            self.wfile.write(f"data: {completion_data}\n\n".encode())
            self.wfile.flush()

            # Update registry
            with _run_lock:
                run_data["status"] = "completed" if exit_code == 0 else "failed"
                run_data["exit_code"] = exit_code

        except Exception as e:
            # Release semaphore on error
            _run_semaphore.release()

            error_data = json.dumps({"type": "error", "message": str(e)})
            self.wfile.write(f"data: {error_data}\n\n".encode())
            self.wfile.flush()

            # Update status
            with _run_lock:
                run_data["status"] = "failed"

    def _watch_run(self, run_id: str) -> None:
        """Stream events.jsonl from a results dir not launched by this server.

        Discovers the results dir by scanning known_results_dirs for a subdir
        named run_id that contains events.jsonl.  Tails the file, normalises
        field names to match the StreamEvent schema the frontend expects, and
        emits each record as an SSE event.  Stops when eval_end is seen or the
        file stops growing (eval finished while we were connecting).

        Field normalisation (events.jsonl → StreamEvent):
          message      → type
          sample_id    → id
          sample_name  → name   (sample_start only)
          eval_name    → name   (eval_start / eval_end only)
        """
        from pathlib import Path

        # Find the results dir across all known dirs
        events_file: Path | None = None
        for search_dir in [self.__class__.results_dir] + list(self.__class__.known_results_dirs):
            candidate = search_dir / run_id / "events.jsonl"
            if candidate.exists():
                events_file = candidate
                break

        if events_file is None:
            self.send_error(404, f"No events.jsonl found for run: {run_id}")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def _normalise(raw: dict) -> dict:
            """Map events.jsonl fields to StreamEvent shape."""
            event_type = raw.get("message", "")
            out: dict = {"type": event_type, "timestamp": raw.get("timestamp", "")}

            if event_type == "eval_start":
                out["name"] = raw.get("eval_name", "")
                out["total"] = raw.get("total", 0)
            elif event_type == "sample_start":
                out["id"] = raw.get("sample_id", "")
                out["name"] = raw.get("sample_name", "")
            elif event_type == "turn":
                out["id"] = raw.get("sample_id", "")
                out["turn"] = raw.get("turn", 0)
                out["status"] = raw.get("status", "")
            elif event_type == "sample_end":
                out["id"] = raw.get("sample_id", "")
                out["score"] = raw.get("score", 0)
            elif event_type == "eval_end":
                out["name"] = raw.get("eval_name", "")
                out["total"] = raw.get("total", 0)
            else:
                # Forward unknown events with all fields, type already set
                out.update({k: v for k, v in raw.items() if k not in ("message", "timestamp")})

            return out

        try:
            import time

            poll_interval = 0.25  # seconds between reads when no new data
            max_idle = 300  # stop after 5 min of no new data post eval_end
            idle_since: float | None = None
            done = False

            with open(events_file) as fh:
                # Replay existing content from the beginning so the UI gets
                # the full picture even when attaching mid-run.
                while True:
                    line = fh.readline()
                    if not line:
                        # Caught up — switch to tail mode
                        break
                    line = line.strip()
                    if not line:
                        continue
                    raw = json.loads(line)
                    normalised = _normalise(raw)
                    self.wfile.write(f"data: {json.dumps(normalised)}\n\n".encode())
                    self.wfile.flush()
                    if raw.get("message") == "eval_end":
                        done = True
                        break

                # Tail mode: poll for new lines
                while not done:
                    line = fh.readline()
                    if line:
                        line = line.strip()
                        if line:
                            raw = json.loads(line)
                            normalised = _normalise(raw)
                            self.wfile.write(f"data: {json.dumps(normalised)}\n\n".encode())
                            self.wfile.flush()
                            if raw.get("message") == "eval_end":
                                done = True
                        idle_since = None
                    else:
                        # No new data
                        if idle_since is None:
                            idle_since = time.monotonic()
                        elif time.monotonic() - idle_since > max_idle:
                            break
                        time.sleep(poll_interval)

            # Emit complete so frontend transitions out of running state
            self.wfile.write(
                f"data: {json.dumps({'type': 'complete', 'exit_code': 0, 'status': 'success'})}\n\n".encode()
            )
            self.wfile.flush()

        except Exception as e:
            logger.exception(f"Error watching run {run_id}: {e}")

    def _log_from_frontend(self) -> None:
        """Receive log messages from frontend."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
            level = data.get("level", "info").upper()
            message = data.get("message", "")
            context = data.get("context", {})

            # Log with appropriate level
            log_msg = f"[FRONTEND] {message}"
            if context:
                log_msg += f" | {json.dumps(context)}"

            if level == "ERROR":
                logger.error(log_msg)
            elif level == "WARN":
                logger.warning(log_msg)
            else:
                logger.info(log_msg)

            self._json_response({"success": True})

        except Exception as e:
            logger.exception(f"Failed to process frontend log: {e}")
            self.send_error(400, str(e))

    def _discover_watching_runs(self) -> list[dict]:
        """Scan results dirs for runs with events.jsonl but no report.json.

        These are externally-launched evals that are either in progress or
        finished without writing a report.  Excludes run_ids already tracked
        in _active_runs (launched by this server).
        """
        watching = []
        seen_ids: set[str] = set(_active_runs.keys())
        all_dirs = [self.__class__.results_dir] + list(self.__class__.known_results_dirs)

        for results_dir in all_dirs:
            if not results_dir.exists():
                continue
            for subdir in results_dir.iterdir():
                if not subdir.is_dir():
                    continue
                run_id = subdir.name
                if run_id in seen_ids:
                    continue
                events_file = subdir / "events.jsonl"
                report_file = subdir / "report.json"
                if not events_file.exists():
                    continue
                # Only surface runs that are still in progress:
                # - no report.json (not finished successfully)
                # - last event is not eval_end (primary signal)
                # - events.jsonl modified within the last 30 minutes (fallback: stale
                #   crashed run that never wrote eval_end)
                if report_file.exists():
                    continue
                # Check last line for eval_end first — this is the primary signal.
                # Must come before the mtime check because repair_stale_runs.py appends
                # eval_end, which updates mtime to now and would otherwise pass the age
                # filter and re-appear as live.
                try:
                    with open(events_file, "rb") as _f:
                        _f.seek(0, 2)
                        _size = _f.tell()
                        _f.seek(max(0, _size - 4096))
                        _tail = _f.read().decode("utf-8", errors="replace")
                    _last_line = next((l for l in reversed(_tail.splitlines()) if l.strip()), "")
                    if '"eval_end"' in _last_line or '"message": "eval_end"' in _last_line:
                        continue
                except OSError:
                    pass
                # Fallback: exclude runs stale for >30 min that never wrote eval_end
                age_seconds = time.time() - events_file.stat().st_mtime
                if age_seconds > 1800:
                    continue
                seen_ids.add(run_id)
                watching.append({
                    "run_id": run_id,
                    "config_name": run_id,
                    "start_time": events_file.stat().st_mtime,
                    "status": "watching",
                    "exit_code": None,
                    "output_length": 0,
                })

        return watching

    def _list_active_runs(self) -> None:
        """List all active and recent runs, including externally-launched ones."""
        global _active_runs

        runs = []
        with _run_lock:
            for run_id, data in _active_runs.items():
                runs.append({
                    "run_id": run_id,
                    "config_name": data["config_name"],
                    "start_time": data["start_time"],
                    "status": data["status"],
                    "exit_code": data.get("exit_code"),
                    "output_length": len(data.get("output_lines", [])),
                })

        runs.extend(self._discover_watching_runs())
        self._json_response({"runs": runs})

    def _list_results_dirs(self) -> None:
        """Return all known results directories and the currently active one."""
        dirs = []
        for d in self.__class__.known_results_dirs:
            label = f"{d.parent.name}/{d.name}" if d.name == "results" else d.name
            dirs.append({"path": str(d), "label": label, "exists": d.exists()})
        self._json_response({
            "current": str(self.__class__.results_dir),
            "dirs": dirs,
        })

    def _set_results_dir(self) -> None:
        """Hot-swap the results directory. Body: {"path": "/abs/path/to/results"}"""
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        new_path = Path(body["path"]).expanduser().resolve()
        if not new_path.exists():
            self.send_error(400, f"Directory does not exist: {new_path}")
            return
        self.__class__.results_dir = new_path
        logger.info(f"Results dir switched to: {new_path}")
        self._json_response({"ok": True, "path": str(new_path)})

    def _kill_run(self, run_id: str) -> None:
        """Kill a running process."""
        global _active_runs

        logger.debug(f"🛑 Kill request received for run_id: {run_id}")

        if run_id not in _active_runs:
            logger.error(f"Kill failed: Run not found: {run_id}")
            logger.debug(f"Available run_ids: {list(_active_runs.keys())}")
            self.send_error(404, f"Run not found: {run_id}")
            return

        run_data = _active_runs[run_id]
        process = run_data["process"]

        logger.debug(f"Run status: {run_data['status']}, PID: {process.pid}")

        if run_data["status"] != "running":
            logger.warning(f"Cannot kill: Run is not running (status: {run_data['status']})")
            self._json_response({
                "success": False,
                "message": f"Run is not running (status: {run_data['status']})",
            })
            return

        try:
            # Kill entire process group (includes child processes)
            if process.poll() is None:  # Process is still running
                pgid = os.getpgid(process.pid)
                logger.debug(f"Killing process group {pgid} (SIGTERM)")
                os.killpg(pgid, signal.SIGTERM)

                # Wait a bit, then force kill if still alive
                time.sleep(0.5)
                if process.poll() is None:
                    logger.debug("Process still alive, force killing (SIGKILL)")
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    logger.debug("Process terminated successfully")
            else:
                logger.debug(f"Process already terminated (exit code: {process.poll()})")

            # Release semaphore
            _run_semaphore.release()
            logger.debug("Released semaphore")

            # Update status
            with _run_lock:
                run_data["status"] = "killed"
                run_data["exit_code"] = -1

            logger.info(f"successfully killed run {run_id}")
            self._json_response({"success": True, "message": f"Killed run {run_id}"})

        except Exception as e:
            logger.error(f"❌ Failed to kill run: {str(e)}", exc_info=True)
            self._json_response({"success": False, "message": f"Failed to kill run: {str(e)}"})

    def _delete_run(self, run_id: str) -> None:
        """Delete a completed/failed run from registry."""
        global _active_runs

        if run_id not in _active_runs:
            self.send_error(404, f"Run not found: {run_id}")
            return

        run_data = _active_runs[run_id]

        # Don't delete running processes
        if run_data["status"] == "running":
            self._json_response({
                "success": False,
                "message": "Cannot delete running process. Kill it first.",
            })
            return

        # Remove from registry
        with _run_lock:
            del _active_runs[run_id]

        self._json_response({"success": True, "message": f"Deleted run {run_id}"})

    def _json_response(self, data: Any) -> None:
        """Send JSON response."""
        json_data = json.dumps(data, indent=2)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(json_data)))
        self.end_headers()
        self.wfile.write(json_data.encode("utf-8"))


def main() -> None:
    """Run the dev loop server."""
    parser = argparse.ArgumentParser(
        description="Agent dev loop tool - config builder & trace viewer"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port to run server on (default: 8080)"
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't automatically open browser"
    )
    parser.add_argument(
        "--results-dirs",
        nargs="*",
        type=Path,
        default=[],
        help="Additional results directories to offer in the UI (space-separated paths)",
    )

    args = parser.parse_args()

    # Set project root on server class
    DevLoopServer.project_root = args.project.resolve()

    # Build known results dirs: project's own results/ first, then any extras
    primary = args.project.resolve() / "results"
    extras = [Path(p).expanduser().resolve() for p in (args.results_dirs or [])]
    all_dirs: list[Path] = []
    seen: set[Path] = set()
    for d in [primary] + extras:
        if d not in seen:
            all_dirs.append(d)
            seen.add(d)
    DevLoopServer.results_dir = all_dirs[0] if all_dirs else primary
    DevLoopServer.known_results_dirs = all_dirs

    # Create server
    server = HTTPServer(("localhost", args.port), DevLoopServer)

    url = f"http://localhost:{args.port}"
    print(f"\n{'=' * 60}")
    print("🚀 Agent Dev Loop Tool")
    print(f"{'=' * 60}")
    print(f"URL: {url}")
    print(f"Project: {DevLoopServer.project_root}")
    print(f"{'=' * 60}\n")

    # Open browser
    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped")


if __name__ == "__main__":
    main()
