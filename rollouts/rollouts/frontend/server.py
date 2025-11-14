#!/usr/bin/env python3
"""HTTP server for agent dev loop tool.

Provides:
- Static file serving (index.html)
- Config generation API
- Trace viewing API

Usage:
    python -m rollouts.frontend.server
    python -m rollouts.frontend.server --port 8080
    python -m rollouts.frontend.server --project ~/wafer_stuff/kernels-gpumode-agent
"""
import argparse
import json
import webbrowser
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


class DevLoopServer(SimpleHTTPRequestHandler):
    """HTTP server for agent dev loop tool.

    Serves static files and provides API endpoints for:
    - /api/configs - List available configs
    - /api/traces - List/load evaluation traces
    - /api/generate - Generate new config files
    """

    # Class variable to store project root (set by main())
    project_root: Path = Path.cwd()

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._serve_index()
        elif path == "/api/configs":
            self._list_configs()
        elif path == "/api/traces":
            self._list_traces()
        elif path.startswith("/api/trace/"):
            # Extract trace ID from path like /api/trace/02_agent_multiturn_20231114_143022
            trace_id = path.split("/api/trace/")[1]
            self._get_trace(trace_id)
        elif path.startswith("/api/load-config/"):
            # Extract config name from path like /api/load-config/02_agent_multiturn
            config_name = path.split("/api/load-config/")[1]
            self._load_config(config_name)
        else:
            # Default behavior for other files
            super().do_GET()

    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/generate":
            self._generate_config()
        else:
            self.send_error(404, "Not found")

    def _serve_index(self):
        """Serve the main HTML file."""
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

    def _list_configs(self):
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

    def _list_traces(self):
        """List available evaluation traces in results/."""
        results_dir = self.project_root / "results"

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

    def _get_trace(self, trace_id: str):
        """Load a specific evaluation trace."""
        trace_dir = self.project_root / "results" / trace_id

        if not trace_dir.exists():
            self.send_error(404, f"Trace not found: {trace_id}")
            return

        report_path = trace_dir / "report.json"
        if not report_path.exists():
            self.send_error(404, f"No report.json in trace: {trace_id}")
            return

        # Load report
        report = json.loads(report_path.read_text())

        # Load trajectories (if they exist)
        trajectories_dir = trace_dir / "trajectories"
        trajectories = []

        if trajectories_dir.exists():
            for traj_file in sorted(trajectories_dir.glob("*.json")):
                traj_data = json.loads(traj_file.read_text())
                trajectories.append(traj_data)

        trace_data = {
            "id": trace_id,
            "report": report,
            "trajectories": trajectories,
        }

        self._json_response(trace_data)

    def _load_config(self, config_name: str):
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
        temp_match = re.search(r'temperature\s*[:=]\s*([0-9.]+)', config_source)
        if temp_match:
            config_data["temperature"] = float(temp_match.group(1))

        # Extract prepare_messages method (full function)
        prepare_msg_match = re.search(
            r'(def prepare_messages\(self.*?^    def \w+|def prepare_messages\(self.*?^class \w+|def prepare_messages\(self.*?$)',
            config_source,
            re.DOTALL | re.MULTILINE
        )
        if prepare_msg_match:
            # Clean up and dedent the function
            func_text = prepare_msg_match.group(1)
            # Remove trailing class/def if captured
            func_text = re.sub(r'\n    (def |class )\w+.*$', '', func_text, flags=re.DOTALL)
            config_data["prepareMessages"] = func_text.strip()

        # Also extract just system_prompt for backward compatibility
        prompt_match = re.search(r'system_prompt\s*=\s*"""([^"]+)"""', config_source, re.DOTALL)
        if prompt_match:
            config_data["systemPrompt"] = prompt_match.group(1).strip()

        # Extract max turns
        turns_match = re.search(r'max_turns\s*[:=]\s*(\d+)', config_source)
        if turns_match:
            config_data["maxTurns"] = int(turns_match.group(1))

        # Extract num samples
        samples_match = re.search(r'num_samples\s*[:=]\s*(\d+)', config_source)
        if samples_match:
            config_data["numSamples"] = int(samples_match.group(1))

        # Extract seed
        seed_match = re.search(r'seed\s*[:=]\s*(\d+)', config_source)
        if seed_match:
            config_data["seed"] = int(seed_match.group(1))

        # Extract start_idx and end_idx
        start_match = re.search(r'start_idx\s*[:=]\s*(\d+)', config_source)
        if start_match:
            config_data["startIdx"] = int(start_match.group(1))

        end_match = re.search(r'end_idx\s*[:=]\s*(\d+)', config_source)
        if end_match:
            config_data["endIdx"] = int(end_match.group(1))

        # Extract environment-specific fields from both direct assignment and environment_config dict
        ssh_match = re.search(r'["\']ssh_target["\']\s*:\s*["\']([^"\']+)["\']', config_source)
        if not ssh_match:
            ssh_match = re.search(r'ssh_target\s*[:=]\s*["\']([^"\']+)["\']', config_source)
        if ssh_match:
            config_data["sshTarget"] = ssh_match.group(1)

        gpu_match = re.search(r'["\']gpu_id["\']\s*:\s*(\d+)', config_source)
        if not gpu_match:
            gpu_match = re.search(r'gpu_id\s*[:=]\s*(\d+)', config_source)
        if gpu_match:
            config_data["gpuId"] = int(gpu_match.group(1))

        dataset_match = re.search(r'["\']dataset_path["\']\s*:\s*Path\(["\']([^"\']+)["\']\)', config_source)
        if not dataset_match:
            dataset_match = re.search(r'dataset_path\s*[:=]\s*Path\(["\']([^"\']+)["\']\)', config_source)
        if dataset_match:
            config_data["datasetPath"] = dataset_match.group(1)

        env_name_match = re.search(r'env_name\s*[:=]\s*["\']([^"\']+)["\']', config_source)
        if env_name_match:
            config_data["envName"] = env_name_match.group(1)

        # Parse tools from get_tools() method
        tools_section = re.search(r'def get_tools\(self\).*?return \[(.*?)\]', config_source, re.DOTALL)
        if tools_section:
            # This is a simplified parser - just check if it returns empty list or has tools
            tools_content = tools_section.group(1).strip()
            config_data["hasTools"] = len(tools_content) > 0 and "Tool(" in tools_content
        else:
            config_data["hasTools"] = False

        self._json_response(config_data)

    def _generate_config(self):
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

        # Return as text
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(config_text)))
        self.end_headers()
        self.wfile.write(config_text.encode("utf-8"))

    def _build_config_file(self, data: dict[str, Any]) -> str:
        """Build config file content from data.

        Args:
            data: Config parameters from frontend

        Returns:
            Python source code for config file
        """
        # Check if we're building from a base config
        base_name = data.get("baseName")

        if base_name:
            # Load base config and modify it
            return self._build_from_base_config(data, base_name)
        else:
            # Build from scratch
            return self._build_new_config(data)

    def _build_from_base_config(self, data: dict[str, Any], base_name: str) -> str:
        """Build config by copying and modifying base config."""
        base_path = self.project_root / "configs" / f"{base_name}.py"

        if not base_path.exists():
            # Fallback to new config
            return self._build_new_config(data)

        # Read base config
        config_source = base_path.read_text()

        # Replace specific values
        import re

        # Update model if changed
        if "model" in data:
            config_source = re.sub(
                r'(model_name\s*[:=]\s*)["\']([^"\']+)["\']',
                f'\\1"{data["model"]}"',
                config_source
            )

        # Update temperature if changed
        if "temperature" in data:
            config_source = re.sub(
                r'(temperature\s*[:=]\s*)([0-9.]+)',
                f'\\g<1>{data["temperature"]}',
                config_source
            )

        # Update system prompt if changed
        if "systemPrompt" in data:
            prompt = data["systemPrompt"]
            config_source = re.sub(
                r'(system_prompt\s*=\s*""")([^"]+)(""")',
                f'\\1{prompt}\\3',
                config_source,
                flags=re.DOTALL
            )

        # Update max turns if changed
        if "maxTurns" in data:
            config_source = re.sub(
                r'(max_turns\s*[:=]\s*)(\d+)',
                f'\\g<1>{data["maxTurns"]}',
                config_source
            )

        # Update num samples if changed
        if "numSamples" in data:
            config_source = re.sub(
                r'(num_samples\s*[:=]\s*)(\d+)',
                f'\\g<1>{data["numSamples"]}',
                config_source
            )

        # Update environment fields if present
        env_fields = data.get("envFields", {})

        if "sshTarget" in env_fields:
            config_source = re.sub(
                r'(ssh_target\s*[:=]\s*)["\']([^"\']+)["\']',
                f'\\1"{env_fields["sshTarget"]}"',
                config_source
            )

        if "gpuId" in env_fields:
            config_source = re.sub(
                r'(gpu_id\s*[:=]\s*)(\d+)',
                f'\\g<1>{env_fields["gpuId"]}',
                config_source
            )

        if "datasetPath" in env_fields:
            config_source = re.sub(
                r'(dataset_path\s*[:=]\s*Path\()["\']([^"\']+)(["\'])',
                f'\\1"{env_fields["datasetPath"]}"\\3',
                config_source
            )

        if "envName" in env_fields:
            config_source = re.sub(
                r'(env_name\s*[:=]\s*)["\']([^"\']+)["\']',
                f'\\1"{env_fields["envName"]}"',
                config_source
            )

        # Add generation comment at top
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        config_name = data.get("configName", "custom_config")
        header = f'"""Agent configuration - {config_name}\n\nGenerated: {timestamp}\nBased on: {base_name}\n"""\n'

        # Replace existing docstring or prepend
        config_source = re.sub(r'^"""[^"]*"""\n', header, config_source)

        return config_source

    def _build_new_config(self, data: dict[str, Any]) -> str:
        """Build a new config from scratch."""
        # Extract config params
        model_name = data.get("model", "gpt-4-turbo")
        system_prompt = data.get("systemPrompt", "You are an expert assistant.")
        max_turns = data.get("maxTurns", 5)
        num_samples = data.get("numSamples", 10)
        temperature = data.get("temperature", 0.1)

        # Build config file
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        config = f'''"""Agent configuration - Generated by dev loop tool

Generated: {timestamp}
Model: {model_name}
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

from rollouts.dtypes import Message, Tool
from rollouts.config import BaseModelConfig, BaseEvaluationConfig


@dataclass
class CustomEnvironment:
    """Custom agent environment."""

    env_name: str = "custom-environment"

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

        user_prompt = sample_data.get("prompt", "")

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
            temperature={temperature},
            max_tokens=16384,
        )
    )

    # Environment configuration
    environment_class: type = CustomEnvironment
    environment_config: dict = field(default_factory=dict)

    # Evaluation settings
    evaluation: BaseEvaluationConfig = field(
        default_factory=lambda: BaseEvaluationConfig(
            environment=None,  # Will be set by factory
            eval_name="custom_eval",
            max_turns={max_turns},
            num_samples={num_samples},
            output_dir=Path("results/custom"),
            verbose=True,
            show_progress=True,
        )
    )

    experiment_name: str = "custom_experiment"

    async def create_environment(self, sample_data: dict):
        """Create environment instance for a sample."""
        return self.environment_class(**self.environment_config)


# Export config instance
config = Config()


# Export prepare_messages for backward compatibility
def prepare_messages(sample_data: Dict[str, Any]) -> List[Message]:
    """Prepare initial messages from dataset sample."""
    env = config.environment_class(**config.environment_config)
    return env.prepare_messages(sample_data)
'''

        return config

    def _json_response(self, data: Any):
        """Send JSON response."""
        json_data = json.dumps(data, indent=2)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(json_data)))
        self.end_headers()
        self.wfile.write(json_data.encode("utf-8"))

    def log_message(self, format, *args):
        """Override to provide cleaner logging."""
        print(f"[{self.log_date_time_string()}] {format % args}")


def main():
    """Run the dev loop server."""
    parser = argparse.ArgumentParser(
        description="Agent dev loop tool - config builder & trace viewer"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run server on (default: 8080)"
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open browser"
    )

    args = parser.parse_args()

    # Set project root on server class
    DevLoopServer.project_root = args.project.resolve()

    # Create server
    server = HTTPServer(("localhost", args.port), DevLoopServer)

    url = f"http://localhost:{args.port}"
    print(f"\n{'='*60}")
    print(f"🚀 Agent Dev Loop Tool")
    print(f"{'='*60}")
    print(f"URL: {url}")
    print(f"Project: {DevLoopServer.project_root}")
    print(f"{'='*60}\n")

    # Open browser
    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped")


if __name__ == "__main__":
    main()
