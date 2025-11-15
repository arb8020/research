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
        elif path == "/api/models":
            self._list_models()
        elif path == "/api/list-datasets":
            self._list_datasets()
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

        # Try to match gpu_ids as a list first
        gpu_list_match = re.search(r'["\']gpu_ids["\']\s*:\s*\[([^\]]+)\]', config_source)
        if not gpu_list_match:
            gpu_list_match = re.search(r'gpu_ids\s*[:=]\s*\[([^\]]+)\]', config_source)

        if gpu_list_match:
            # Parse list of GPU IDs
            gpu_ids_str = gpu_list_match.group(1)
            config_data["gpuIds"] = [int(x.strip()) for x in gpu_ids_str.split(',') if x.strip().isdigit()]
        else:
            # Fallback to single gpu_id for backwards compatibility
            gpu_match = re.search(r'["\']gpu_id["\']\s*:\s*(\d+)', config_source)
            if not gpu_match:
                gpu_match = re.search(r'gpu_id\s*[:=]\s*(\d+)', config_source)
            if gpu_match:
                config_data["gpuIds"] = [int(gpu_match.group(1))]

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

    def _get_dataset_preview(self, config_name: str):
        """Get preview of dataset for a config - shows first sample with all fields."""
        import re

        config_path = self.project_root / "configs" / f"{config_name}.py"

        if not config_path.exists():
            self._json_response({"error": f"Config not found: {config_name}"})
            return

        # Read config to extract dataset path
        config_source = config_path.read_text()

        # Try to find dataset_path in the source
        dataset_match = re.search(r'["\']dataset_path["\']\s*:\s*Path\(["\']([^"\']+)["\']\)', config_source)
        if not dataset_match:
            dataset_match = re.search(r'dataset_path\s*[:=]\s*Path\(["\']([^"\']+)["\']\)', config_source)

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
                "error": None
            })

        except Exception as e:
            self._json_response({"error": f"Error reading dataset: {str(e)}"})

    def _parse_messages(self, config_name: str):
        """Parse prepare_messages() method from config to extract message list."""
        import re

        config_path = self.project_root / "configs" / f"{config_name}.py"

        if not config_path.exists():
            self._json_response({"error": f"Config not found: {config_name}"})
            return

        config_source = config_path.read_text()

        # Find prepare_messages method
        method_match = re.search(
            r'def prepare_messages\(self.*?\).*?:\s*\n(.*?)(?=\n    def |\nclass |\Z)',
            config_source,
            re.DOTALL
        )

        if not method_match:
            self._json_response({"error": "Could not find prepare_messages() method"})
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
            variable_values[var_name] = f'{{{field_name}}}'

        # Match simple string assignments
        simple_string_pattern = r'(\w+)\s*=\s*"([^"]+)"'
        for match in re.finditer(simple_string_pattern, method_body):
            var_name = match.group(1)
            var_value = match.group(2)
            # Don't override if already captured by triple-quote pattern
            if var_name not in variable_values:
                variable_values[var_name] = var_value

        # Find return statement with Message list
        return_match = re.search(r'return\s*\[(.*?)\]', method_body, re.DOTALL)

        if not return_match:
            self._json_response({"error": "Could not parse messages from return statement"})
            return

        messages_str = return_match.group(1)

        # Extract Message() calls
        messages = []
        message_pattern = r'Message\(\s*role\s*=\s*["\'](\w+)["\']\s*,\s*content\s*=\s*(.*?)\s*\)(?=\s*(?:,|\]))'

        for match in re.finditer(message_pattern, messages_str, re.DOTALL):
            role = match.group(1)
            content_expr = match.group(2).strip()

            # Handle different content formats
            if content_expr.startswith('f"""') or content_expr.startswith("f'''"):
                # f-string with triple quotes
                content = re.search(r'f["\']{{3}}(.*?)["\']{{3}}', content_expr, re.DOTALL).group(1).strip()
            elif content_expr.startswith('"""') or content_expr.startswith("'''"):
                # Regular triple-quoted string
                content = re.search(r'["\']{{3}}(.*?)["\']{{3}}', content_expr, re.DOTALL).group(1).strip()
            elif content_expr.startswith('f"') or content_expr.startswith("f'"):
                # f-string with single quotes
                quote_char = content_expr[1]
                content = content_expr[2:content_expr.rfind(quote_char)]
            elif content_expr.startswith('"') or content_expr.startswith("'"):
                # Regular string
                quote_char = content_expr[0]
                content = content_expr[1:content_expr.rfind(quote_char)]
            elif content_expr in variable_values:
                # Variable reference - look up the value
                content = variable_values[content_expr]
            else:
                # Complex expression - use as-is
                content = content_expr

            messages.append({"role": role, "content": content})

        self._json_response({"messages": messages, "error": None})

    def _parse_tools(self, config_name: str):
        """Parse get_tools() method from config to extract tool definitions."""
        import re

        config_path = self.project_root / "configs" / f"{config_name}.py"

        if not config_path.exists():
            self._json_response({"error": f"Config not found: {config_name}"})
            return

        config_source = config_path.read_text()

        # Find get_tools method
        method_match = re.search(
            r'def get_tools\(self\).*?:\s*\n(.*?)(?=\n    def |\nclass |\Z)',
            config_source,
            re.DOTALL
        )

        if not method_match:
            self._json_response({"tools": [], "hasTools": False, "error": None})
            return

        method_body = method_match.group(1)

        # Find return statement
        return_match = re.search(r'return\s*\[(.*?)\]', method_body, re.DOTALL)

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
                    "required": param_required
                })

            tools.append({
                "name": tool_name,
                "description": tool_desc,
                "parameters": parameters
            })

        self._json_response({"tools": tools, "hasTools": len(tools) > 0, "error": None})

    def _view_hook(self, config_name: str):
        """View on_assistant_message() implementation from environment."""
        import re

        config_path = self.project_root / "configs" / f"{config_name}.py"

        if not config_path.exists():
            self._json_response({"error": f"Config not found: {config_name}"})
            return

        config_source = config_path.read_text()

        # Try to find environment class name or environment file
        env_match = re.search(r'from\s+([\w.]+)\s+import\s+(\w+Environment)', config_source)

        if not env_match:
            self._json_response({"error": "Could not find environment import"})
            return

        env_module = env_match.group(1)
        env_class = env_match.group(2)

        # Convert module path to file path
        module_parts = env_module.split('.')
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
            r'(async def on_assistant_message\(.*?\).*?:\s*\n.*?)(?=\n    async def |\n    def |\nclass |\Z)',
            env_source,
            re.DOTALL
        )

        if not method_match:
            self._json_response({"error": "Could not find on_assistant_message() method"})
            return

        method_source = method_match.group(1).strip()

        self._json_response({"source": method_source, "error": None})

    def _list_models(self):
        """Fetch available models from OpenAI and Anthropic APIs."""
        import os
        import urllib.request
        import urllib.error

        models = []
        errors = []

        # Fetch OpenAI models
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            try:
                req = urllib.request.Request(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {openai_key}"}
                )
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    # Filter to relevant models (gpt-4*, o1*, o3*)
                    for model in data.get("data", []):
                        model_id = model.get("id", "")
                        if any(model_id.startswith(prefix) for prefix in ["gpt-4", "o1", "o3"]):
                            models.append({
                                "id": model_id,
                                "provider": "openai",
                                "name": model_id
                            })
            except Exception as e:
                errors.append(f"OpenAI: {str(e)}")

        # Fetch Anthropic models
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                req = urllib.request.Request(
                    "https://api.anthropic.com/v1/models",
                    headers={
                        "x-api-key": anthropic_key,
                        "anthropic-version": "2023-06-01"
                    }
                )
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    for model in data.get("data", []):
                        model_id = model.get("id", "")
                        models.append({
                            "id": model_id,
                            "provider": "anthropic",
                            "name": model.get("display_name", model_id)
                        })
            except Exception as e:
                errors.append(f"Anthropic: {str(e)}")

        self._json_response({
            "models": models,
            "errors": errors if errors else None
        })

    def _list_datasets(self):
        """List available dataset files in the datasets directory."""
        datasets_dir = self.project_root / "datasets"

        if not datasets_dir.exists():
            self._json_response({"datasets": [], "error": "datasets/ directory not found"})
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
                        "size": file_path.stat().st_size
                    })

            # Sort by name
            datasets.sort(key=lambda x: x["name"])

            self._json_response({"datasets": datasets, "error": None})

        except Exception as e:
            self._json_response({"datasets": [], "error": f"Error listing datasets: {str(e)}"})

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
        # Normalize gpu_ids: convert to list if needed
        if "gpu_ids" in data:
            gpu_ids = data["gpu_ids"]
            if not isinstance(gpu_ids, list):
                data["gpu_ids"] = [gpu_ids]

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

        # Handle both gpu_ids (list) and gpuId (single, backwards compat)
        if "gpu_ids" in data:
            gpu_ids = data["gpu_ids"]
            gpu_ids_str = str(gpu_ids)  # Convert list to string like [0, 1, 2]

            # Try to replace gpu_ids list first
            config_source = re.sub(
                r'(gpu_ids\s*[:=]\s*)\[[^\]]*\]',
                f'\\g<1>{gpu_ids_str}',
                config_source
            )
            # Also try in dict format
            config_source = re.sub(
                r'(["\']gpu_ids["\']\s*:\s*)\[[^\]]*\]',
                f'\\g<1>{gpu_ids_str}',
                config_source
            )
            # Fallback: replace old gpu_id with first GPU from list
            if gpu_ids:
                config_source = re.sub(
                    r'(gpu_id\s*[:=]\s*)(\d+)',
                    f'\\g<1>{gpu_ids[0]}',
                    config_source
                )
                config_source = re.sub(
                    r'(["\']gpu_id["\']\s*:\s*)(\d+)',
                    f'\\g<1>{gpu_ids[0]}',
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

        # Update messages if provided
        if "messages" in data and data["messages"]:
            messages_code = self._generate_prepare_messages_method(data["messages"])
            # Replace the existing prepare_messages method
            config_source = re.sub(
                r'def prepare_messages\(self.*?\).*?:\s*\n.*?(?=\n    def |\nclass |\Z)',
                messages_code,
                config_source,
                flags=re.DOTALL
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

    def _generate_prepare_messages_method(self, messages: list[dict]) -> str:
        """Generate prepare_messages() method code from message list."""
        lines = []
        lines.append("    def prepare_messages(self, sample_data: dict[str, Any]) -> list[Message]:")
        lines.append('        """Prepare initial messages for the agent."""')

        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]

            # Check if content has f-string placeholders like {field_name}
            has_placeholders = "{" in content and "}" in content

            # Escape any existing triple quotes in content
            content_escaped = content.replace('"""', r'\"\"\"')

            if has_placeholders:
                # Use f-string
                lines.append(f"        msg{i}_content = f\"\"\"")
                lines.append(content_escaped)
                lines.append('        """')
            else:
                # Use regular string
                lines.append(f'        msg{i}_content = """')
                lines.append(content_escaped)
                lines.append('        """')

        # Build return statement
        lines.append("        return [")
        for i, msg in enumerate(messages):
            role = msg["role"]
            lines.append(f'            Message(role="{role}", content=msg{i}_content),')
        lines.append("        ]")

        return "\n".join(lines)

    def _update_tool_descriptions(self, config_source: str, tools: list[dict]) -> str:
        """Update tool and parameter descriptions in config source."""
        import re

        for tool in tools:
            tool_name = tool["name"]
            tool_desc = tool["description"]

            # Update tool description
            pattern = rf'(Tool\(\s*name\s*=\s*["\']){tool_name}(["\']\\s*,\s*description\s*=\s*["\']{{3}}).*?(["\']{{3}})'
            replacement = rf'\1{tool_name}\2{tool_desc}\3'
            config_source = re.sub(pattern, replacement, config_source, flags=re.DOTALL)

            # Update parameter descriptions
            for param in tool.get("parameters", []):
                param_name = param["name"]
                param_desc = param["description"]

                # Find and replace parameter description
                param_pattern = rf'(["\']){param_name}\1\s*:\s*ToolParam\(\s*type\s*=\s*["\'](\w+)["\']\\s*,\s*description\s*=\s*["\']([^"\']*)["\']'
                param_replacement = rf'\1{param_name}\1: ToolParam(type="\2", description="{param_desc}"'
                config_source = re.sub(param_pattern, param_replacement, config_source)

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
