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

        # Extract system prompt (multiline string)
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

        # Extract environment-specific fields (kernel env)
        ssh_match = re.search(r'ssh_target\s*[:=]\s*["\']([^"\']+)["\']', config_source)
        if ssh_match:
            config_data["sshTarget"] = ssh_match.group(1)

        gpu_match = re.search(r'gpu_id\s*[:=]\s*(\d+)', config_source)
        if gpu_match:
            config_data["gpuId"] = int(gpu_match.group(1))

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
        """Get preview of dataset for a config - shows first sample with all fields.

        This helps users see what fields are available when building messages.
        """
        import re

        # Load config to get dataset path
        config_path = self.project_root / "configs" / f"{config_name}.py"

        if not config_path.exists():
            self._json_response({
                "error": f"Config not found: {config_name}",
                "datasetPath": None,
                "fields": [],
                "sample": {}
            })
            return

        # Read config and extract dataset path
        config_source = config_path.read_text()
        dataset_match = re.search(r'dataset_path\s*[:=]\s*Path\(["\']([^"\']+)["\']\)', config_source)

        if not dataset_match:
            self._json_response({
                "error": "No dataset_path found in config",
                "datasetPath": None,
                "fields": [],
                "sample": {}
            })
            return

        dataset_path_str = dataset_match.group(1)
        dataset_path = self.project_root / dataset_path_str

        if not dataset_path.exists():
            self._json_response({
                "error": f"Dataset file not found: {dataset_path_str}",
                "datasetPath": dataset_path_str,
                "fields": [],
                "sample": {}
            })
            return

        # Load first sample from dataset
        try:
            with open(dataset_path, 'r') as f:
                content = f.read()

            # Try JSON array format first
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    if len(data) == 0:
                        self._json_response({
                            "error": "Dataset array is empty",
                            "datasetPath": dataset_path_str,
                            "fields": [],
                            "sample": {}
                        })
                        return
                    sample = data[0]
                else:
                    # Single JSON object
                    sample = data
            except json.JSONDecodeError:
                # Try JSONL format (one JSON object per line)
                first_line = content.split('\n')[0].strip()
                if not first_line:
                    self._json_response({
                        "error": "Dataset is empty",
                        "datasetPath": dataset_path_str,
                        "fields": [],
                        "sample": {}
                    })
                    return
                sample = json.loads(first_line)

            # Extract field names and truncate long values for preview
            fields = list(sample.keys())
            preview_sample = {}

            for key, value in sample.items():
                # Truncate long strings for preview
                if isinstance(value, str) and len(value) > 100:
                    preview_sample[key] = value[:100] + "..."
                else:
                    preview_sample[key] = value

            self._json_response({
                "error": None,
                "datasetPath": dataset_path_str,
                "fields": fields,
                "sample": preview_sample
            })

        except json.JSONDecodeError as e:
            self._json_response({
                "error": f"Invalid JSON in dataset: {str(e)}",
                "datasetPath": dataset_path_str,
                "fields": [],
                "sample": {}
            })
        except Exception as e:
            self._json_response({
                "error": f"Error loading dataset: {str(e)}",
                "datasetPath": dataset_path_str,
                "fields": [],
                "sample": {}
            })

    def _parse_messages(self, config_name: str):
        """Parse prepare_messages() method from config to extract message list.

        Extracts role and content from each Message() call in the return statement.
        """
        import re

        config_path = self.project_root / "configs" / f"{config_name}.py"

        if not config_path.exists():
            self._json_response({
                "error": f"Config not found: {config_name}",
                "messages": []
            })
            return

        config_source = config_path.read_text()

        # Find prepare_messages method
        method_match = re.search(
            r'def prepare_messages\(.*?\):.*?return \[(.*?)\]',
            config_source,
            re.DOTALL
        )

        if not method_match:
            self._json_response({
                "error": "No prepare_messages() method found or it has complex logic",
                "messages": []
            })
            return

        messages_block = method_match.group(1)

        # Parse each Message() call
        # Match: Message(role="system", content="""...""") or Message(role="user", content=f"""...""")
        message_pattern = r'Message\(\s*role\s*=\s*["\'](\w+)["\']\s*,\s*content\s*=\s*(.+?)\s*\)'

        messages = []
        for match in re.finditer(message_pattern, messages_block, re.DOTALL):
            role = match.group(1)
            content_expr = match.group(2).strip()

            # Extract content from triple-quoted strings or regular strings
            # Handle: """content""", "content", f"""content""", f"content"
            content = None

            # Try triple-quoted string first (most common in our configs)
            triple_quote_match = re.match(r'f?"""(.+?)"""', content_expr, re.DOTALL)
            if triple_quote_match:
                content = triple_quote_match.group(1).strip()
            else:
                # Try single/double quoted string
                quote_match = re.match(r'f?["\'](.+?)["\']', content_expr, re.DOTALL)
                if quote_match:
                    content = quote_match.group(1).strip()
                else:
                    # Complex expression (f-string with sample_data, etc.) - preserve as-is
                    content = content_expr

            if content is not None:
                messages.append({
                    "role": role,
                    "content": content
                })

        if not messages:
            self._json_response({
                "error": "Could not parse messages - prepare_messages() may have complex logic",
                "messages": []
            })
            return

        self._json_response({
            "error": None,
            "messages": messages
        })

    def _parse_tools(self, config_name: str):
        """Parse get_tools() method from config to extract tool definitions.

        Extracts name, description, and parameters for each tool.
        """
        import re

        config_path = self.project_root / "configs" / f"{config_name}.py"

        if not config_path.exists():
            self._json_response({
                "error": f"Config not found: {config_name}",
                "tools": [],
                "hasTools": False
            })
            return

        config_source = config_path.read_text()

        # Find get_tools method
        method_match = re.search(
            r'def get_tools\(self\).*?return \[(.*?)\]',
            config_source,
            re.DOTALL
        )

        if not method_match:
            self._json_response({
                "error": None,
                "tools": [],
                "hasTools": False
            })
            return

        tools_block = method_match.group(1).strip()

        # Check if empty list
        if not tools_block or "Tool(" not in tools_block:
            self._json_response({
                "error": None,
                "tools": [],
                "hasTools": False
            })
            return

        # Parse each Tool() definition
        # This is simplified - we extract name and description only (editable fields)
        # Parameter parsing is complex, so we'll keep it simple for now
        tools = []

        # Find all Tool() blocks
        tool_pattern = r'Tool\((.*?)\)'
        for tool_match in re.finditer(tool_pattern, tools_block, re.DOTALL):
            tool_content = tool_match.group(1)

            # Extract tool name
            name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', tool_content)
            if not name_match:
                continue
            tool_name = name_match.group(1)

            # Extract tool description
            desc_match = re.search(r'description\s*=\s*["\']([^"\']+)["\']', tool_content)
            tool_description = desc_match.group(1) if desc_match else ""

            # Extract parameters (simplified - just names and descriptions)
            params = []
            properties_match = re.search(r'properties\s*=\s*\{(.*?)\}', tool_content, re.DOTALL)
            if properties_match:
                properties_block = properties_match.group(1)

                # Find each parameter definition
                param_pattern = r'["\'](\w+)["\']\s*:\s*\{([^}]+)\}'
                for param_match in re.finditer(param_pattern, properties_block):
                    param_name = param_match.group(1)
                    param_def = param_match.group(2)

                    # Extract param type
                    type_match = re.search(r'["\']type["\']\s*:\s*["\']([^"\']+)["\']', param_def)
                    param_type = type_match.group(1) if type_match else "string"

                    # Extract param description
                    desc_match = re.search(r'["\']description["\']\s*:\s*["\']([^"\']+)["\']', param_def)
                    param_desc = desc_match.group(1) if desc_match else ""

                    # Check if required
                    required_match = re.search(r'required\s*=\s*\[(.*?)\]', tool_content, re.DOTALL)
                    is_required = False
                    if required_match:
                        required_list = required_match.group(1)
                        is_required = f'"{param_name}"' in required_list or f"'{param_name}'" in required_list

                    params.append({
                        "name": param_name,
                        "type": param_type,
                        "description": param_desc,
                        "required": is_required
                    })

            tools.append({
                "name": tool_name,
                "description": tool_description,
                "parameters": params
            })

        self._json_response({
            "error": None,
            "tools": tools,
            "hasTools": len(tools) > 0
        })

    def _update_tool_descriptions(self, config_source: str, tools: list[dict]) -> str:
        """Update tool and parameter descriptions in config source.

        Args:
            config_source: Python source code of config
            tools: List of tool dicts with name, description, and parameters

        Returns:
            Updated config source with new descriptions
        """
        import re

        for tool in tools:
            tool_name = tool["name"]
            new_description = tool["description"]

            # Find and replace tool description
            # Pattern: name="tool_name", ... description="old description"
            pattern = f'(name\\s*=\\s*["\'{tool_name}"][\'"],.*?description\\s*=\\s*["\'])([^"\']+)(["\'])'
            config_source = re.sub(
                pattern,
                f'\\g<1>{new_description}\\g<3>',
                config_source,
                flags=re.DOTALL
            )

            # Update parameter descriptions
            for param in tool.get("parameters", []):
                param_name = param["name"]
                param_description = param.get("description", "")

                # Find parameter definition within properties dict
                # Pattern: "param_name": {..., "description": "old desc", ...}
                param_pattern = f'(["\'{param_name}"][\'"]\\s*:\\s*{{.*?["\']description["\']\\s*:\\s*["\'])([^"\']+)(["\'])'
                config_source = re.sub(
                    param_pattern,
                    f'\\g<1>{param_description}\\g<3>',
                    config_source,
                    flags=re.DOTALL
                )

        return config_source

    def _generate_prepare_messages_method(self, messages: list[dict]) -> str:
        """Generate prepare_messages() method code from message list.

        Args:
            messages: List of {role, content} dicts from UI

        Returns:
            Python code for prepare_messages() method
        """
        # Build message creation code
        message_lines = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Check if content contains {placeholders} for f-string
            if "{" in content and "}" in content:
                # Use f-string for dynamic content
                # Escape any existing triple quotes in content
                escaped_content = content.replace('"""', r'\"\"\"')
                message_lines.append(f'            Message(role="{role}", content=f"""{escaped_content}"""),')
            else:
                # Use plain string for static content
                escaped_content = content.replace('"""', r'\"\"\"')
                message_lines.append(f'            Message(role="{role}", content="""{escaped_content}"""),')

        messages_code = "\n".join(message_lines)

        method_code = f'''def prepare_messages(self, sample_data: Dict[str, Any]) -> List[Message]:
        """Prepare initial messages for the agent.

        Generated by rollouts frontend.
        """
        return [
{messages_code}
        ]'''

        return method_code

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

        # Replace prepare_messages() if messages provided from UI
        if "messages" in data and data["messages"]:
            new_prepare_messages = self._generate_prepare_messages_method(data["messages"])
            # Find and replace the entire prepare_messages method
            config_source = re.sub(
                r'def prepare_messages\(self, sample_data:.*?\n        \]',
                new_prepare_messages.rstrip() + '\n        ]',
                config_source,
                flags=re.DOTALL
            )

        # Update tool descriptions if tools provided from UI
        if "tools" in data and data["tools"]:
            config_source = self._update_tool_descriptions(config_source, data["tools"])

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
