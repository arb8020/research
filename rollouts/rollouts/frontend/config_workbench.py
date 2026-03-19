from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def list_configs(project_root: Path) -> list[dict[str, Any]]:
    configs_dir = project_root / "configs"
    if not configs_dir.exists():
        return []

    configs: list[dict[str, Any]] = []
    for config_file in sorted(configs_dir.glob("*.py")):
        if config_file.stem.startswith("_"):
            continue
        configs.append({
            "name": config_file.stem,
            "path": str(config_file.relative_to(project_root)),
            "modified": config_file.stat().st_mtime,
        })
    return configs


def load_config_payload(project_root: Path, config_name: str) -> dict[str, Any] | None:
    config_path = project_root / "configs" / f"{config_name}.py"
    if not config_path.exists():
        return None

    config_source = config_path.read_text()
    config_data: dict[str, Any] = {
        "name": config_name,
        "source": config_source,
    }

    model_match = re.search(r'model_name\s*[:=]\s*["\']([^"\']+)["\']', config_source)
    if model_match:
        config_data["model"] = model_match.group(1)

    temp_match = re.search(r"temperature\s*[:=]\s*([0-9.]+)", config_source)
    if temp_match:
        config_data["temperature"] = float(temp_match.group(1))

    prepare_msg_match = re.search(
        r"(def prepare_messages\(self.*?^    def \w+|def prepare_messages\(self.*?^class \w+|def prepare_messages\(self.*?$)",
        config_source,
        re.DOTALL | re.MULTILINE,
    )
    if prepare_msg_match:
        func_text = prepare_msg_match.group(1)
        func_text = re.sub(r"\n    (def |class )\w+.*$", "", func_text, flags=re.DOTALL)
        config_data["prepareMessages"] = func_text.strip()

    prompt_match = re.search(r'system_prompt\s*=\s*"""([^"]+)"""', config_source, re.DOTALL)
    if prompt_match:
        config_data["systemPrompt"] = prompt_match.group(1).strip()

    turns_match = re.search(r"max_turns\s*[:=]\s*(\d+)", config_source)
    if turns_match:
        config_data["maxTurns"] = int(turns_match.group(1))

    samples_match = re.search(r"num_samples\s*[:=]\s*(\d+)", config_source)
    if samples_match:
        config_data["numSamples"] = int(samples_match.group(1))

    seed_match = re.search(r"seed\s*[:=]\s*(\d+)", config_source)
    if seed_match:
        config_data["seed"] = int(seed_match.group(1))

    start_match = re.search(r"start_idx\s*[:=]\s*(\d+)", config_source)
    if start_match:
        config_data["startIdx"] = int(start_match.group(1))

    end_match = re.search(r"end_idx\s*[:=]\s*(\d+)", config_source)
    if end_match:
        config_data["endIdx"] = int(end_match.group(1))

    ssh_match = re.search(r'["\']ssh_target["\']\s*:\s*["\']([^"\']+)["\']', config_source)
    if not ssh_match:
        ssh_match = re.search(r'ssh_target\s*[:=]\s*["\']([^"\']+)["\']', config_source)
    if ssh_match:
        config_data["sshTarget"] = ssh_match.group(1)

    gpu_list_match = re.search(r'["\']cuda_device_ids["\']\s*:\s*\[([^\]]+)\]', config_source)
    if not gpu_list_match:
        gpu_list_match = re.search(r"cuda_device_ids\s*[:=]\s*\[([^\]]+)\]", config_source)
    if gpu_list_match:
        cuda_device_ids_str = gpu_list_match.group(1)
        config_data["gpuIds"] = [
            int(x.strip()) for x in cuda_device_ids_str.split(",") if x.strip().isdigit()
        ]
    else:
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

    tools_section = re.search(
        r"def get_tools\(self\).*?return \[(.*?)\]",
        config_source,
        re.DOTALL,
    )
    if tools_section:
        tools_content = tools_section.group(1).strip()
        config_data["hasTools"] = len(tools_content) > 0 and "Tool(" in tools_content
    else:
        config_data["hasTools"] = False

    return config_data


def _config_source(project_root: Path, config_name: str) -> str | None:
    config_path = project_root / "configs" / f"{config_name}.py"
    if not config_path.exists():
        return None
    return config_path.read_text()


def dataset_preview_for_config(project_root: Path, config_name: str) -> dict[str, Any]:
    config_source = _config_source(project_root, config_name)
    if config_source is None:
        return {"error": f"Config not found: {config_name}"}

    dataset_match = re.search(
        r'["\']dataset_path["\']\s*:\s*Path\(["\']([^"\']+)["\']\)', config_source
    )
    if not dataset_match:
        dataset_match = re.search(
            r'dataset_path\s*[:=]\s*Path\(["\']([^"\']+)["\']\)', config_source
        )
    if not dataset_match:
        return {"error": "Could not find dataset_path in config"}

    return preview_dataset(project_root, dataset_match.group(1))


def preview_dataset(project_root: Path, dataset_path_str: str) -> dict[str, Any]:
    dataset_path = project_root / dataset_path_str
    if not dataset_path.exists():
        return {"error": f"Dataset not found: {dataset_path_str}"}

    try:
        dataset_size = 0
        if dataset_path.suffix == ".jsonl":
            with dataset_path.open() as f:
                first_line = f.readline()
                if not first_line:
                    return {"error": "Dataset is empty"}
                sample = json.loads(first_line)
                dataset_size = 1 + sum(1 for _ in f)
        else:
            data = json.loads(dataset_path.read_text())
            if isinstance(data, list) and len(data) > 0:
                sample = data[0]
                dataset_size = len(data)
            else:
                return {"error": "Dataset is empty or not a list"}

        fields = list(sample.keys())
        preview_sample = {}
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 100:
                preview_sample[key] = value[:100] + "..."
            else:
                preview_sample[key] = value

        return {
            "datasetPath": dataset_path_str,
            "fields": fields,
            "sample": preview_sample,
            "datasetSize": dataset_size,
            "error": None,
        }
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON in dataset: {str(e)}"}
    except Exception as e:
        return {"error": f"Error reading dataset: {str(e)}"}


def parse_messages_from_config(project_root: Path, config_name: str) -> dict[str, Any]:
    config_source = _config_source(project_root, config_name)
    if config_source is None:
        return {"error": f"Config not found: {config_name}"}

    method_match = re.search(
        r"def prepare_messages\(.*?\).*?:\s*\n(.*?)(?=\ndef |\nclass |\Z)",
        config_source,
        re.DOTALL,
    )
    if not method_match:
        return {"error": "Could not find prepare_messages() function"}

    method_body = method_match.group(1)
    variable_values: dict[str, str] = {}

    triple_quote_pattern = r'(\w+)\s*=\s*"""(.*?)"""'
    for match in re.finditer(triple_quote_pattern, method_body, re.DOTALL):
        variable_values[match.group(1)] = match.group(2).strip()

    f_triple_quote_pattern = r'(\w+)\s*=\s*f"""(.*?)"""'
    for match in re.finditer(f_triple_quote_pattern, method_body, re.DOTALL):
        variable_values[match.group(1)] = match.group(2).strip()

    sample_data_pattern = r'(\w+)\s*=\s*sample_data\[["\'](\w+)["\']\]'
    for match in re.finditer(sample_data_pattern, method_body):
        variable_values[match.group(1)] = f"{{{match.group(2)}}}"

    simple_string_pattern = r'(\w+)\s*=\s*"([^"]+)"'
    for match in re.finditer(simple_string_pattern, method_body):
        variable_values.setdefault(match.group(1), match.group(2))

    return_match = re.search(r"return\s*\[(.*?)\]", method_body, re.DOTALL)
    if not return_match:
        return {"error": "Could not parse messages from return statement"}

    messages_str = return_match.group(1)
    message_pattern = (
        r'Message\(\s*role\s*=\s*["\'](\w+)["\']\s*,\s*content\s*=\s*(.*?)\s*\)(?=\s*(?:,|\]))'
    )

    messages = []
    for match in re.finditer(message_pattern, messages_str, re.DOTALL):
        role = match.group(1)
        content_expr = match.group(2).strip()

        if content_expr.startswith('f"""') or content_expr.startswith("f'''"):
            triple_match = re.search(r'f["\']{3}(.*?)["\']{3}', content_expr, re.DOTALL)
            assert triple_match is not None
            content = triple_match.group(1).strip()
        elif content_expr.startswith('"""') or content_expr.startswith("'''"):
            triple_match = re.search(r'["\']{3}(.*?)["\']{3}', content_expr, re.DOTALL)
            assert triple_match is not None
            content = triple_match.group(1).strip()
        elif content_expr.startswith('f"') or content_expr.startswith("f'"):
            quote_char = content_expr[1]
            content = content_expr[2 : content_expr.rfind(quote_char)]
        elif content_expr.startswith('"') or content_expr.startswith("'"):
            quote_char = content_expr[0]
            content = content_expr[1 : content_expr.rfind(quote_char)]
        elif content_expr in variable_values:
            content = variable_values[content_expr]
        else:
            content = content_expr

        messages.append({"role": role, "content": content})

    return {"messages": messages, "error": None}


def parse_tools_from_config(project_root: Path, config_name: str) -> dict[str, Any]:
    config_source = _config_source(project_root, config_name)
    if config_source is None:
        return {"error": f"Config not found: {config_name}"}

    method_match = re.search(
        r"def get_tools\(self\).*?:\s*\n(.*?)(?=\n    def |\nclass |\Z)",
        config_source,
        re.DOTALL,
    )
    if not method_match:
        return {"tools": [], "hasTools": False, "error": None}

    method_body = method_match.group(1)
    return_match = re.search(r"return\s*\[(.*?)\]", method_body, re.DOTALL)
    if not return_match:
        return {"tools": [], "hasTools": False, "error": None}

    tools_str = return_match.group(1)
    if not tools_str.strip() or "Tool(" not in tools_str:
        return {"tools": [], "hasTools": False, "error": None}

    tools = []
    tool_pattern = r'Tool\(\s*name\s*=\s*["\'](\w+)["\']\s*,\s*description\s*=\s*["\']{3}(.*?)["\']{3}\s*,\s*parameters\s*=\s*\{(.*?)\}\s*\)'
    for match in re.finditer(tool_pattern, tools_str, re.DOTALL):
        tool_name = match.group(1)
        tool_desc = match.group(2).strip()
        params_str = match.group(3)
        parameters = []
        param_pattern = r'["\'](\w+)["\']\s*:\s*ToolParam\(\s*type\s*=\s*["\'](\w+)["\']\s*,\s*description\s*=\s*["\']([^"\']+)["\']\s*(?:,\s*required\s*=\s*(True|False))?\s*\)'
        for param_match in re.finditer(param_pattern, params_str):
            parameters.append({
                "name": param_match.group(1),
                "type": param_match.group(2),
                "description": param_match.group(3),
                "required": param_match.group(4) != "False" if param_match.group(4) else True,
            })
        tools.append({"name": tool_name, "description": tool_desc, "parameters": parameters})

    return {"tools": tools, "hasTools": len(tools) > 0, "error": None}


def view_hook_source(project_root: Path, config_name: str) -> dict[str, Any]:
    loaded = _load_environment_file(project_root, config_name)
    if isinstance(loaded, dict):
        return loaded
    _, env_source, _env_class = loaded

    method_match = re.search(
        r"(async def on_assistant_message\(.*?\).*?:\s*\n.*?)(?=\n    async def |\n    def |\nclass |\Z)",
        env_source,
        re.DOTALL,
    )
    if not method_match:
        return {"error": "Could not find on_assistant_message() method"}

    return {"source": method_match.group(1).strip(), "error": None}


def view_environment_source(project_root: Path, config_name: str) -> dict[str, Any]:
    loaded = _load_environment_file(project_root, config_name)
    if isinstance(loaded, dict):
        return loaded
    env_file, env_source, env_class = loaded
    return {
        "source": env_source,
        "file_path": str(env_file.relative_to(project_root)),
        "class_name": env_class,
        "error": None,
    }


def _load_environment_file(project_root: Path, config_name: str) -> tuple[Path, str, str] | dict[str, Any]:
    config_source = _config_source(project_root, config_name)
    if config_source is None:
        return {"error": f"Config not found: {config_name}"}

    env_match = re.search(r"from\s+([\w.]+)\s+import\s+(\w+Environment)", config_source)
    if not env_match:
        return {"error": "Could not find environment import"}

    env_module = env_match.group(1)
    env_class = env_match.group(2)
    module_parts = env_module.split(".")
    env_file = project_root / Path(*module_parts[:-1]) / f"{module_parts[-1]}.py"
    if not env_file.exists():
        env_file = project_root / f"{module_parts[0]}.py"
    if not env_file.exists():
        return {"error": f"Could not find environment file: {env_module}"}

    return env_file, env_file.read_text(), env_class


def list_datasets(project_root: Path) -> dict[str, Any]:
    datasets_dir = project_root / "data"
    if not datasets_dir.exists():
        return {"datasets": [], "error": "data/ directory not found"}

    try:
        datasets = []
        for file_path in datasets_dir.rglob("*"):
            if file_path.suffix in [".json", ".jsonl"] and file_path.is_file():
                datasets.append({
                    "path": str(file_path.relative_to(project_root)),
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                })
        datasets.sort(key=lambda x: x["name"])
        return {"datasets": datasets, "error": None}
    except Exception as e:
        return {"datasets": [], "error": f"Error listing datasets: {str(e)}"}


def write_generated_config(project_root: Path, config_data: dict[str, Any]) -> dict[str, Any]:
    config_text = build_config_file(project_root, config_data)
    config_name = config_data.get("configName", "untitled_config")
    config_path = project_root / "configs" / f"{config_name}.py"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_text)
    return {
        "success": True,
        "file_path": str(config_path.relative_to(project_root)),
        "config_name": config_name,
    }


def build_config_file(project_root: Path, data: dict[str, Any]) -> str:
    if "cuda_device_ids" in data and not isinstance(data["cuda_device_ids"], list):
        data["cuda_device_ids"] = [data["cuda_device_ids"]]

    base_name = data.get("baseName")
    if base_name:
        return _build_from_base_config(project_root, data, base_name)
    return _build_new_config(data)


def _build_from_base_config(project_root: Path, data: dict[str, Any], base_name: str) -> str:
    base_path = project_root / "configs" / f"{base_name}.py"
    if not base_path.exists():
        return _build_new_config(data)

    config_source = base_path.read_text()

    if "model" in data:
        model_name = data["model"]
        config_source = re.sub(
            r'(model_name\s*[:=]\s*)["\']([^"\']+)["\']',
            f'\\1"{model_name}"',
            config_source,
        )

        if "claude" in model_name.lower() or "anthropic" in model_name.lower():
            provider = "anthropic"
            api_key_env_var = "ANTHROPIC_API_KEY"
            api_base = "https://api.anthropic.com"
        elif any(prefix in model_name.lower() for prefix in ["gpt", "o1", "o3"]):
            provider = "openai"
            api_key_env_var = "OPENAI_API_KEY"
            api_base = "https://api.openai.com/v1"
        else:
            provider = "openai"
            api_key_env_var = "OPENAI_API_KEY"
            api_base = "https://api.openai.com/v1"

        config_source = re.sub(
            r'(provider\s*[:=]\s*)["\']([^"\']+)["\']',
            f'\\1"{provider}"',
            config_source,
        )
        config_source = re.sub(
            r'(api_key_env_var\s*[:=]\s*)["\']([^"\']+)["\']',
            f'\\1"{api_key_env_var}"',
            config_source,
        )
        config_source = re.sub(
            r'(api_base\s*[:=]\s*)["\']([^"\']+)["\']',
            f'\\1"{api_base}"',
            config_source,
        )

    if "temperature" in data:
        config_source = re.sub(
            r"(temperature\s*[:=]\s*)([0-9.]+)",
            f"\\g<1>{data['temperature']}",
            config_source,
        )
    if "systemPrompt" in data:
        prompt = data["systemPrompt"]
        config_source = re.sub(
            r'(system_prompt\s*=\s*""")([^"]+)(""")',
            f"\\1{prompt}\\3",
            config_source,
            flags=re.DOTALL,
        )
    if "maxTurns" in data:
        config_source = re.sub(
            r"(max_turns\s*[:=]\s*)(\d+)",
            f"\\g<1>{data['maxTurns']}",
            config_source,
        )
    if "numSamples" in data:
        config_source = re.sub(
            r"(num_samples\s*[:=]\s*)(\d+)",
            f"\\g<1>{data['numSamples']}",
            config_source,
        )

    env_fields = data.get("envFields", {})
    if "sshTarget" in env_fields:
        config_source = re.sub(
            r'(ssh_target\s*[:=]\s*)["\']([^"\']+)["\']',
            f'\\1"{env_fields["sshTarget"]}"',
            config_source,
        )

    if "cuda_device_ids" in data:
        cuda_device_ids = data["cuda_device_ids"]
        cuda_device_ids_str = str(cuda_device_ids)
        config_source = re.sub(
            r"(cuda_device_ids\s*[:=]\s*)\[[^\]]*\]",
            f"\\g<1>{cuda_device_ids_str}",
            config_source,
        )
        config_source = re.sub(
            r'(["\']cuda_device_ids["\']\s*:\s*)\[[^\]]*\]',
            f"\\g<1>{cuda_device_ids_str}",
            config_source,
        )
        if cuda_device_ids:
            config_source = re.sub(
                r"(gpu_id\s*[:=]\s*)(\d+)",
                f"\\g<1>{cuda_device_ids[0]}",
                config_source,
            )
            config_source = re.sub(
                r'(["\']gpu_id["\']\s*:\s*)(\d+)',
                f"\\g<1>{cuda_device_ids[0]}",
                config_source,
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
    if "messages" in data and data["messages"]:
        existing_match = re.search(r"def prepare_messages\((.*?)\)", config_source)
        is_standalone = bool(existing_match and "self" not in existing_match.group(1))
        messages_code = generate_prepare_messages_method(
            data["messages"],
            is_standalone=is_standalone,
        )
        config_source = re.sub(
            r"def prepare_messages\(.*?\).*?:\s*\n.*?(?=\ndef |\nclass |\Z)",
            messages_code,
            config_source,
            flags=re.DOTALL,
        )
    if "tools" in data and data["tools"]:
        config_source = update_tool_descriptions(config_source, data["tools"])

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config_name = data.get("configName", "custom_config")
    header = (
        f'"""Agent configuration - {config_name}\n\n'
        f"Generated: {timestamp}\n"
        f"Based on: {base_name}\n"
        '"""\n'
    )
    return re.sub(r'^"""[^"]*"""\n', header, config_source)


def generate_prepare_messages_method(
    messages: list[dict[str, Any]],
    *,
    is_standalone: bool = True,
) -> str:
    lines = []
    if is_standalone:
        lines.append("def prepare_messages(sample_data: Dict[str, Any]) -> List[Message]:")
        lines.append('    """Prepare initial messages from dataset sample."""')
        indent = "    "
    else:
        lines.append("    def prepare_messages(self, sample_data: dict[str, Any]) -> list[Message]:")
        lines.append('        """Prepare initial messages for the agent."""')
        indent = "        "

    for i, msg in enumerate(messages):
        content = msg["content"]
        has_placeholders = "{" in content and "}" in content
        content_escaped = content.replace('"""', r"\"\"\"")
        if has_placeholders:
            def replace_placeholder(match: re.Match[str]) -> str:
                field = match.group(1)
                return f"{{sample_data.get('{field}', '')}}"

            content_escaped = re.sub(r"\{(\w+)\}", replace_placeholder, content_escaped)
            lines.append(f'{indent}msg{i}_content = f"""')
            lines.append(content_escaped)
            lines.append(f'{indent}"""')
        else:
            lines.append(f'{indent}msg{i}_content = """')
            lines.append(content_escaped)
            lines.append(f'{indent}"""')

    lines.append(f"{indent}return [")
    for i, msg in enumerate(messages):
        lines.append(f'{indent}    Message(role="{msg["role"]}", content=msg{i}_content),')
    lines.append(f"{indent}]")
    return "\n".join(lines)


def update_tool_descriptions(config_source: str, tools: list[dict[str, Any]]) -> str:
    for tool in tools:
        tool_name = tool["name"]
        tool_desc = tool["description"]
        pattern = rf'(Tool\(\s*name\s*=\s*["\']){tool_name}(["\']\s*,\s*description\s*=\s*["\']{{3}}).*?(["\']{{3}})'
        replacement = rf"\1{tool_name}\2{tool_desc}\3"
        config_source = re.sub(pattern, replacement, config_source, flags=re.DOTALL)
        for param in tool.get("parameters", []):
            param_name = param["name"]
            param_desc = param["description"]
            param_pattern = rf'(["\']){param_name}\1\s*:\s*ToolParam\(\s*type\s*=\s*["\'](\w+)["\']\s*,\s*description\s*=\s*["\']([^"\']*)["\']'
            param_replacement = (
                rf'\1{param_name}\1: ToolParam(type="\2", description="{param_desc}"'
            )
            config_source = re.sub(param_pattern, param_replacement, config_source)
    return config_source


def _build_new_config(data: dict[str, Any]) -> str:
    model_name = data.get("model", "gpt-4-turbo")
    system_prompt = data.get("systemPrompt", "You are an expert assistant.")
    num_samples = data.get("numSamples", 10)
    temperature = data.get("temperature", 0.1)
    stream_tokens = data.get("stream_tokens", True)

    ssh_target = data.get("ssh_target", "")
    cuda_device_ids = data.get("cuda_device_ids", [0])
    dataset_path = data.get("dataset_path", "data/default.json")

    if "claude" in model_name.lower() or "anthropic" in model_name.lower():
        provider = "anthropic"
        api_key_env_var = "ANTHROPIC_API_KEY"
        api_base = "https://api.anthropic.com"
    elif any(prefix in model_name.lower() for prefix in ["gpt", "o1", "o3"]):
        provider = "openai"
        api_key_env_var = "OPENAI_API_KEY"
        api_base = "https://api.openai.com/v1"
    else:
        provider = "openai"
        api_key_env_var = "OPENAI_API_KEY"
        api_base = "https://api.openai.com/v1"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f'''"""Agent configuration - Generated by dev loop tool

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

    env_name: str = "custom-environment"
    ssh_target: str = ""
    cuda_device_ids: List[int] = field(default_factory=lambda: [0])
    dataset_path: Path = field(default_factory=lambda: Path("data/default.json"))

    def get_tools(self) -> List[Tool]:
        """Return tools available to agent."""
        return []

    def prepare_messages(self, sample_data: Dict[str, Any]) -> List[Message]:
        """Prepare initial messages for task.

        Args:
            sample_data: Sample from dataset

        Returns:
            List of messages to initialize conversation
        """
        system_prompt = """{system_prompt}"""

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
        """Handle assistant messages."""
        return state


@dataclass(frozen=True)
class Config:
    """Main configuration."""

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

    environment_class: type = CustomEnvironment
    environment_config: dict = field(default_factory=lambda: {{
        'ssh_target': '{ssh_target}',
        'cuda_device_ids': {cuda_device_ids},
        'dataset_path': Path('{dataset_path}'),
    }})

    evaluation: BaseEvaluationConfig = field(
        default_factory=lambda: BaseEvaluationConfig(
            environment=None,
            eval_name="custom_eval",
            num_samples={num_samples},
            output_dir=Path("results/custom"),
            verbose=True,
            show_progress=True,
            stream_tokens={stream_tokens},
        )
    )

    experiment_name: str = "custom_experiment"

    async def create_environment(self, sample_data: dict):
        return self.environment_class(**self.environment_config)

    def to_endpoint(self):
        return self.model.to_endpoint()

    def to_eval_config(self, score_fn):
        return self.evaluation.to_eval_config(score_fn)

    @property
    def dataset_path(self):
        return self.environment_config.get('dataset_path')

    @property
    def ssh_target(self) -> str:
        return self.environment_config.get('ssh_target')

    @property
    def gpu_id(self) -> int:
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


config = Config()


def prepare_messages(sample_data: Dict[str, Any]) -> List[Message]:
    """Prepare initial messages from dataset sample.

    Note: You can use f-string placeholders to reference dataset fields.
    For example: "Solve this problem: {{problem_description}}"
    """
    system_prompt = """{system_prompt}"""

    user_prompt = f"""{{sample_data.get("problem_description") or sample_data.get("prompt") or sample_data.get("question") or sample_data.get("input") or str(sample_data)}}"""

    return [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ]
'''
