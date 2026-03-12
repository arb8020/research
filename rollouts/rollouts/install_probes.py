"""Generic post-install probes for dependency materialization."""

from __future__ import annotations

import base64

DEFAULT_PYTHON_PROBE_PACKAGES: tuple[str, ...] = (
    "torch",
    "torchtitan",
    "sglang",
    "transformers",
    "huggingface-hub",
)

PYTHON_RUNTIME_CONTRACT_PATH = "/tmp/rollouts-python-runtime-contract.json"


def _python_probe_script(label: str) -> str:
    return f"""
import importlib.util
import json

def exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False

data = {{
    "label": {label!r},
    "torch_version": None,
    "torch_file": None,
    "has_torch_attention_varlen": exists("torch.nn.attention.varlen"),
    "torchtitan_present": exists("torchtitan"),
    "sglang_present": exists("sglang"),
    "transformers_present": exists("transformers"),
    "huggingface_hub_present": exists("huggingface_hub"),
}}

try:
    import torch
    data["torch_version"] = getattr(torch, "__version__", None)
    data["torch_file"] = getattr(torch, "__file__", None)
except Exception as exc:
    data["torch_import_error"] = f"{{type(exc).__name__}}: {{exc}}"

print("INSTALL_PROBE_PY " + json.dumps(data, sort_keys=True))
""".strip()


def python_install_probe_command(
    label: str,
    *,
    packages: tuple[str, ...] = DEFAULT_PYTHON_PROBE_PACKAGES,
    python_bin: str = "python3",
    uv_bin: str = "~/.local/bin/uv",
) -> str:
    encoded = base64.b64encode(_python_probe_script(label).encode()).decode()
    quoted_packages = " ".join(packages)
    return (
        f"printf '%s\\n' 'INSTALL_PROBE_STEP {label}'; "
        f"{uv_bin} pip show --python {python_bin} {quoted_packages} 2>/dev/null || "
        f"{python_bin} -m pip show {quoted_packages} 2>/dev/null || true; "
        f"{python_bin} -c \"import base64; exec(base64.b64decode('"
        f"{encoded}"
        "').decode('utf-8'))\""
    )


def apt_install_probe_command(label: str, *, packages: tuple[str, ...]) -> str:
    quoted_packages = " ".join(packages)
    return (
        f"printf '%s\\n' 'INSTALL_PROBE_STEP {label}'; "
        f"dpkg-query -W {quoted_packages} 2>/dev/null || true"
    )


def _python_contract_script() -> str:
    return """
import importlib.util
import json
from pathlib import Path

CONTRACT_PATH = Path("__CONTRACT_PATH__")

def exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False

state = {
    "torch_version": None,
    "torch_file": None,
    "has_torch_attention_varlen": exists("torch.nn.attention.varlen"),
}

try:
    import torch
    state["torch_version"] = getattr(torch, "__version__", None)
    state["torch_file"] = getattr(torch, "__file__", None)
except Exception as exc:
    state["torch_import_error"] = f"{type(exc).__name__}: {exc}"

CONTRACT_PATH.write_text(json.dumps(state, sort_keys=True))
print("INSTALL_PROBE_CONTRACT " + json.dumps(state, sort_keys=True))
""".strip()


def python_runtime_contract_snapshot_command(
    label: str,
    *,
    contract_path: str = PYTHON_RUNTIME_CONTRACT_PATH,
    python_bin: str = "python3",
) -> str:
    encoded = base64.b64encode(
        _python_contract_script().replace("__CONTRACT_PATH__", contract_path).encode()
    ).decode()
    return (
        f"printf '%s\\n' 'INSTALL_PROBE_SNAPSHOT {label}'; "
        f"{python_bin} -c \"import base64; exec(base64.b64decode('"
        f"{encoded}"
        "').decode('utf-8'))\""
    )


def _python_verify_script() -> str:
    return """
import importlib.util
import json
from pathlib import Path

CONTRACT_PATH = Path("__CONTRACT_PATH__")
if not CONTRACT_PATH.exists():
    raise SystemExit(0)

expected = json.loads(CONTRACT_PATH.read_text())

def exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False

current = {
    "torch_version": None,
    "torch_file": None,
    "has_torch_attention_varlen": exists("torch.nn.attention.varlen"),
}

try:
    import torch
    current["torch_version"] = getattr(torch, "__version__", None)
    current["torch_file"] = getattr(torch, "__file__", None)
except Exception as exc:
    current["torch_import_error"] = f"{type(exc).__name__}: {exc}"

if current != expected:
    print("INSTALL_PROBE_CONTRACT_VIOLATION " + json.dumps({"expected": expected, "current": current}, sort_keys=True))
    raise SystemExit(91)

print("INSTALL_PROBE_CONTRACT_OK " + json.dumps(current, sort_keys=True))
""".strip()


def python_runtime_contract_verify_command(
    label: str,
    *,
    contract_path: str = PYTHON_RUNTIME_CONTRACT_PATH,
    python_bin: str = "python3",
) -> str:
    encoded = base64.b64encode(
        _python_verify_script().replace("__CONTRACT_PATH__", contract_path).encode()
    ).decode()
    return (
        f"printf '%s\\n' 'INSTALL_PROBE_VERIFY {label}'; "
        f"{python_bin} -c \"import base64; exec(base64.b64decode('"
        f"{encoded}"
        "').decode('utf-8'))\""
    )


def command_looks_like_install(command: str) -> bool:
    lowered = command.lower()
    return any(
        marker in lowered
        for marker in ("uv pip install", "pip install", "apt-get install", "apt install")
    )
