from __future__ import annotations

from ..image_spec import DEFAULT_IMAGE_MANIFEST_PATH, USER_IMAGE_MANIFEST_PATH


def build_gpu_runtime_probe_script() -> str:
    """Return a Python probe for GPU Python runtimes plus image manifest metadata."""
    return f"""
import json
import os
import socket
from pathlib import Path

runtime = {{"hostname": socket.gethostname(), "runtime_ok": True}}
errors = []
manifest = None
for candidate in ({DEFAULT_IMAGE_MANIFEST_PATH!r}, {USER_IMAGE_MANIFEST_PATH!r}):
    path = Path(candidate).expanduser()
    if not path.exists():
        continue
    try:
        manifest = json.loads(path.read_text())
        manifest["manifest_path"] = str(path)
        break
    except Exception as exc:
        errors.append(f"failed to read image manifest {{path}}: {{exc!r}}")

runtime["image_manifest"] = manifest

try:
    import torch

    tk_root = os.environ.get("THUNDERKITTENS_ROOT", "/root/ThunderKittens")
    try:
        import ninja  # noqa: F401
        ninja_available = True
    except Exception:
        ninja_available = False
    try:
        import setuptools  # noqa: F401
        setuptools_available = True
    except Exception:
        setuptools_available = False

    torch_info = {{
        "available": True,
        "version": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "ninja_available": ninja_available,
        "setuptools_available": setuptools_available,
        "thunderkittens_root_exists": os.path.isdir(tk_root),
        "thunderkittens_root": tk_root,
    }}
    for module_name, field_name in (
        ("triton", "triton_available"),
        ("cupy", "cupy_available"),
        ("tilelang", "tilelang_available"),
        ("cutlass", "cutlass_available"),
    ):
        try:
            __import__(module_name)
            torch_info[field_name] = True
        except Exception:
            torch_info[field_name] = False
    if torch_info["cuda_available"]:
        torch_info["device_name"] = torch.cuda.get_device_name(0)
    runtime["torch"] = torch_info
    runtime["thunderkittens_root_exists"] = torch_info["thunderkittens_root_exists"]
    runtime["thunderkittens_root"] = tk_root
except Exception as exc:
    runtime["torch"] = {{"available": False}}
    runtime["runtime_ok"] = False
    runtime["error"] = f"import torch failed: {{exc!r}}"
    errors.append(runtime["error"])

runtime["errors"] = errors
print(json.dumps(runtime))
"""
