"""Cheap remote smoke for the TorchTitan witness runtime.

Reuses the real witness config and runtime contract, but stops after the
training backend init stage. This catches backend API mismatches before
vLLM startup and before the full RL loop.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

from rollouts.training.smoke import run_torchtitan_backend_init_smoke


def _load_witness_config():
    witness_path = Path(__file__).with_name("grpo_qwen3_0_6b_torchtitan_modal_witness.py")
    spec = importlib.util.spec_from_file_location("torchtitan_modal_witness", witness_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config, module.hardware


config, hardware = _load_witness_config()
config = deepcopy(config)
hardware = deepcopy(hardware)


def train(config=config, **kwargs):
    return run_torchtitan_backend_init_smoke(config=config or globals()["config"], **kwargs)
