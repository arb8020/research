from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trusted_eval_reverse_text_config_exports_sample_scorer() -> None:
    module = _load_module(Path("rollouts/trusted_configs/eval_reverse_text_api.py").resolve())

    assert module.sample_scorer is not None
    assert module.endpoint.provider == "anthropic"
    assert module.output.experiment_name == "trusted_eval_reverse_text_api"


def test_trusted_rl_reverse_text_configs_use_expected_pipeline_modes() -> None:
    sync_module = _load_module(Path("rollouts/trusted_configs/rl_reverse_text_sync.py").resolve())
    async_module = _load_module(Path("rollouts/trusted_configs/rl_reverse_text_async.py").resolve())

    assert sync_module.config.checkpoint.pipeline_mode == "sync"
    assert async_module.config.checkpoint.pipeline_mode == "async"
    assert async_module.config.checkpoint.pipeline_queue_size == 0
