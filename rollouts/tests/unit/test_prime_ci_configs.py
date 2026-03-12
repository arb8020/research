from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from rollouts.config_status import ConfigConfidence


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prime_ci_reverse_text_eval_exports_sample_scorer_and_status() -> None:
    module = _load_module(Path("rollouts/configs/prime_ci/reverse_text/eval_api.py").resolve())

    assert module.sample_scorer is not None
    assert module.endpoint.provider == "anthropic"
    assert module.output.experiment_name == "prime_ci_reverse_text_eval_api"
    assert module.config_status.confidence == ConfigConfidence.IMPORT_TESTED
    assert module.config_status.verified_commit == "70bce1bf"


def test_prime_ci_rl_configs_use_expected_pipeline_modes() -> None:
    sync_module = _load_module(Path("rollouts/configs/prime_ci/reverse_text/rl_sync.py").resolve())
    async_module = _load_module(
        Path("rollouts/configs/prime_ci/reverse_text/rl_async.py").resolve()
    )
    alphabet_sort_module = _load_module(
        Path("rollouts/configs/prime_ci/alphabet_sort/rl.py").resolve()
    )

    assert sync_module.config.checkpoint.pipeline_mode == "sync"
    assert async_module.config.checkpoint.pipeline_mode == "async"
    assert async_module.config.checkpoint.pipeline_queue_size == 0
    assert alphabet_sort_module.config.checkpoint.pipeline_mode == "async"
    assert alphabet_sort_module.config_status.confidence == ConfigConfidence.IMPORT_TESTED


def test_prime_ci_stubs_are_explicit_drafts() -> None:
    for rel_path in [
        "rollouts/configs/prime_ci/wordle/rl.py",
        "rollouts/configs/prime_ci/wiki_search/rl.py",
        "rollouts/configs/prime_ci/hendrycks_sanity/rl.py",
        "rollouts/configs/prime_ci/acereason_math/rl.py",
        "rollouts/configs/prime_ci/multimodal_color_codeword/rl.py",
    ]:
        module = _load_module(Path(rel_path).resolve())
        assert module.config_status.confidence == ConfigConfidence.DRAFT
