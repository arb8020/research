"""Console script entry module for corpus-proximity CLI."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_cli_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parent
    module_path = repo_root / "examples" / "corpus-proximity" / "cli.py"
    if not module_path.exists():
        raise FileNotFoundError(f"CLI module not found: {module_path}")

    module_dir = module_path.parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

    spec = importlib.util.spec_from_file_location("corpus_proximity_cli_impl", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: Iterable[str] | None = None) -> Any:
    cli_module = _load_cli_module()
    main_func = getattr(cli_module, "main", None)
    assert callable(main_func), "CLI module missing 'main' callable"
    return main_func(list(argv) if argv is not None else None)


if __name__ == "__main__":
    sys.exit(main())
