"""Functional Extractor Eval.

Converts HuggingFace models to minimal functional PyTorch code.
"""

from .eval import (
    get_spec,
    load_tasks,
    make_environment,
    prepare_messages,
    run,
    score_sample,
    setup_workspace,
)
from .modal_sandbox import ModalSandboxCodingEnvironment, SandboxConfig

__all__ = [
    "ModalSandboxCodingEnvironment",
    "SandboxConfig",
    "get_spec",
    "load_tasks",
    "make_environment",
    "prepare_messages",
    "run",
    "score_sample",
    "setup_workspace",
]
