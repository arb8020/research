# Re-export from _legacy for backwards compatibility
# Import directly from submodules to avoid broken _legacy/__init__.py
from .._legacy.backends.generate import (
    GenerationOutput,
    generate_hf,
    generate_sglang,
    generate_vllm,
)
from .._legacy.backends.tokenize import (
    append_suffix_with_overlap,
    compute_suffix_ids,
    log_token_mismatch,
    tokenize_chat,
    tokenize_message_with_delimiter,
)

__all__ = [
    "GenerationOutput",
    "append_suffix_with_overlap",
    "compute_suffix_ids",
    "generate_hf",
    "generate_sglang",
    "generate_vllm",
    "log_token_mismatch",
    "tokenize_chat",
    "tokenize_message_with_delimiter",
]
