"""Token-level generation for TI/TO (Tokens-In/Tokens-Out).

Pure functions for generating tokens via SGLang, vLLM, or HuggingFace,
avoiding the retokenization problem that causes RL training collapse.

Usage:
    from ...inference.backends import (
        generate_sglang,
        generate_vllm,
        generate_hf,
        tokenize_chat,
        tokenize_message_with_delimiter,
    )

    # Tokenize
    input_ids = tokenize_chat(tokenizer, messages)

    # Generate (pick your backend)
    result = await generate_sglang("http://localhost:30000", input_ids, max_tokens=100)
    # or
    result = await generate_vllm("http://localhost:8000", input_ids, max_tokens=100)
    # or
    result = generate_hf(model, input_ids, max_tokens=100)

    # Result
    result.output_ids   # Generated tokens (NOT re-tokenized)
    result.logprobs     # Per-token logprobs
"""

from ...inference.backends.generate import (
    GenerationOutput,
    generate_hf,
    generate_sglang,
    generate_vllm,
)
from ...inference.backends.tokenize import (
    append_suffix_with_overlap,
    build_loss_mask,
    check_token_mismatch,
    compute_suffix_ids,
    find_largest_overlap,
    log_token_mismatch,
    tokenize_chat,
    tokenize_message_with_delimiter,
)

__all__ = [
    # Generation
    "GenerationOutput",
    "generate_sglang",
    "generate_vllm",
    "generate_hf",
    # Tokenization - prefix trick
    "tokenize_chat",
    "tokenize_message_with_delimiter",
    "build_loss_mask",
    # Tokenization - cached suffix
    "compute_suffix_ids",
    "find_largest_overlap",
    "append_suffix_with_overlap",
    # Debugging
    "check_token_mismatch",
    "log_token_mismatch",
]
