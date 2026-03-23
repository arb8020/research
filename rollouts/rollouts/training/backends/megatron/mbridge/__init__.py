"""Custom mbridge model bridges for unsupported architectures.

mbridge (https://github.com/ISEEKYAN/mbridge) provides HF <-> Megatron weight conversion.
AutoBridge supports standard architectures (Llama, Qwen), but GLM models need custom bridges.

These bridges are registered via @register_model decorator and auto-discovered by AutoBridge.

Usage:
    # Import to register bridges before using AutoBridge
    import rollouts.training.backends.megatron.mbridge  # noqa: F401
    from mbridge import AutoBridge

    bridge = AutoBridge.from_pretrained("THUDM/GLM-4.7-Flash", trust_remote_code=True)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register_bridges() -> None:
    """Register custom bridges with mbridge.

    Call this before using AutoBridge to ensure GLM models are supported.
    Safe to call multiple times.
    """
    try:
        # Import triggers @register_model decorators
        from . import glm4, glm4moe, glm4moe_lite, qwen3, qwen3_5, qwen3_next

        logger.debug(
            "Registered custom bridges: %s",
            [glm4, glm4moe, glm4moe_lite, qwen3, qwen3_5, qwen3_next],
        )
    except ImportError as e:
        logger.warning("Failed to register custom bridges: %s", e)


# Auto-register on import
register_bridges()
