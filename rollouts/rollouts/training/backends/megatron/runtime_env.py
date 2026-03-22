"""Megatron runtime environment policy.

These helpers set backend-selection environment variables at the Megatron
boundary when current runtime stacks need an explicit hint to stay live.
"""

from __future__ import annotations

import os

from rollouts.training.models import ModelDenotation


def configure_transformer_engine_attention_env(denotation: ModelDenotation) -> dict[str, str]:
    """Set runtime attention backend knobs required by current MLA training paths.

    Current Megatron/Transformer-Engine packed-sequence MLA runs on Ampere are
    brittle if we leave backend selection fully implicit. For GLM MLA families,
    prefer the flash-attention backend and disable fused attention unless the
    caller already chose something else explicitly.
    """
    # TODO(runtime-profile): encode TE/flash backend capability in the runtime
    # profile or provider/session capability surface instead of relying on env
    # vars at model-setup time.
    if denotation.architecture.attention != "mla":
        return {}

    env_updates: dict[str, str] = {}
    desired = {
        "NVTE_FUSED_ATTN": "0",
        "NVTE_FLASH_ATTN": "1",
    }
    for key, value in desired.items():
        if os.environ.get(key) is None:
            os.environ[key] = value
            env_updates[key] = value
    return env_updates
