"""Inference engine presets for kerbal.

Known-good configurations for SGLang and vLLM servers.
These encode battle-tested flags, deps, and env vars.

Usage:
    from kerbal.inference import sglang, vllm

    # Build SGLang command
    cmd, env_vars = sglang.build_command(
        model="Qwen/Qwen2.5-7B-Instruct",
        port=30000,
        tp_size=2,
        gpu_ids=[0, 1],
    )

    # Or use with serve() directly
    from kerbal import serve
    server = serve(
        client,
        command=cmd,
        workspace=workspace,
        port=30000,
        gpu_ids=[0, 1],
        env_vars=env_vars,
        deps=sglang.DEPS,
    )
"""

from kerbal.inference import sglang, vllm

__all__ = ["sglang", "vllm"]
