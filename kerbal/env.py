"""Environment variable helpers for remote execution.

This module handles building shell export statements for environment variables.
Purely about env vars - no knowledge of Python envs, GPUs, or deployments.

Tiger Style:
- Functions < 70 lines
- Assert preconditions
- Explicit control flow
"""


def build_env_prefix(env_vars: dict[str, str] | None) -> str:
    """Build shell export prefix for environment variables.

    Casey: Granular helper - just build the prefix string.
    Tiger Style: < 70 lines, explicit.

    Args:
        env_vars: Dictionary of environment variables to export

    Returns:
        Shell command prefix with exports, or empty string if no env_vars

    Example:
        prefix = build_env_prefix({"HF_TOKEN": "abc", "CUDA_VISIBLE_DEVICES": "0,1"})
        # Returns: "export HF_TOKEN='abc' CUDA_VISIBLE_DEVICES='0,1' && "
    """
    if not env_vars:
        return ""

    # Build export statements (quote values to handle special chars)
    exports = []
    for key, value in env_vars.items():
        assert key, "env var key cannot be empty"
        assert value is not None, f"env var {key} value cannot be None"
        # Escape single quotes in value
        escaped_value = value.replace("'", "'\\''")
        exports.append(f"{key}='{escaped_value}'")

    return "export " + " ".join(exports) + " && "
