"""SGLang inference server configuration.

Known-good configurations for SGLang servers.
Based on battle-tested code from qwen3_next/engines/sglang.py.

Usage:
    from kerbal.inference import sglang
    from kerbal import serve

    cmd, env_vars = sglang.build_command(
        model="Qwen/Qwen2.5-7B-Instruct",
        port=30000,
        tp_size=2,
        gpu_ids=[0, 1],
    )

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

from typing import Any


# Known-good dependencies for SGLang
# From clicker/pyproject.toml - these versions work together
DEPS = DependencyConfig = None  # Set below after import

def get_deps():
    """Get SGLang dependencies as DependencyConfig."""
    from kerbal.protocol import DependencyConfig
    return DependencyConfig(
        project_name="sglang-server",
        dependencies=[
            "sglang[all] @ git+https://github.com/sgl-project/sglang.git@main#subdirectory=python",
            "huggingface-hub>=0.36,<1.0",
            "hf_transfer>=0.1.8",
            "requests>=2.28.0",  # For health checks
        ],
    )


# Valid attention backends (from SGLang source)
VALID_ATTENTION_BACKENDS = [
    "triton",
    "torch_native",
    "flex_attention",
    "nsa",
    "cutlass_mla",
    "fa3",
    "fa4",
    "flashinfer",
    "flashmla",
    "trtllm_mla",
    "trtllm_mha",
    "dual_chunk_flash_attn",
    "aiter",
    "wave",
    "intel_amx",
    "ascend",
    "intel_xpu",
]


def build_command(
    model: str,
    port: int = 30000,
    tp_size: int = 1,
    gpu_ids: list[int] | None = None,
    host: str = "0.0.0.0",
    context_length: int | None = None,
    dtype: str | None = None,
    quantization: str | None = None,
    attention_backend: str | None = None,
    mem_fraction: float = 0.9,
    kv_cache_dtype: str | None = None,
    enable_prefix_caching: bool = False,
    trust_remote_code: bool = True,
    hf_cache_dir: str = "/root/.cache/huggingface",
    enable_hf_transfer: bool = True,
    extra_flags: dict[str, Any] | None = None,
) -> tuple[str, dict[str, str]]:
    """Build SGLang launch_server command with known-good defaults.

    Args:
        model: HuggingFace model name or path
        port: Server port (default: 30000)
        tp_size: Tensor parallel size (default: 1)
        gpu_ids: GPU IDs to use (for CUDA_VISIBLE_DEVICES)
        host: Host to bind (default: 0.0.0.0)
        context_length: Max context length (None for model default)
        dtype: Data type (None for auto)
        quantization: Quantization method (None for none)
        attention_backend: Attention backend (None for auto)
        mem_fraction: GPU memory fraction (default: 0.9)
        kv_cache_dtype: KV cache dtype (None for auto)
        enable_prefix_caching: Enable RadixAttention (default: False)
        trust_remote_code: Trust remote code (default: True)
        hf_cache_dir: HuggingFace cache directory
        enable_hf_transfer: Enable fast HF transfers (default: True)
        extra_flags: Additional CLI flags as dict

    Returns:
        (command, env_vars) tuple
    """
    # Validate attention backend
    if attention_backend is not None:
        assert attention_backend in VALID_ATTENTION_BACKENDS, \
            f"Invalid attention_backend: {attention_backend}. Valid: {VALID_ATTENTION_BACKENDS}"

    # Build command parts
    cmd_parts = [
        "python -m sglang.launch_server",
        f"--model {model}",
        f"--host {host}",
        f"--port {port}",
    ]

    # Tensor parallelism
    if tp_size > 1:
        cmd_parts.append(f"--tp {tp_size}")

    # Context length
    if context_length is not None:
        cmd_parts.append(f"--context-length {context_length}")

    # Dtype
    if dtype is not None and dtype != "auto":
        cmd_parts.append(f"--dtype {dtype}")

    # Quantization
    if quantization is not None:
        cmd_parts.append(f"--quantization {quantization}")

    # Attention backend
    if attention_backend is not None:
        cmd_parts.append(f"--attention-backend {attention_backend}")

    # Memory fraction (only if not default)
    if mem_fraction != 0.9:
        cmd_parts.append(f"--mem-fraction-static {mem_fraction}")

    # KV cache dtype
    if kv_cache_dtype is not None and kv_cache_dtype != "auto":
        cmd_parts.append(f"--kv-cache-dtype {kv_cache_dtype}")

    # Prefix caching (RadixAttention)
    if enable_prefix_caching:
        cmd_parts.append("--enable-radix-cache")

    # Trust remote code
    if trust_remote_code:
        cmd_parts.append("--trust-remote-code")

    # Extra flags
    if extra_flags:
        for flag_name, flag_value in extra_flags.items():
            _add_flag(cmd_parts, flag_name, flag_value)

    command = " ".join(cmd_parts)

    # Build environment variables
    env_vars: dict[str, str] = {}

    # GPU assignment
    if gpu_ids:
        env_vars["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    # HuggingFace config
    env_vars["HF_HOME"] = hf_cache_dir
    if enable_hf_transfer:
        env_vars["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # Python faulthandler for better crash debugging
    env_vars["PYTHONFAULTHANDLER"] = "1"

    return command, env_vars


def build_bench_serving_command(
    model: str,
    port: int = 30000,
    host: str = "localhost",
    backend: str = "sglang",
    dataset_name: str = "random",
    input_len: int = 1024,
    output_len: int = 128,
    num_prompts: int = 100,
    output_file: str | None = None,
    max_concurrency: int | None = None,
    extra_flags: dict[str, Any] | None = None,
) -> str:
    """Build bench_serving command to benchmark a running SGLang server.

    Args:
        model: Model name (for tokenizer)
        port: Server port
        host: Server host
        backend: Backend type (default: sglang)
        dataset_name: Dataset to use (random, sharegpt, etc.)
        input_len: Random input length
        output_len: Random output length
        num_prompts: Number of prompts to run
        output_file: Output file path (optional)
        max_concurrency: Max concurrent requests (optional)
        extra_flags: Additional flags

    Returns:
        Command string
    """
    cmd_parts = [
        "python -m sglang.bench_serving",
        f"--backend {backend}",
        f"--base-url http://{host}:{port}",
        f"--model {model}",
        f"--dataset-name {dataset_name}",
        f"--random-input-len {input_len}",
        f"--random-output-len {output_len}",
        f"--num-prompts {num_prompts}",
    ]

    if output_file:
        cmd_parts.append(f"--output-file {output_file}")

    if max_concurrency is not None:
        cmd_parts.append(f"--max-concurrency {max_concurrency}")

    if extra_flags:
        for flag_name, flag_value in extra_flags.items():
            _add_flag(cmd_parts, flag_name, flag_value)

    return " ".join(cmd_parts)


def _add_flag(cmd_parts: list[str], flag_name: str, flag_value: Any) -> None:
    """Add a flag to command parts.

    Handles different value types:
    - bool: --flag (only if True)
    - list: --flag val1 val2
    - None: skip
    - other: --flag value
    """
    cli_flag = flag_name.replace("_", "-")

    if isinstance(flag_value, bool):
        if flag_value:
            cmd_parts.append(f"--{cli_flag}")
    elif isinstance(flag_value, (list, tuple)):
        values_str = " ".join(str(v) for v in flag_value)
        cmd_parts.append(f"--{cli_flag} {values_str}")
    elif flag_value is None:
        pass
    else:
        cmd_parts.append(f"--{cli_flag} {flag_value}")
