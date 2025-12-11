"""vLLM inference server configuration.

Known-good configurations for vLLM servers.
Based on battle-tested code from qwen3_next/engines/vllm.py.

Usage:
    from kerbal.inference import vllm
    from kerbal import serve

    cmd, env_vars = vllm.build_command(
        model="Qwen/Qwen2.5-7B-Instruct",
        port=30001,
        tp_size=2,
        gpu_ids=[0, 1],
    )

    server = serve(
        client,
        command=cmd,
        workspace=workspace,
        port=30001,
        gpu_ids=[0, 1],
        env_vars=env_vars,
        deps=vllm.DEPS,
    )
"""

from typing import Any


def get_deps():
    """Get vLLM dependencies as DependencyConfig."""
    from kerbal.protocol import DependencyConfig
    return DependencyConfig(
        project_name="vllm-server",
        dependencies=[
            "vllm>=0.6.0",
            "huggingface-hub>=0.20.0",
            "hf_transfer>=0.1.8",
            "requests>=2.28.0",  # For health checks
        ],
    )


def build_command(
    model: str,
    port: int = 30001,
    tp_size: int = 1,
    gpu_ids: list[int] | None = None,
    host: str = "0.0.0.0",
    max_model_len: int | None = None,
    dtype: str | None = None,
    quantization: str | None = None,
    gpu_memory_utilization: float = 0.9,
    kv_cache_dtype: str | None = None,
    enable_prefix_caching: bool = False,
    enable_chunked_prefill: bool | None = None,
    enforce_eager: bool = False,
    trust_remote_code: bool = True,
    hf_cache_dir: str = "/root/.cache/huggingface",
    enable_hf_transfer: bool = True,
    allow_long_max_model_len: bool = False,
    extra_flags: dict[str, Any] | None = None,
) -> tuple[str, dict[str, str]]:
    """Build vLLM OpenAI API server command with known-good defaults.

    Args:
        model: HuggingFace model name or path
        port: Server port (default: 30001)
        tp_size: Tensor parallel size (default: 1)
        gpu_ids: GPU IDs to use (for CUDA_VISIBLE_DEVICES)
        host: Host to bind (default: 0.0.0.0)
        max_model_len: Max model length (None for model default)
        dtype: Data type (None for auto)
        quantization: Quantization method (None for none)
        gpu_memory_utilization: GPU memory fraction (default: 0.9)
        kv_cache_dtype: KV cache dtype (None for auto)
        enable_prefix_caching: Enable prefix caching (default: False)
        enable_chunked_prefill: Enable chunked prefill (None for auto)
        enforce_eager: Disable CUDA graphs (default: False)
        trust_remote_code: Trust remote code (default: True)
        hf_cache_dir: HuggingFace cache directory
        enable_hf_transfer: Enable fast HF transfers (default: True)
        allow_long_max_model_len: Allow long context (default: False)
        extra_flags: Additional CLI flags as dict

    Returns:
        (command, env_vars) tuple
    """
    # Build command parts
    cmd_parts = [
        "python -m vllm.entrypoints.openai.api_server",
        f"--model {model}",
        f"--host {host}",
        f"--port {port}",
    ]

    # Tensor parallelism
    if tp_size > 1:
        cmd_parts.append(f"--tensor-parallel-size {tp_size}")

    # Max model length
    if max_model_len is not None:
        cmd_parts.append(f"--max-model-len {max_model_len}")

    # Dtype
    if dtype is not None and dtype != "auto":
        cmd_parts.append(f"--dtype {dtype}")

    # Quantization
    if quantization is not None:
        cmd_parts.append(f"--quantization {quantization}")

    # GPU memory utilization (only if not default)
    if gpu_memory_utilization != 0.9:
        cmd_parts.append(f"--gpu-memory-utilization {gpu_memory_utilization}")

    # KV cache dtype
    if kv_cache_dtype is not None and kv_cache_dtype != "auto":
        cmd_parts.append(f"--kv-cache-dtype {kv_cache_dtype}")

    # Prefix caching
    if enable_prefix_caching:
        cmd_parts.append("--enable-prefix-caching")

    # Chunked prefill
    if enable_chunked_prefill is not None:
        if enable_chunked_prefill:
            cmd_parts.append("--enable-chunked-prefill")
        else:
            cmd_parts.append("--enable-chunked-prefill=False")

    # Enforce eager (disable CUDA graphs)
    if enforce_eager:
        cmd_parts.append("--enforce-eager")

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

    # Allow long context
    if allow_long_max_model_len:
        env_vars["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    # Python faulthandler for better crash debugging
    env_vars["PYTHONFAULTHANDLER"] = "1"

    return command, env_vars


def build_bench_latency_command(
    model: str,
    input_len: int = 1024,
    output_len: int = 128,
    batch_size: int = 1,
    tp_size: int = 1,
    num_iters: int = 30,
    num_iters_warmup: int = 5,
    output_json: str | None = None,
    extra_flags: dict[str, Any] | None = None,
) -> str:
    """Build vLLM bench latency command.

    Args:
        model: Model name or path
        input_len: Input length
        output_len: Output length
        batch_size: Batch size
        tp_size: Tensor parallel size
        num_iters: Number of iterations
        num_iters_warmup: Number of warmup iterations
        output_json: Output JSON file path (optional)
        extra_flags: Additional flags

    Returns:
        Command string
    """
    cmd_parts = [
        "vllm bench latency",
        f"--model {model}",
        f"--input-len {input_len}",
        f"--output-len {output_len}",
        f"--batch-size {batch_size}",
        f"--tensor-parallel-size {tp_size}",
        f"--num-iters {num_iters}",
        f"--num-iters-warmup {num_iters_warmup}",
    ]

    if output_json:
        cmd_parts.append(f"--output-json {output_json}")

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
