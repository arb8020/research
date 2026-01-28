"""Serving Recipe Schema

Frozen dataclasses defining how to serve a model optimally.
Based on qwen3_next/src/config/base_config.py but simplified for serving only.

A recipe fully specifies:
1. What model to serve (model, quantization, context length)
2. How to serve it (TP, attention backend, memory settings)
3. What hardware to target (GPU type, count)
4. How to set up the environment (deps, env vars)

Usage:
    from recipes.schema import ServingRecipe, ModelConfig, EngineConfig, TargetConfig

    recipe = ServingRecipe(
        name="qwen3-0.6b-4090",
        model=ModelConfig(model="Qwen/Qwen3-0.6B"),
        engine=EngineConfig(engine="sglang"),
        target=TargetConfig(gpu_type="RTX4090", gpu_count=1),
    )

    # Serialize to JSON
    recipe.to_json("recipes/qwen3-0.6b-4090.json")

    # Load from JSON
    recipe = ServingRecipe.from_json("recipes/qwen3-0.6b-4090.json")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """What model to serve."""

    model: str  # HuggingFace model ID (e.g., "Qwen/Qwen3-0.6B")

    # Precision and quantization
    dtype: str = "auto"  # auto, bfloat16, float16
    quantization: str | None = None  # None, fp8, awq, gptq

    # Context
    max_model_len: int | None = None  # None = model default
    trust_remote_code: bool = True

    def __post_init__(self) -> None:
        assert len(self.model) > 0, "model cannot be empty"
        assert self.dtype in ("auto", "bfloat16", "float16", "float32")
        if self.quantization is not None:
            assert self.quantization in ("fp8", "awq", "gptq", "int8")
        if self.max_model_len is not None:
            assert self.max_model_len > 0


@dataclass(frozen=True)
class SpeculativeConfig:
    """Speculative decoding configuration.

    Supports:
    - Draft model (traditional spec decode)
    - EAGLE/NEXTN (SGLang native)
    - MTP (Qwen3-Next native multi-token prediction)
    """

    # Method selection
    method: str  # "draft_model", "eagle", "nextn", "mtp"

    # Draft model config (for method="draft_model")
    draft_model: str | None = None  # HuggingFace model ID
    num_speculative_tokens: int = 5

    # EAGLE/NEXTN config (for method in ["eagle", "nextn"])
    num_steps: int | None = None  # Verification rounds
    num_draft_tokens: int | None = None  # Draft tokens per step
    eagle_topk: int | None = None  # Branching factor

    def __post_init__(self) -> None:
        assert self.method in ("draft_model", "eagle", "nextn", "mtp")
        if self.method == "draft_model":
            assert self.draft_model is not None, "draft_model required for method='draft_model'"
        assert self.num_speculative_tokens > 0

    def to_sglang_config(self) -> dict[str, Any]:
        """Convert to SGLang speculative_config dict."""
        if self.method == "draft_model":
            return {
                "method": "draft_model",
                "draft_model_path": self.draft_model,
                "num_speculative_tokens": self.num_speculative_tokens,
            }
        elif self.method in ("eagle", "nextn"):
            config: dict[str, Any] = {"algorithm": self.method.upper()}
            if self.num_steps is not None:
                config["num_steps"] = self.num_steps
            if self.num_draft_tokens is not None:
                config["num_draft_tokens"] = self.num_draft_tokens
            if self.eagle_topk is not None:
                config["eagle_topk"] = self.eagle_topk
            return config
        elif self.method == "mtp":
            return {
                "method": "qwen3_next_mtp",
                "num_speculative_tokens": self.num_speculative_tokens,
            }
        else:
            raise ValueError(f"Unknown method: {self.method}")


@dataclass(frozen=True)
class EngineConfig:
    """How to serve the model."""

    engine: str = "sglang"  # sglang, vllm

    # Parallelism
    tensor_parallel_size: int = 1

    # Memory
    gpu_memory_utilization: float = 0.9
    enable_prefix_caching: bool = False  # RadixAttention

    # Attention backend (important for newer GPUs)
    attention_backend: str | None = None  # None=auto, triton, flashinfer

    # Server
    port: int = 30000
    host: str = "0.0.0.0"

    # Speculative decoding (optional)
    speculative: SpeculativeConfig | None = None

    # Installation source
    git_repo: str | None = None  # e.g., "https://github.com/sgl-project/sglang.git"
    git_ref: str = "main"  # Branch/tag/SHA

    def __post_init__(self) -> None:
        assert self.engine in ("sglang", "vllm")
        assert self.tensor_parallel_size >= 1
        assert 0 < self.gpu_memory_utilization <= 1.0
        assert 1024 < self.port < 65536
        if self.attention_backend is not None:
            assert self.attention_backend in ("triton", "flashinfer", "flash_attn")


@dataclass(frozen=True)
class TargetConfig:
    """What hardware to target."""

    gpu_type: str  # H100, A100, RTX4090, etc.
    gpu_count: int = 1

    # Auto-derived (can override)
    gpu_memory_gb: int | None = None
    compute_capability: str | None = None

    # Runtime
    python_version: str = "3.11"
    cuda_version: str = "12.4"

    # Cloud provider (optional metadata)
    provider: str | None = None  # runpod, vast, lambda, modal

    def __post_init__(self) -> None:
        assert len(self.gpu_type) > 0
        assert self.gpu_count >= 1

        # Auto-derive specs for known GPUs
        gpu_specs = {
            "H100": (80, "9.0"),
            "H200": (141, "9.0"),
            "A100": (80, "8.0"),
            "A100-40GB": (40, "8.0"),
            "RTX4090": (24, "8.9"),
            "RTX3090": (24, "8.6"),
            "L40S": (48, "8.9"),
            "L40": (48, "8.9"),
        }

        if self.gpu_type in gpu_specs and self.gpu_memory_gb is None:
            mem, cc = gpu_specs[self.gpu_type]
            object.__setattr__(self, "gpu_memory_gb", mem)
            object.__setattr__(self, "compute_capability", cc)


@dataclass(frozen=True)
class DepsConfig:
    """Environment dependencies — explicit, not derived.

    Tiger Style: explicitly pass options at the call site instead of
    relying on defaults or implicit derivation. A recipe that says
    engine="sglang" should NOT cause a provider to silently guess
    "pip install sglang[all]". The recipe states what it needs.
    """

    base_image: str = "debian:bookworm-slim"
    python_version: str = "3.12"
    system_packages: tuple[str, ...] = ("bash", "curl", "git", "build-essential")
    pip_packages: tuple[str, ...] = ()  # e.g., ("torch>=2.4", "sglang[all]")
    pip_index_url: str | None = None  # e.g., "https://download.pytorch.org/whl/cu124"
    pip_extra_index_url: str | None = None  # e.g., "https://pypi.org/simple"
    bootstrap_commands: tuple[str, ...] = ()  # arbitrary setup commands run after pip install

    def __post_init__(self) -> None:
        assert len(self.python_version) > 0, "python_version cannot be empty"
        assert len(self.base_image) > 0, "base_image cannot be empty"


@dataclass(frozen=True)
class EnvConfig:
    """Runtime environment variables."""

    hf_cache_dir: str = "/home/ubuntu/.cache/huggingface"
    use_hf_transfer: bool = True

    # Engine-specific env vars
    sglang_allow_longer_context: bool = False  # SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN

    # Extra env vars (escape hatch)
    extra_env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ServingRecipe:
    """Complete recipe for serving a model.

    Combines all config sections into a single deployable specification.
    """

    name: str  # Recipe identifier (e.g., "qwen3-0.6b-4090")

    model: ModelConfig
    engine: EngineConfig = field(default_factory=EngineConfig)
    target: TargetConfig = field(default_factory=lambda: TargetConfig(gpu_type="A100"))
    deps: DepsConfig = field(default_factory=DepsConfig)
    env: EnvConfig = field(default_factory=EnvConfig)

    # Optional description
    description: str = ""

    def __post_init__(self) -> None:
        assert len(self.name) > 0, "name cannot be empty"

        # Validate TP matches GPU count
        if self.engine.tensor_parallel_size != self.target.gpu_count:
            raise ValueError(
                f"tensor_parallel_size ({self.engine.tensor_parallel_size}) "
                f"must match gpu_count ({self.target.gpu_count})"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict (for JSON serialization)."""
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        """Save recipe to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ServingRecipe:
        """Load recipe from dict."""
        # Reconstruct nested dataclasses
        model = ModelConfig(**data["model"])

        engine_data = data["engine"]
        if engine_data.get("speculative"):
            engine_data["speculative"] = SpeculativeConfig(**engine_data["speculative"])
        engine = EngineConfig(**engine_data)

        target = TargetConfig(**data["target"])

        deps_data = data.get("deps", {})
        # Convert lists to tuples for frozen dataclass
        for key in ("system_packages", "pip_packages", "bootstrap_commands"):
            if key in deps_data and isinstance(deps_data[key], list):
                deps_data[key] = tuple(deps_data[key])
        deps = DepsConfig(**deps_data)

        env = EnvConfig(**data["env"])

        return cls(
            name=data["name"],
            model=model,
            engine=engine,
            target=target,
            deps=deps,
            env=env,
            description=data.get("description", ""),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ServingRecipe:
        """Load recipe from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))
