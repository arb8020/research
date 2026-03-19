"""Evaluation configuration dataclasses.

Composable building blocks for running evals against APIs or SGLang endpoints.
Follows the same pattern as training/configs.py for consistency.

Example usage:
    # API endpoint (Anthropic)
    endpoint = EndpointConfig(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
    )

    # SGLang endpoint (local server)
    endpoint = EndpointConfig(
        provider="sglang",
        model="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:30000/v1",
    )

    # SGLang with auto-provisioning
    endpoint = EndpointConfig(
        provider="sglang",
        model="Qwen/Qwen2.5-7B-Instruct",
    )
    hardware = HardwareConfig(gpu_type="A100", provider="runpod")
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from rollouts.agents.types import AgentState
    from rollouts.agents.types import RunConfig as AgentRunConfig
    from rollouts.training.types import AttemptResult

# Reuse HardwareConfig from training
from rollouts.training.configs import HardwareConfig

__all__ = [
    "AgentRunSpec",
    "AttemptExecutor",
    "EndpointConfig",
    "EvalRunConfig",
    "EvalOutputConfig",
    "HardwareConfig",
    "SupportedStopHandler",
    "EvalStopHandler",
    "MaxTurnsStop",
    "TokenBudgetStop",
    "CostBudgetStop",
    "WallClockStop",
]


@dataclass(frozen=True)
class MaxTurnsStop:
    max_turns: int

    def __post_init__(self) -> None:
        if self.max_turns <= 0:
            raise ValueError("MaxTurnsStop.max_turns must be positive")


@dataclass(frozen=True)
class TokenBudgetStop:
    max_tokens: int

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("TokenBudgetStop.max_tokens must be positive")


@dataclass(frozen=True)
class CostBudgetStop:
    max_cost_usd: float

    def __post_init__(self) -> None:
        if self.max_cost_usd <= 0:
            raise ValueError("CostBudgetStop.max_cost_usd must be positive")


@dataclass(frozen=True)
class WallClockStop:
    max_seconds: float

    def __post_init__(self) -> None:
        if self.max_seconds <= 0:
            raise ValueError("WallClockStop.max_seconds must be positive")


SupportedStopHandler = MaxTurnsStop | TokenBudgetStop | CostBudgetStop | WallClockStop
EvalStopHandler = SupportedStopHandler | Callable[["AgentState"], "AgentState"]
AttemptExecutor = Callable[
    [dict[str, Any], str, Any | None, "AgentRunConfig"],
    "AttemptResult | Awaitable[AttemptResult]",
]


@dataclass(frozen=True)
class AgentRunSpec:
    """Per-sample agent execution spec.

    This is the eval-side product type closest to what one `run_agent(...)`
    execution needs: either an endpoint-driven agent loop or a custom
    attempt executor, plus optional environment ownership and stop/no-tool
    behavior. Dataset iteration, concurrency, retries, and output policy
    still belong to EvalRunConfig/EvalOutputConfig.
    """

    endpoint: EndpointConfig | None = None
    prepare_messages: Callable[[dict[str, Any]], list[Any]] | None = None
    environment: Any | None = None
    environment_factory: Callable[[dict[str, Any]], Any] | None = None
    attempt_executor: AttemptExecutor | None = None
    stop_handler: EvalStopHandler | None = None
    handle_no_tool: Callable[[AgentState, AgentRunConfig], Awaitable[AgentState]] | None = None

    def __post_init__(self) -> None:
        if self.environment is not None and self.environment_factory is not None:
            raise ValueError("AgentRunSpec cannot define both environment and environment_factory")
        if self.prepare_messages is None and self.attempt_executor is None:
            raise ValueError("AgentRunSpec requires either prepare_messages or attempt_executor")


@dataclass(frozen=True)
class EndpointConfig:
    """LLM endpoint configuration.

    Supports both API providers (Anthropic, OpenAI, Google) and
    self-hosted inference servers (SGLang, vLLM).

    For API providers:
        - API key is read from environment (ANTHROPIC_API_KEY, etc.)
        - base_url defaults to the provider's public API

    For SGLang/vLLM:
        - If base_url is provided, connects to existing server
        - If base_url is None and HardwareConfig is provided, provisions and launches server
    """

    # Provider: "anthropic", "openai", "google", "sglang", "vllm"
    provider: Literal["anthropic", "openai", "google", "sglang", "vllm"] = "anthropic"

    # Model name (provider-specific format)
    # For API: "claude-sonnet-4-20250514", "gpt-4o", etc.
    # For SGLang/vLLM: HuggingFace model ID like "Qwen/Qwen2.5-7B-Instruct"
    model: str = "claude-sonnet-4-20250514"

    # Base URL (optional - derived from provider if not set)
    # For SGLang: "http://localhost:30000/v1"
    base_url: str | None = None

    # API key (optional - read from env if not set)
    api_key: str | None = None

    # Generation parameters
    temperature: float = 0.0
    max_tokens: int = 4096

    # Extended thinking (Anthropic only)
    thinking: bool = False
    thinking_budget: int | None = None

    # Reasoning effort (OpenAI o1/o3 only)
    reasoning_effort: Literal["low", "medium", "high"] | None = None

    def get_base_url(self) -> str:
        """Get base URL, using provider defaults if not set."""
        if self.base_url:
            return self.base_url

        defaults = {
            "anthropic": "https://api.anthropic.com/v1",
            "openai": "https://api.openai.com/v1",
            "google": "https://generativelanguage.googleapis.com/v1beta",
            "sglang": "http://localhost:30000/v1",
            "vllm": "http://localhost:8000/v1",
        }
        return defaults.get(self.provider, "http://localhost:30000/v1")

    def get_api_format(self) -> str:
        """Get wire protocol format for this provider."""
        formats = {
            "anthropic": "anthropic-messages",
            "openai": "openai-completions",
            "google": "google-generative-ai",
            "sglang": "openai-completions",
            "vllm": "openai-completions",
        }
        return formats.get(self.provider, "openai-completions")

    @property
    def requires_server(self) -> bool:
        """True if this endpoint needs an inference server to be launched."""
        return self.provider in ("sglang", "vllm") and self.base_url is None


@dataclass(frozen=True)
class EvalRunConfig:
    """Evaluation execution settings."""

    # Concurrency
    max_concurrent: int = 1  # Parallel samples
    max_api_concurrent: int | None = None  # Parallel API calls (None = no limit)
    max_tool_concurrent: int | None = None  # Parallel tool executions

    # Limits
    max_samples: int | None = None  # Limit dataset size
    max_turns: int = 10  # Legacy default; translated into MaxTurnsStop when stop_handler is unset
    stop_handler: EvalStopHandler | None = None

    # Display
    verbose: bool = True
    show_progress: bool = True
    stream_tokens: bool = False  # Stream tokens to stdout

    # Retry
    max_sample_retries: int = 2  # Retry failed samples

    def resolved_stop_handler(self) -> EvalStopHandler:
        return self.stop_handler or MaxTurnsStop(self.max_turns)


@dataclass(frozen=True)
class EvalOutputConfig:
    """Evaluation output settings."""

    experiment_name: str = "eval"
    output_dir: Path | None = None  # Auto-generated if None

    # What to save
    save_report: bool = True
    save_samples: bool = True
    save_trajectories: bool = True


@dataclass(frozen=True)
class InferenceServerConfig:
    """SGLang/vLLM server settings (when auto-provisioning)."""

    port: int = 30000
    mem_fraction: float = 0.9
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"

    # Server startup
    startup_timeout: int = 300  # seconds
    health_check_interval: int = 5  # seconds
