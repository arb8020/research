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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from rollouts.agents.types import AgentState
    from rollouts.agents.types import RunConfig as AgentRunConfig
    from rollouts.eval.external_attempts import ExternalRuntime
    from rollouts.training.types import AttemptResult, Scorer

# Reuse HardwareConfig from training
from rollouts.training.configs import HardwareConfig

__all__ = [
    "AgentRunSpec",
    "AttemptExecutor",
    "EvalTaskSpec",
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
    "resolve_eval_run_spec",
    "resolve_eval_task_spec",
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
PromptBuilder = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class AgentRunSpec:
    """Per-sample agent execution spec.

    This is the eval-side product type closest to what one `run_agent(...)`
    execution needs: either an endpoint-driven agent loop or a custom
    attempt executor, plus optional environment ownership and stop/no-tool
    behavior. Runtime-specific external CLI knobs belong in
    `external_agent_args`; they are intentionally passed through without
    pretending Claude/Codex/OpenHands share one honest permission algebra.
    Dataset iteration, concurrency, retries, and output policy still belong
    to EvalRunConfig/EvalOutputConfig.
    """

    # TODO(sum-type): This dataclass currently encodes two execution branches:
    # native rollouts execution (`endpoint` + `prepare_messages`) and external/custom
    # execution (`attempt_executor`). If the surface keeps growing, split this into
    # an explicit sum type instead of adding more branch-specific optional fields.

    endpoint: EndpointConfig | None = None
    external_runtime: ExternalRuntime | None = None
    prepare_messages: Callable[[dict[str, Any]], list[Any]] | None = None
    prompt_builder: PromptBuilder | None = None
    environment: Any | None = None
    environment_factory: Callable[[dict[str, Any]], Any] | None = None
    attempt_executor: AttemptExecutor | None = None
    external_agent_args: dict[str, Any] = field(default_factory=dict)
    stop_handler: EvalStopHandler | None = None
    handle_no_tool: Callable[[AgentState, AgentRunConfig], Awaitable[AgentState]] | None = None

    def __post_init__(self) -> None:
        if self.environment is not None and self.environment_factory is not None:
            raise ValueError("AgentRunSpec cannot define both environment and environment_factory")
        if self.attempt_executor is not None and self.external_runtime is not None:
            raise ValueError(
                "AgentRunSpec cannot define both attempt_executor and external_runtime"
            )
        if self.external_runtime is not None and self.endpoint is not None:
            raise ValueError("AgentRunSpec cannot define both endpoint and external_runtime")
        if (
            self.external_runtime is not None
            and self.prompt_builder is None
            and self.prepare_messages is None
        ):
            raise ValueError(
                "AgentRunSpec external_runtime requires prompt_builder or prepare_messages"
            )
        if (
            self.prepare_messages is None
            and self.prompt_builder is None
            and self.attempt_executor is None
            and self.external_runtime is None
        ):
            raise ValueError(
                "AgentRunSpec requires prepare_messages, prompt_builder, attempt_executor, or external_runtime"
            )


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

    provider: Literal["anthropic", "openai", "google", "sglang", "vllm"] = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    thinking: bool = False
    thinking_budget: int | None = None
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

    max_concurrent: int = 1
    max_api_concurrent: int | None = None
    max_tool_concurrent: int | None = None
    max_samples: int | None = None
    max_turns: int = 10
    stop_handler: EvalStopHandler | None = None
    verbose: bool = True
    show_progress: bool = True
    stream_tokens: bool = False
    max_sample_retries: int = 2

    def resolved_stop_handler(self) -> EvalStopHandler:
        return self.stop_handler or MaxTurnsStop(self.max_turns)


@dataclass(frozen=True)
class EvalOutputConfig:
    """Evaluation output settings."""

    experiment_name: str = "eval"
    output_dir: Path | None = None
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
    startup_timeout: int = 300
    health_check_interval: int = 5


@dataclass(frozen=True)
class EvalTaskSpec:
    """Explicit eval task product type.

    This is the first honest eval authoring surface for the current runner:
    dataset source, per-sample execution spec, scoring path, run settings, and
    output policy all travel together instead of being inferred from a bag of
    top-level module exports.
    """

    run_spec: AgentRunSpec
    tasks: list[dict[str, Any]] | None = None
    tasks_path: Path | None = None
    scorer: Scorer | None = None
    run: EvalRunConfig = field(default_factory=EvalRunConfig)
    output: EvalOutputConfig = field(default_factory=EvalOutputConfig)
    hardware: HardwareConfig | None = None
    server: InferenceServerConfig = field(default_factory=InferenceServerConfig)

    def __post_init__(self) -> None:
        if (self.tasks is None) == (self.tasks_path is None):
            raise ValueError("EvalTaskSpec must define exactly one of tasks or tasks_path")
        if self.scorer is None:
            raise ValueError("EvalTaskSpec requires an explicit scorer")


def resolve_eval_run_spec(config_module: Any) -> AgentRunSpec:
    """Normalize just the per-sample execution part of an eval config."""
    from .external_attempts import make_external_attempt_executor

    def _message_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
                    continue
                thinking = getattr(block, "thinking", None)
                if isinstance(thinking, str):
                    parts.append(thinking)
                    continue
                if isinstance(block, dict):
                    if isinstance(block.get("text"), str):
                        parts.append(block["text"])
                    elif isinstance(block.get("thinking"), str):
                        parts.append(block["thinking"])
            return "\n".join(part for part in parts if part)
        return str(content) if content is not None else ""

    def _prompt_builder_from_messages(
        prepare_messages: Callable[[dict[str, Any]], list[Any]],
    ) -> PromptBuilder:
        def build_prompt(sample_data: dict[str, Any]) -> str:
            messages = prepare_messages(sample_data)
            if not isinstance(messages, list) or not messages:
                raise ValueError("prepare_messages(...) must return a non-empty message list")

            rendered: list[str] = []
            for msg in messages:
                role = getattr(msg, "role", None)
                content = getattr(msg, "content", None)
                if role is None and isinstance(msg, dict):
                    role = msg.get("role")
                    content = msg.get("content")
                if role not in {"system", "user", "developer"}:
                    continue
                text = _message_text(content).strip()
                if not text:
                    continue
                rendered.append(f"{str(role).upper()}:\n{text}")

            if not rendered:
                raise ValueError(
                    "prepare_messages(...) produced no system/user/developer text to launch"
                )
            return "\n\n".join(rendered)

        return build_prompt

    def _lower_external_runtime(run_spec: AgentRunSpec) -> AgentRunSpec:
        if run_spec.external_runtime is None:
            return run_spec
        prompt_builder = run_spec.prompt_builder
        if prompt_builder is None:
            assert run_spec.prepare_messages is not None
            prompt_builder = _prompt_builder_from_messages(run_spec.prepare_messages)
        return replace(
            run_spec,
            external_runtime=None,
            attempt_executor=make_external_attempt_executor(
                run_spec.external_runtime,
                prompt_builder=prompt_builder,
                **run_spec.external_agent_args,
            ),
        )

    eval_task = getattr(config_module, "eval_task", None)
    if eval_task is not None:
        if not isinstance(eval_task, EvalTaskSpec):
            raise ValueError("Eval config must export eval_task: EvalTaskSpec")
        return _lower_external_runtime(eval_task.run_spec)

    run_spec = getattr(config_module, "run_spec", None)
    if run_spec is not None:
        if not isinstance(run_spec, AgentRunSpec):
            raise ValueError("Eval config must export run_spec: AgentRunSpec")
        endpoint = getattr(config_module, "endpoint", None)
        if endpoint is not None and run_spec.endpoint is None:
            run_spec = replace(run_spec, endpoint=endpoint)
        return _lower_external_runtime(run_spec)

    prepare_messages = getattr(config_module, "prepare_messages", None)
    attempt_executor = getattr(config_module, "attempt_executor", None)
    if not callable(prepare_messages) and not callable(attempt_executor):
        raise ValueError(
            "Eval config must export callable prepare_messages or attempt_executor or run_spec"
        )

    environment = None
    environment_factory = None
    if hasattr(config_module, "make_environment"):
        make_env = config_module.make_environment
        if (
            hasattr(config_module, "per_sample_environment")
            and config_module.per_sample_environment
        ):
            environment_factory = make_env
        else:
            environment = make_env()

    return AgentRunSpec(
        endpoint=getattr(config_module, "endpoint", None),
        prepare_messages=prepare_messages if callable(prepare_messages) else None,
        environment=environment,
        environment_factory=environment_factory,
        attempt_executor=attempt_executor if callable(attempt_executor) else None,
    )


def resolve_eval_task_spec(config_module: Any) -> EvalTaskSpec:
    """Normalize legacy eval module exports into an explicit EvalTaskSpec."""

    eval_task = getattr(config_module, "eval_task", None)
    if eval_task is not None:
        if not isinstance(eval_task, EvalTaskSpec):
            raise ValueError("Eval config must export eval_task: EvalTaskSpec")
        return eval_task

    run_spec = resolve_eval_run_spec(config_module)

    tasks = getattr(config_module, "tasks", None)
    tasks_path = getattr(config_module, "tasks_path", None)
    if tasks is None and tasks_path is None:
        raise ValueError("Eval config must define 'tasks' or 'tasks_path'")

    return EvalTaskSpec(
        run_spec=run_spec,
        tasks=tasks,
        tasks_path=Path(tasks_path) if tasks_path is not None else None,
        scorer=getattr(config_module, "scorer", None),
        run=getattr(config_module, "run", EvalRunConfig()),
        output=getattr(config_module, "output", EvalOutputConfig()),
        hardware=getattr(config_module, "hardware", None),
        server=getattr(config_module, "server", InferenceServerConfig()),
    )
