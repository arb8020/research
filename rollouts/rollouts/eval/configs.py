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

import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import httpx
import trio

if TYPE_CHECKING:
    from rollouts.agents.types import AgentState
    from rollouts.agents.types import RunConfig as AgentRunConfig
    from rollouts.core import Endpoint
    from rollouts.eval.external_attempts import ExternalRuntime
    from rollouts.training.types import RowAttempt, Scorer

# Reuse HardwareConfig from training
from rollouts.core.eval import DistributionPercentileSpec
from rollouts.training.configs import (
    HardwareConfig,
    InferenceRole,
    InferenceWorkerConfig,
    WorkerTopologyConfig,
)

__all__ = [
    "AgentRunSpec",
    "AttemptExecutor",
    "EvalTaskSpec",
    "EndpointCapabilities",
    "EndpointConfig",
    "EvalRunConfig",
    "EvalOutputConfig",
    "ExternalEndpoint",
    "HardwareConfig",
    "InferenceEndpoint",
    "OwnedEndpoint",
    "SupportedStopHandler",
    "EvalStopHandler",
    "MaxTurnsStop",
    "TokenBudgetStop",
    "CostBudgetStop",
    "WallClockStop",
    "endpoint_config_from_inference_worker",
    "server_config_from_inference_worker",
    "endpoint_and_server_for_role",
    "materialize_endpoint",
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
    "RowAttempt | Awaitable[RowAttempt]",
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

    endpoint: InferenceEndpoint | EndpointConfig | None = None
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


# ---------------------------------------------------------------------------
# MIGRATION: Toward explicit lifecycle ownership
#
# The current EndpointConfig encodes a semantic distinction (do we own this
# server's lifecycle?) implicitly via the presence/absence of base_url. That's
# dishonest - the type looks the same whether we own the process or not.
#
# Target shape (see design discussion in docs/design/external_agent_observability_cleanup.md):
#
#   ExternalEndpoint - we don't own it, just a URL + wire format
#   OwnedEndpoint    - we launch it, we kill it, we set its devices
#
#   InferenceEndpoint = ExternalEndpoint | OwnedEndpoint
#
# Both produce a base_url the eval harness can send requests to. The distinction
# is at construction time (which type you use), not inferred from field presence.
#
# Migration plan:
#   1. Introduce ExternalEndpoint + OwnedEndpoint as new types (below)
#   2. Update endpoint_realization.py to dispatch on the sum type
#   3. Update eval/run.py and eval/native.py to accept InferenceEndpoint
#   4. Migrate callsites (32 files) from EndpointConfig → ExternalEndpoint
#   5. Deprecate and remove EndpointConfig + InferenceServerConfig
#
# OwnedEndpoint also unifies eval and RL: RL currently owns inference server
# lifecycle via SGLangEngine/VLLMEngine/EngineV2Engine, which duplicate the
# same tmux+health-poll machinery. OwnedEndpoint extracts that into one place
# and makes it reachable from eval without going through Modal.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExternalEndpoint:
    """An inference endpoint we do NOT own.

    Use this when pointing at an already-running server (local or remote)
    or an API provider. We just have a URL - no lifecycle management.

    Examples:
        # Local server started manually
        ExternalEndpoint(
            url="http://localhost:30000/v1",
            model="Qwen/Qwen2.5-7B-Instruct",
            provider="sglang",
        )

        # API provider
        ExternalEndpoint(
            url="https://api.anthropic.com/v1",
            model="claude-sonnet-4-20250514",
            provider="anthropic",
        )
    """

    url: str
    model: str
    provider: Literal["anthropic", "openai", "google", "sglang", "vllm", "openrouter"]
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    thinking: bool = False
    thinking_budget: int | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    extra_params: dict[str, Any] | None = None

    @property
    def base_url(self) -> str:
        return self.url

    @property
    def requires_server(self) -> bool:
        return False

    def get_api_format(self) -> str:
        formats = {
            "anthropic": "anthropic-messages",
            "openai": "openai-completions",
            "google": "google-generative-ai",
            "sglang": "openai-completions",
            "vllm": "openai-completions",
            "openrouter": "openai-completions",
        }
        return formats.get(self.provider, "openai-completions")


@dataclass(frozen=True)
class EndpointCapabilities:
    """What an inference endpoint advertises it can do.

    Advertised at construction time from the engine spec - not discovered
    at runtime. Training checks weight_sync before attempting NCCL setup.
    """

    # None = inference-only (eval). Present = can receive weight updates (RL).
    weight_sync: str | None = (
        None  # name of InferenceSyncRealization, e.g. "sglang-http-path-reload"
    )


@dataclass(frozen=True)
class OwnedEndpoint:
    """An inference endpoint whose process lifecycle we own.

    We launch it (tmux, survives parent crash), wait for /health, and kill
    it on shutdown. The eval harness and RL training loop both use this -
    the only difference is whether capabilities.weight_sync is set.

    Device assignment is explicit here (cuda_device_ids → CUDA_VISIBLE_DEVICES
    at launch), not reconstructed per-engine-class as it is today.

    Examples:
        # Eval - no weight sync
        OwnedEndpoint(
            spec="slime-sglang",
            model="Qwen/Qwen2.5-7B-Instruct",
            cuda_device_ids=(0,),
            port=30000,
            mem_fraction=0.9,
            capabilities=EndpointCapabilities(weight_sync=None),
        )

        # RL - with weight sync (produced by InferenceConfig internally)
        OwnedEndpoint(
            spec="slime-sglang",
            model="Qwen/Qwen2.5-7B-Instruct",
            cuda_device_ids=(0,),
            port=30000,
            mem_fraction=0.55,
            capabilities=EndpointCapabilities(weight_sync="sglang-http-path-reload"),
        )

    TODO: implement launch/wait_until_ready/shutdown here by extracting the
    shared tmux+health-poll machinery from SGLangEngine/VLLMEngine/EngineV2Engine
    in training/weight_sync.py (all three are copy-pastes of the same logic).
    Those classes should then compose on OwnedEndpoint rather than reimplement it.
    """

    spec: str  # name of InferenceEngineSpec - determines launch_cmd
    model: str
    cuda_device_ids: tuple[int, ...]
    port: int
    capabilities: EndpointCapabilities
    output_dir: Path | None = None
    # Exactly one of launch_cmd or launch_module must be set.
    # launch_cmd: full shell command, local paths only (use for local runs).
    # launch_module: Python module path, e.g. "rollouts.inference.gold_server".
    #   Runs as `python -m <launch_module> --model ... --port ...`.
    #   Works locally and remotely (module path is repo-relative, survives bifrost sync).
    launch_cmd: str | None = None
    launch_module: str | None = None
    # Extra CLI args appended to the launch_module command, e.g. ("--trace-path", "/tmp/trace.jsonl").
    # Ignored when launch_cmd is set (caller owns the full command string in that case).
    extra_launch_args: tuple[str, ...] = ()
    mem_fraction: float = 0.7
    startup_timeout: float = 300.0
    temperature: float = 0.0
    max_tokens: int = 4096
    extra_params: dict[str, Any] | None = None
    readiness_path: str = "/health"
    # Per-request timeout in seconds. Naive HF inference (gold_server.py) on
    # large MoE models can be slow; increase this if requests time out.
    request_timeout: float = 120.0

    @property
    def provider(self) -> Literal["sglang", "vllm"]:
        from rollouts.training.inference_realizations import get_inference_engine_spec

        provider = get_inference_engine_spec(self.spec).api_format
        if provider not in ("sglang", "vllm"):
            raise ValueError(f"OwnedEndpoint does not support provider {provider!r}")
        return provider

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"

    @property
    def health_url(self) -> str:
        return f"http://localhost:{self.port}{self.readiness_path}"

    @property
    def api_base(self) -> str:
        return f"http://localhost:{self.port}"

    @property
    def requires_server(self) -> bool:
        return True

    @property
    def session_name(self) -> str:
        if self.output_dir is None:
            raise ValueError("OwnedEndpoint.session_name requires output_dir")
        return f"{self.spec}-{self.output_dir.name}-{self.port}"

    @property
    def log_path(self) -> Path:
        if self.output_dir is None:
            raise ValueError("OwnedEndpoint.log_path requires output_dir")
        return self.output_dir / f"{self.provider}_{self.port}.log"

    @property
    def trace_path(self) -> Path | None:
        if self.output_dir is None or self.provider != "sglang":
            return None
        return self.output_dir / f"{self.provider}_{self.port}_trace.jsonl"

    def build_launch_cmd(self) -> str:
        if self.launch_cmd is not None:
            return self.launch_cmd
        if self.launch_module is not None:
            base = f"python -m {self.launch_module} --model {self.model} --port {self.port}"
            if self.extra_launch_args:
                base += " " + " ".join(self.extra_launch_args)
            return base
        raise ValueError("OwnedEndpoint requires either launch_cmd or launch_module")

    def get_api_format(self) -> str:
        return "openai-completions"

    def launch(self) -> str:
        return _launch_in_tmux(
            self.session_name,
            self.build_launch_cmd(),
            self.log_path,
            trace_file=self.trace_path,
            port=self.port,
            cuda_device_ids=self.cuda_device_ids,
        )

    async def wait_until_ready(self, max_wait: float | None = None) -> None:
        await _poll_until_healthy(
            self.health_url,
            self.session_name,
            self.log_path,
            self.startup_timeout if max_wait is None else max_wait,
        )

    def shutdown(self) -> None:
        _kill_tmux_session(self.session_name)

    def start_log_tailer(self) -> None:
        return None


# Sum type: the two honest variants. Use this as the type annotation wherever
# code currently accepts EndpointConfig and needs to handle both cases.
InferenceEndpoint = ExternalEndpoint | OwnedEndpoint


def _launch_in_tmux(
    session_name: str,
    cmd: str,
    log_file: Path,
    *,
    trace_file: Path | None,
    port: int,
    cuda_device_ids: tuple[int, ...],
) -> str:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.touch(exist_ok=True)
    if trace_file is not None:
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        trace_file.touch(exist_ok=True)

    subprocess.run(
        ["tmux", "kill-session", "-t", session_name],
        capture_output=True,
    )
    subprocess.run(
        f"fuser -k {port}/tcp 2>/dev/null || true",
        shell=True,
        capture_output=True,
    )
    for gpu_id in cuda_device_ids:
        subprocess.run(
            f"nvidia-smi --id={gpu_id} --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9",
            shell=True,
            capture_output=True,
        )

    full_cmd = f"{cmd} 2>&1 | tee {log_file}"
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_name, full_cmd],
        check=True,
    )
    return session_name


def _is_tmux_session_alive(session_name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        capture_output=True,
    )
    return result.returncode == 0


async def _poll_until_healthy(
    health_url: str,
    session_name: str,
    log_file: Path,
    max_wait: float,
) -> None:
    async with httpx.AsyncClient(timeout=5.0) as client:
        for _attempt in range(int(max_wait)):
            if not _is_tmux_session_alive(session_name):
                raise RuntimeError(
                    "Inference server crashed during startup. "
                    f"session_name={session_name!r} log_path={log_file}"
                )
            try:
                resp = await client.get(health_url)
                if resp.status_code == 200:
                    return
            except Exception:
                pass
            await trio.sleep(1.0)

    raise RuntimeError(
        f"Inference server failed to become healthy within {max_wait}s. log_path={log_file}"
    )


def _kill_tmux_session(session_name: str) -> None:
    subprocess.run(
        ["tmux", "kill-session", "-t", session_name],
        capture_output=True,
    )


@dataclass(frozen=True)
class EndpointConfig:
    """LLM endpoint configuration.

    DEPRECATED: Use ExternalEndpoint or OwnedEndpoint instead.

    This type conflates two distinct cases:
    - External endpoint (base_url set): we don't own the server
    - Owned endpoint (base_url None): we launch and manage the server
    That distinction should be explicit in the type, not inferred from
    field presence. See migration comment above.

    Supports both API providers (Anthropic, OpenAI, Google) and
    self-hosted inference servers (SGLang, vLLM).
    """

    provider: Literal["anthropic", "openai", "google", "sglang", "vllm", "openrouter"] = "anthropic"
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


def endpoint_config_from_inference_worker(worker: InferenceWorkerConfig) -> EndpointConfig:
    """Lower a named inference worker into the current eval endpoint surface.

    This preserves the current runner contract while we migrate evals toward the
    shared worker-topology model used by RL.
    """

    return EndpointConfig(
        provider=worker.resolved_provider,
        model=worker.model,
        base_url=worker.base_url,
        temperature=0.0,
    )


def server_config_from_inference_worker(worker: InferenceWorkerConfig) -> InferenceServerConfig:
    """Lower a named inference worker into the current local-server settings."""

    return InferenceServerConfig(
        port=worker.inference.port,
        mem_fraction=worker.inference.mem_fraction,
        tensor_parallel_size=worker.inference.tensor_parallel_size,
        startup_timeout=int(worker.inference.startup_timeout),
    )


def endpoint_and_server_for_role(
    topology: WorkerTopologyConfig,
    role: InferenceRole,
) -> tuple[EndpointConfig, InferenceServerConfig]:
    # TODO(worker-topology): today's eval runner still auto-realizes one local
    # inference service. This helper is honest only for the role whose worker is
    # being launched there. Additional judge/reference workers need either:
    # - pre-launched base_url endpoints, or
    # - a multi-worker launch layer above eval.run.
    worker = topology.get_worker_for_role(role)
    return (
        endpoint_config_from_inference_worker(worker),
        server_config_from_inference_worker(worker),
    )


def materialize_endpoint(endpoint_config: InferenceEndpoint | EndpointConfig) -> Endpoint:
    """Lower an eval endpoint surface into the core Endpoint type.

    This is the shared endpoint contract for:
    - native eval execution
    - scorer-owned LLM judge calls
    - future multi-role worker bindings

    Keep the provider/model normalization here so examples and scorers do not
    re-derive API base URLs, wire formats, or credential lookup ad hoc.
    """
    from difflib import get_close_matches
    from typing import cast

    from rollouts.core import Endpoint
    from rollouts.credentials import get_api_key
    from rollouts.fuzzy import fuzzy_filter
    from rollouts.models import MODELS, Provider, get_model

    provider = endpoint_config.provider
    model = endpoint_config.model

    if isinstance(endpoint_config, ExternalEndpoint):
        configured_base_url = endpoint_config.url
        api_key = endpoint_config.api_key or get_api_key(provider) or ""
    elif isinstance(endpoint_config, OwnedEndpoint):
        configured_base_url = endpoint_config.base_url
        api_key = get_api_key(provider) or ""
    else:
        configured_base_url = endpoint_config.base_url
        api_key = endpoint_config.api_key or get_api_key(provider) or ""
    if not api_key and provider in ("anthropic", "openai", "google", "openrouter"):
        raise ValueError(
            f"No API key found for {provider}. Set {provider.upper()}_API_KEY in environment."
        )

    resolved_base_url: str | None = None
    resolved_api_format: str | None = None
    if provider in MODELS:
        provider_models = MODELS[cast("Provider", provider)]
        metadata = get_model(cast("Provider", provider), model)
        if metadata is not None:
            resolved_base_url = metadata.base_url
            resolved_api_format = metadata.api
        elif provider_models:
            model_ids = list(provider_models.keys())
            suggestions = fuzzy_filter(model_ids, model, lambda x: x)[:3]
            if not suggestions:
                suggestions = get_close_matches(model, model_ids, n=3, cutoff=0.5)
            error_msg = f"Model '{model}' not found for provider '{provider}'."
            if suggestions:
                error_msg += "\n\nDid you mean one of these?\n"
                for suggestion in suggestions:
                    error_msg += f"  - {provider}/{suggestion}\n"
            error_msg += f"\nSee available models: rollouts --list-models {provider}"
            raise ValueError(error_msg)

    return Endpoint(
        model=f"{provider}/{model}",
        base_url=configured_base_url or resolved_base_url or endpoint_config.get_base_url(),
        api_format=resolved_api_format or endpoint_config.get_api_format(),
        api_key=api_key,
        temperature=endpoint_config.temperature,
        max_tokens=endpoint_config.max_tokens,
        reasoning_effort=getattr(endpoint_config, "reasoning_effort", None),
        extra_params=getattr(endpoint_config, "extra_params", None),
    )


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
    summary_distribution_percentiles: DistributionPercentileSpec | None = None

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
        return replace(eval_task, run_spec=resolve_eval_run_spec(config_module))

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
