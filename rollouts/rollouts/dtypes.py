import json
from pathlib import Path
from enum import Enum
import os
import trio
from abc import ABC
from dataclasses import dataclass, field, asdict, fields, replace
from typing import Any, Dict, List, Optional, Mapping, Union, TypeVar, Type, Iterator, Callable, Awaitable, Tuple, Protocol, runtime_checkable
from pathlib import Path
from datetime import datetime, timezone
import dacite

# TODO: Better torch typing options explored:
# 1. Create a Protocol for tensor-like objects (has .tolist(), .shape, .dtype) - cleanest approach
# 2. Use torch-stubs package if available for lightweight type info
# 3. Define proper Union types for tensor alternatives
# 4. Previous approach used TYPE_CHECKING conditional imports but created dependency issues
# 
# Current: Simple fallback for type hints - actual tensor handling is done at runtime via hasattr checks
TorchTensor = Any

# Verbose function for debugging
def verbose(level=1):
    """Check if verbose logging is enabled at given level"""
    return int(os.getenv("VERBOSE", 0)) >= level

class JsonSerializable:
    """Base class for dataclasses with JSON serialization support.

    Tiger Style: Pure serialization, no I/O side effects.
    Caller controls where the JSON goes (file, network, memory, etc.).
    """

    def to_json(self) -> str:
        """Serialize to JSON string"""
        assert self is not None
        result = json.dumps(asdict(self), ensure_ascii=False) #type:ignore
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        return result

    @classmethod
    def from_json(cls, json_str: str):
        """Deserialize from JSON string using dacite"""
        assert json_str is not None
        assert isinstance(json_str, str)
        assert len(json_str) > 0
        data = json.loads(json_str)
        assert data is not None
        assert isinstance(data, dict)
        result = dacite.from_dict(data_class=cls, data=data)
        assert result is not None
        return result

@dataclass(frozen=True)
class ToolCall(JsonSerializable):
    id: str
    name: str
    args: Mapping[str, Any]

@dataclass(frozen=True)
class StreamChunk(JsonSerializable):
    """A chunk of data emitted during streaming"""
    kind: str  # "token", "tool_call_complete", "tool_result", etc.
    data: Mapping[str, Any]

@dataclass(frozen=True)
class Message(JsonSerializable):
    role: str
    content: Optional[str | List[Dict[str, Any]]]  # str for text, List[Dict] for vision (OpenAI format)
    reasoning_content: Optional[Any] = None
    thinking_content: Optional[str] = None
    thinking_signature: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None

    def __repr__(self) -> str:
        """Tiger Style: Bounded repr, truncate large content.

        Vision messages can contain base64 images (100KB+).
        Always truncate to prevent terminal spam.
        """
        # Truncate content for display
        if isinstance(self.content, str):
            content_preview = self.content[:100] + "..." if len(self.content) > 100 else self.content
        elif isinstance(self.content, list):
            # Vision message - show structure but not base64 data
            content_preview = f"[vision message with {len(self.content)} parts]"
        else:
            content_preview = str(self.content)

        return f"Message(role={self.role!r}, content={content_preview!r})"

@dataclass(frozen=True)
class Usage(JsonSerializable):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[Any] = None

@dataclass(frozen=True)
class Logprob(JsonSerializable):
    token: str
    logprob: float
    bytes: List[int]
    top_logprobs: List[float]

@dataclass(frozen=True)
class Logprobs(JsonSerializable):
    content: List[Logprob] = field(default_factory=list)

@dataclass(frozen=True)
class Choice(JsonSerializable):
    index: int
    message: Message
    finish_reason: str
    logprobs: Optional[Logprobs] = None
    stop_reason: Optional[Any] = None


@dataclass(frozen=True)
class TokenInfo(JsonSerializable):
    logprob: float
    rank: int
    decoded_token: str

PromptLogprob = Optional[Dict[str, TokenInfo]]
"""
{
"8948": { # key is different every token
"logprob": -12.845086097717285,
"rank": 60822,
"decoded_token": "system"
}
}
"""

@dataclass(frozen=True)
class ChatCompletion(JsonSerializable):
    id: str
    object: str
    created: int
    model: str
    usage: Usage
    kv_transfer_params: Optional[Any] = None
    choices: List[Choice] = field(default_factory=list)
    prompt_logprobs: Optional[List[PromptLogprob]] = None

@dataclass
class Trajectory(JsonSerializable):
    completions: List[ChatCompletion] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)   # debugging only
    rewards: float = 0.0
    group: int = 0
    replica: int = 0
    advantages: float = 0.0     # scalar; broadcast later if needed
    metadata: Dict[str, Any] = field(default_factory=dict)  # For dataset-specific info (e.g., ground truth)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Trajectory":
        """Rebuild nested dataclasses so type hints stay correct."""
        assert data is not None
        assert isinstance(data, dict)

        comps: List[ChatCompletion] = []
        for comp in data.get("completions", []):
            assert comp is not None
            assert isinstance(comp, dict)
            usage_dict = comp.get("usage", {})
            assert "prompt_tokens" in usage_dict
            assert "completion_tokens" in usage_dict
            assert "total_tokens" in usage_dict
            usage = Usage(
                prompt_tokens       = usage_dict["prompt_tokens"],
                completion_tokens   = usage_dict["completion_tokens"],
                total_tokens        = usage_dict["total_tokens"],
            )
            assert usage is not None
            # Construct ChatCompletion with explicit parameters for type safety
            comps.append(ChatCompletion(
                id=comp.get("id", "unknown"),
                object=comp.get("object", "chat.completion"),
                created=comp.get("created", 0),
                model=comp.get("model", "unknown"),
                usage=usage,
                kv_transfer_params=comp.get("kv_transfer_params"),
                choices=comp.get("choices", []),
                prompt_logprobs=comp.get("prompt_logprobs")
            ))

        result = Trajectory(
            completions=comps,
            messages=data.get("messages", []),
            rewards=data.get("rewards", 0.0),
            group=data.get("group", 0),
            replica=data.get("replica", 0),
            advantages=data.get("advantages", 0.0),
        )
        assert result is not None
        return result

    # ---------- JSONL convenience layer -----------------------------------
    def to_json(self) -> str:
        assert self is not None
        result = json.dumps(asdict(self), ensure_ascii=False)
        assert result is not None
        assert isinstance(result, str)
        return result

    @staticmethod
    def to_jsonl(trajectories: List["Trajectory"]) -> str:
        assert trajectories is not None
        assert isinstance(trajectories, list)
        result = "\n".join(t.to_json() for t in trajectories)
        assert isinstance(result, str)
        return result

    @staticmethod
    def from_json(json_str: str) -> "Trajectory":
        assert json_str is not None
        assert isinstance(json_str, str)
        assert len(json_str) > 0
        data = json.loads(json_str)
        result = Trajectory.from_dict(data)
        assert result is not None
        return result

    @staticmethod
    def from_jsonl(jsonl_str: str) -> List["Trajectory"]:
        assert jsonl_str is not None
        assert isinstance(jsonl_str, str)
        result = [Trajectory.from_json(line) for line in jsonl_str.strip().splitlines() if line]
        assert isinstance(result, list)
        return result

    # ---------- disk I/O ---------------------------------------------------
    @staticmethod
    def save_jsonl(trajectories: List["Trajectory"], filepath: str) -> None:
        assert trajectories is not None
        assert isinstance(trajectories, list)
        assert filepath is not None
        assert len(filepath) > 0
        jsonl_content = Trajectory.to_jsonl(trajectories)
        assert jsonl_content is not None
        path_obj = Path(filepath)
        path_obj.write_text(jsonl_content, encoding="utf-8")
        assert path_obj.exists()

    @staticmethod
    def load_jsonl(filepath: str) -> List["Trajectory"]:
        assert filepath is not None
        assert len(filepath) > 0
        path_obj = Path(filepath)
        assert path_obj.exists(), f"File not found: {filepath}"
        assert path_obj.is_file()
        content = path_obj.read_text(encoding="utf-8")
        result = Trajectory.from_jsonl(content)
        assert result is not None
        assert isinstance(result, list)
        return result

    @staticmethod
    def load_jsonl_streaming(filepath: str) -> Iterator["Trajectory"]:
        assert filepath is not None
        assert len(filepath) > 0
        path_obj = Path(filepath)
        assert path_obj.exists(), f"File not found: {filepath}"
        assert path_obj.is_file()

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line_stripped = line.strip()
                if line_stripped:  # Skip empty lines
                    yield Trajectory.from_json(line_stripped)

    # ---------- helpers that work pre-/post-serialisation ------------------
    @staticmethod
    def _usage_total(usage: Union[Usage, Dict[str, Any]], key: str) -> int:
        assert usage is not None
        assert key is not None
        assert isinstance(key, str)
        if isinstance(usage, Usage):
            result = getattr(usage, key, 0)
        else:
            result = usage.get(key, 0)
        assert isinstance(result, int)
        assert result >= 0
        return result

    @staticmethod
    def get_completion_tokens(traj: "Trajectory") -> int:
        assert traj is not None
        assert isinstance(traj, Trajectory)
        result = sum(Trajectory._usage_total(c.usage, "completion_tokens") for c in traj.completions)
        assert result >= 0
        return result

    @staticmethod
    def get_total_tokens(traj: "Trajectory") -> int:
        assert traj is not None
        assert isinstance(traj, Trajectory)
        result = sum(Trajectory._usage_total(c.usage, "total_tokens") for c in traj.completions[-1:])
        assert result >= 0
        return result

    @staticmethod
    def hash(trajectory: "Trajectory") -> str:
        """Generate a hash for a single trajectory."""
        import hashlib
        assert trajectory is not None
        assert isinstance(trajectory, Trajectory)
        traj_dict = asdict(trajectory)
        assert traj_dict is not None
        traj_str = json.dumps(traj_dict, sort_keys=True)
        assert traj_str is not None
        result = hashlib.sha256(traj_str.encode()).hexdigest()[:16]
        assert result is not None
        assert isinstance(result, str)
        assert len(result) == 16
        return result

@dataclass(frozen=True)
class ToolFunctionParameter(JsonSerializable):
    properties: Dict[str, Any]
    type: str = "object"

@dataclass(frozen=True)
class ToolFunction(JsonSerializable):
    name: str
    description: str
    parameters: ToolFunctionParameter
    required: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class Tool(JsonSerializable):
    function: ToolFunction
    type: str = "function"

class StopReason(Enum):
    MAX_TURNS = "MAX_TURNS"
    TOOL_ERROR = "TOOL_ERROR"
    USER_ABORT = "USER_ABORT"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    NO_TOOL_CALLED = "NO_TOOL_CALLED"
    TASK_COMPLETED = "TASK_COMPLETED"

@dataclass(frozen=True)
class ToolResult(JsonSerializable):
    call_id: str = ""
    ok: bool = False
    content: str = ""
    error: Optional[str] = None
    stop_reason: Optional[StopReason] = None

@dataclass(frozen=True)
class ToolConfirmResult(JsonSerializable):
    """Result of tool confirmation"""
    proceed: bool
    tool_result: Optional[ToolResult] = None
    user_message: Optional[str] = None

# ── Core Agent Framework Types ────────────────────────────────────────────────

@runtime_checkable
class Environment(Protocol):
    """Protocol that all environments must satisfy for composition over inheritance."""

    def get_tools(self) -> List[Tool]:
        """Return available tools for this environment."""
        ...

    async def exec_tool(self, tool_call: ToolCall, current_state: 'AgentState',
                       run_config: 'RunConfig', checkpoint_store = None) -> ToolResult:
        """Execute a tool call in this environment."""
        ...

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        """Check if tool requires confirmation."""
        ...

    async def on_assistant_message(self, message: 'Message', state: 'AgentState') -> 'AgentState':
        """Called after each assistant message, before tool processing.

        Allows environment to respond to assistant messages with feedback, regardless
        of whether tools were called. Useful for message-based environments that need
        to execute code, provide feedback, or inject responses.

        Args:
            message: The assistant's message (may contain tool calls)
            state: Current agent state

        Returns:
            Updated agent state with environment feedback injected into trajectory.
            Return unchanged state for no response.

        Example (backend-bench):
            Parse code from message, execute it, inject feedback:
            ```python
            async def on_assistant_message(self, message, state):
                code = self.parser.parse([{"role": "assistant", "content": message.content}])
                result = await self.code_evaluator(code)
                feedback_msg = Message(role="user", content=result.feedback)
                # Inject feedback into trajectory
                return replace(state, actor=replace(state.actor, trajectory=replace(
                    state.actor.trajectory,
                    messages=[*state.actor.trajectory.messages, feedback_msg]
                )))
            ```
        """
        ...

    async def serialize(self) -> dict:
        """Serialize environment state to dictionary."""
        ...

    @staticmethod
    async def deserialize(data: dict) -> 'Environment':
        """Deserialize environment from dictionary."""
        ...

@dataclass(frozen=True)
class Endpoint(JsonSerializable):
    provider: str
    model: str
    api_base: str = ""
    api_key: str = ""
    max_tokens: int = 8192
    temperature: float = 1.0
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    reasoning_effort: Optional[str] = None  # for openai
    max_completion_tokens: Optional[int] = None  # for openai
    thinking: Optional[Dict[str, Any]] = None # for anthropic
    # Retry configuration
    max_retries: int = 3  # Number of retries for rate limits/transient errors
    timeout: float = 120.0  # Timeout in seconds for API calls
    # Extra params merged into the raw chat request for custom servers
    extra_params: Optional[Dict[str, Any]] = None

@dataclass(frozen=True)
class Actor(JsonSerializable):
    trajectory: Trajectory
    endpoint: Endpoint
    tools: List[Tool] = field(default_factory=list)

@dataclass(frozen=True)
class AgentState:
    actor: Actor
    environment: Environment | None
    max_turns: int
    stop: Optional[StopReason] = None
    turn_idx: int = 0
    pending_tool_calls: List[ToolCall] = field(default_factory=list)
    next_tool_idx: int = 0  # Which tool we're about to process
    timestamp: str = datetime.now(timezone.utc).isoformat() + 'Z'

# Forward declarations for RunConfig (needs to be after AgentState but before default handlers)
async def default_stdin_handler(prompt: str) -> str:
    """Default input handler using trio.to_thread.run_sync for non-blocking input."""
    return await trio.to_thread.run_sync(input, prompt)

async def default_confirm_tool(tc: ToolCall, state: 'AgentState', run_config: 'RunConfig') -> Tuple['AgentState', ToolConfirmResult]:
    """Default tool confirmation handler - auto-confirm all tools."""
    return state, ToolConfirmResult(proceed=True)

async def default_no_tool_handler(state: 'AgentState', run_config: 'RunConfig') -> 'AgentState':
    """Default no-tool handler - do nothing."""
    return state

@dataclass(frozen=True)
class RunConfig:
    # TODO: Add runtime validation for on_chunk parameter to catch sync functions early
    # Currently if a sync function is passed, it gets set to None silently, causing
    # "object NoneType can't be used in 'await' expression" errors later. Should validate
    # that on_chunk is properly async and has correct signature at construction time.
    on_chunk: Callable[[StreamChunk], Awaitable[None]]
    on_input: Callable[[str], Awaitable[str]] = field(default_factory=lambda: default_stdin_handler)
    confirm_tool: Callable[[ToolCall, 'AgentState', 'RunConfig'], Awaitable[Tuple['AgentState', ToolConfirmResult]]] = field(default_factory=lambda: default_confirm_tool)
    handle_tool_error: Callable[[ToolResult, 'AgentState'], 'AgentState'] = lambda tr, s: s
    on_step_start: Callable[['AgentState'], 'AgentState'] = lambda s: s
    handle_stop: Callable[['AgentState'], 'AgentState'] = lambda s: s
    handle_no_tool: Callable[['AgentState', 'RunConfig'], Awaitable['AgentState']] = field(default_factory=lambda: default_no_tool_handler)
    user_message_for_thinking: Optional[str] = None
    inline_thinking: Optional[str] = None
    checkpoint_store: Optional[Any] = None
    show_progress: bool = False  # Enable turn-level progress tracking

# ── Evaluation Types ──────────────────────────────────────────────────────────

# Reward function: pure transform from Trajectory -> Trajectory with rewards populated
# Supports both sync and async (for integrations like Prime that need async scoring)
RewardFunction = Callable[[Trajectory], Trajectory] | Callable[[Trajectory], Awaitable[Trajectory]]

@dataclass(frozen=True)
class EvalConfig:
    """Configuration for evaluation runs.

    Tiger Style: All configuration explicit, immutable, composable.

    Example:
        >>> def my_reward(traj: Trajectory) -> Trajectory:
        ...     score = 1.0 if check_correctness(traj) else 0.0
        ...     return replace(traj, rewards=score)
        >>>
        >>> config = EvalConfig(
        ...     reward_fn=my_reward,
        ...     max_turns=10,
        ...     max_concurrent=4,
        ... )
    """
    # Required: how to compute rewards
    reward_fn: RewardFunction

    # Agent execution
    run_config: Optional[RunConfig] = None  # If None, use silent default
    max_turns: int = 10

    # Dataset control
    max_samples: Optional[int] = None  # If None, evaluate all
    sample_id_fn: Callable[[int, Dict[str, Any]], str] = field(
        default_factory=lambda: lambda i, _: f"sample_{i:04d}"
    )

    # Parallelization
    max_concurrent: int = 1

    # Output
    output_dir: Optional[Path] = None
    eval_name: str = "evaluation"

    # Logging
    verbose: bool = True
    show_progress: bool = False  # Enable sample-level progress tracking
    stream_tokens: bool = False  # Stream LLM tokens to stdout (used if run_config is None)
