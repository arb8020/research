import json
import os
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Literal,
    Optional,
    Protocol,
    runtime_checkable,
)

import dacite
import trio

# TODO: Better torch typing options explored:
# 1. Create a Protocol for tensor-like objects (has .tolist(), .shape, .dtype) - cleanest approach
# 2. Use torch-stubs package if available for lightweight type info
# 3. Define proper Union types for tensor alternatives
# 4. Previous approach used TYPE_CHECKING conditional imports but created dependency issues
# 
# Current: Simple fallback for type hints - actual tensor handling is done at runtime via hasattr checks
TorchTensor = Any

# TUI formatter type - receives (tool_name, args, result, expanded) and returns formatted string
ToolFormatter = Callable[[str, dict[str, Any], dict[str, Any] | None, bool], str]


# Verbose function for debugging
def verbose(level=1):
    """Check if verbose logging is enabled at given level"""
    return int(os.getenv("VERBOSE", 0)) >= level


def parse_streaming_json(partial_json: str) -> dict[str, Any]:
    """Parse partial JSON string, returning best-effort partial object.

    During streaming, tool call arguments arrive incrementally as incomplete JSON.
    This function attempts to extract valid key-value pairs from incomplete JSON.

    Examples:
        '{"foo": "bar"'          -> {"foo": "bar"}
        '{"foo": "bar", "baz":'  -> {"foo": "bar"}
        '{"nested": {"a": 1'     -> {"nested": {"a": 1}}
        '{"arr": [1, 2'          -> {"arr": [1, 2]}
        ''                       -> {}
        '{'                      -> {}

    Tiger Style: Best-effort parsing, crash-loud on programmer error.
    - Invalid UTF-8 -> crash (caller must ensure valid encoding)
    - Incomplete JSON -> return partial parsed dict (expected during streaming)
    - Malformed JSON -> return empty dict (streaming hasn't started yet)
    """
    assert isinstance(partial_json, str), f"Expected str, got {type(partial_json)}"

    if not partial_json or partial_json.strip() == "":
        return {}

    # Try parsing as complete JSON first
    try:
        result = json.loads(partial_json)
        assert isinstance(result, dict), f"Tool args must be object, got {type(result)}"
        return result
    except json.JSONDecodeError:
        pass

    # Incomplete JSON - try to extract what we can
    # Strategy: Progressively trim incomplete parts from the end
    # 1. Close incomplete string values
    # 2. Remove incomplete keys
    # 3. Close incomplete arrays
    # 4. Close incomplete objects

    cleaned = partial_json.strip()

    # Handle edge cases
    if cleaned in ("{", "[", ""):
        return {}

    # Try adding closing braces/brackets progressively
    attempts = [
        cleaned + '"}',      # Close incomplete string value
        cleaned + ']',       # Close incomplete array
        cleaned + '}',       # Close incomplete object
        cleaned + '"}]',     # Close string in array
        cleaned + '"}}'      # Close string in nested object
    ]

    # Also try removing trailing incomplete key/value
    if "," in cleaned:
        # Remove everything after the last comma (incomplete key-value pair)
        last_comma = cleaned.rfind(",")
        truncated = cleaned[:last_comma]
        attempts.extend([
            truncated + "}",
            truncated + "]}",
            truncated + "}}"
        ])

    # If there's a colon without a value, remove the incomplete pair
    if ":" in cleaned:
        # Find the last complete comma before incomplete value
        parts = cleaned.split(",")
        for i in range(len(parts) - 1, -1, -1):
            # Check if this part has both key and value
            truncated = ",".join(parts[:i])
            if truncated:
                attempts.extend([
                    truncated + "}",
                    truncated + "]}",
                    truncated + "}}"
                ])

    # Try each repair strategy
    for attempt in attempts:
        try:
            result = json.loads(attempt)
            if isinstance(result, dict):
                return result
            elif isinstance(result, list):
                # Array of objects - return last object if available
                if result and isinstance(result[-1], dict):
                    return result[-1]
        except json.JSONDecodeError:
            continue

    # All strategies failed - return empty dict
    return {}


class JsonSerializable:
    """Base class for dataclasses with JSON serialization support.

    Tiger Style: Pure serialization, no I/O side effects.
    Caller controls where the JSON goes (file, network, memory, etc.).
    """

    def to_json(self) -> str:
        """Serialize to JSON string"""
        assert self is not None
        result = json.dumps(asdict(self), ensure_ascii=False)  # type:ignore
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
    """DEPRECATED: Legacy streaming event format. Use StreamEvent types instead.

    This class is kept temporarily for backward compatibility during migration.
    Will be removed once all consumers switch to the new granular event types.
    """
    type: str  # "token", "tool_call_complete", "tool_result", etc.
    data: Mapping[str, Any]
    timestamp: float = field(default_factory=time.time)


# New granular streaming events (inspired by pi-ai)
# Each event includes content_index for tracking which content block and timestamp for logging


@dataclass(frozen=True)
class LLMCallStart(JsonSerializable):
    """Emitted before making the LLM API call (before connection established)"""
    type: Literal["llm_call_start"] = "llm_call_start"
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class StreamStart(JsonSerializable):
    """Emitted at the start of a streaming response (connection established, first event received)"""
    type: Literal["start"] = "start"
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class TextStart(JsonSerializable):
    """Emitted when a text content block begins"""
    content_index: int
    type: Literal["text_start"] = "text_start"
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class TextDelta(JsonSerializable):
    """Emitted for each text token/chunk during streaming"""
    content_index: int
    delta: str
    type: Literal["text_delta"] = "text_delta"
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class TextEnd(JsonSerializable):
    """Emitted when a text content block completes"""
    content_index: int
    content: str  # Complete accumulated text
    type: Literal["text_end"] = "text_end"
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ThinkingStart(JsonSerializable):
    """Emitted when a thinking/reasoning content block begins"""
    content_index: int
    type: Literal["thinking_start"] = "thinking_start"
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ThinkingDelta(JsonSerializable):
    """Emitted for each thinking token/chunk during streaming"""
    content_index: int
    delta: str
    type: Literal["thinking_delta"] = "thinking_delta"
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ThinkingEnd(JsonSerializable):
    """Emitted when a thinking/reasoning content block completes"""
    content_index: int
    content: str  # Complete accumulated thinking
    type: Literal["thinking_end"] = "thinking_end"
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ToolCallStart(JsonSerializable):
    """Emitted when a tool call content block begins"""
    content_index: int
    tool_call_id: str
    tool_name: str
    type: Literal["toolcall_start"] = "toolcall_start"
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ToolCallDelta(JsonSerializable):
    """Emitted for each chunk of tool call arguments during streaming

    The partial_args field contains the best-effort parsed JSON from the
    accumulated argument string so far. May be incomplete objects/arrays.
    """
    content_index: int
    tool_call_id: str
    delta: str  # Raw JSON chunk
    partial_args: dict[str, Any]  # Best-effort parsed partial JSON
    type: Literal["toolcall_delta"] = "toolcall_delta"
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ToolCallEnd(JsonSerializable):
    """Emitted when a tool call content block completes"""
    content_index: int
    tool_call: ToolCall  # Complete parsed tool call
    type: Literal["toolcall_end"] = "toolcall_end"
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ToolCallError(JsonSerializable):
    """Emitted when tool call argument parsing fails"""
    content_index: int
    tool_call_id: str
    tool_name: str
    error: str
    raw_arguments: str
    type: Literal["toolcall_error"] = "toolcall_error"
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ToolResultReceived(JsonSerializable):
    """Emitted when a tool execution result is received"""
    tool_call_id: str
    content: str
    is_error: bool = False
    error: Optional[str] = None
    details: Optional[dict[str, Any]] = None  # UI-only structured data (e.g., diff for edit tool)
    type: Literal["tool_result"] = "tool_result"
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class StreamDone(JsonSerializable):
    """Emitted when streaming completes successfully"""
    finish_reason: str  # "stop", "length", "tool_calls", etc.
    type: Literal["done"] = "done"
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class StreamError(JsonSerializable):
    """Emitted when streaming encounters an error"""
    error: str
    type: Literal["error"] = "error"
    timestamp: float = field(default_factory=time.time)


# Union type for all streaming events
StreamEvent = (
    LLMCallStart
    | StreamStart
    | TextStart
    | TextDelta
    | TextEnd
    | ThinkingStart
    | ThinkingDelta
    | ThinkingEnd
    | ToolCallStart
    | ToolCallDelta
    | ToolCallEnd
    | ToolCallError
    | ToolResultReceived
    | StreamDone
    | StreamError
)


# Provider abstraction protocol (inspired by pi-ai)
# All provider streaming functions must implement this interface
@runtime_checkable
class ProviderStreamFunction(Protocol):
    """Protocol for provider-specific streaming functions.

    All providers (OpenAI, Anthropic, Google, etc.) must implement a function
    matching this signature. The function accepts an Actor (with endpoint, trajectory, tools)
    and an event callback, then streams granular events back via the callback.

    Providers may accept additional provider-specific parameters via **kwargs.

    Example implementations:
    - rollout_openai(actor, on_chunk) -> Actor
    - rollout_anthropic(actor, on_chunk, user_message_for_thinking=..., **kwargs) -> Actor
    - rollout_google(actor, on_chunk) -> Actor
    """

    async def __call__(
        self,
        actor: 'Actor',
        on_chunk: Callable[[StreamEvent], Awaitable[None]],
        **kwargs: Any,
    ) -> 'Actor':
        """Stream LLM response and return updated Actor with new trajectory message.

        Args:
            actor: Current actor state (endpoint, trajectory, tools)
            on_chunk: Async callback for streaming events
            **kwargs: Provider-specific optional parameters

        Returns:
            Updated actor with new assistant message appended to trajectory
        """
        ...


# ContentBlock types for structured message content (inspired by pi-ai)
# These allow messages to contain mixed content: text, thinking, tool calls, images


@dataclass(frozen=True)
class TextContent(JsonSerializable):
    """Text content block in a message."""
    type: Literal["text"] = "text"
    text: str = ""
    text_signature: str | None = None  # Provider-specific identifier


@dataclass(frozen=True)
class ThinkingContent(JsonSerializable):
    """Thinking/reasoning content block in a message.

    Used by Anthropic (thinking blocks) and OpenAI o1/o3 (reasoning_content).
    """
    type: Literal["thinking"] = "thinking"
    thinking: str = ""
    thinking_signature: str | None = None  # Provider-specific identifier (e.g., GPT-5 Codex reasoning item ID)


@dataclass(frozen=True)
class ToolCallContent(JsonSerializable):
    """Tool call content block in a message."""
    type: Literal["toolCall"] = "toolCall"
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    thought_signature: str | None = None  # Google-specific opaque context


@dataclass(frozen=True)
class ImageContent(JsonSerializable):
    """Image content block in a message (for vision models)."""
    type: Literal["image"] = "image"
    image_url: str = ""  # base64 data URL or HTTP URL
    detail: str | None = None  # OpenAI detail parameter: "low", "high", "auto"


# Union type for all content blocks
ContentBlock = TextContent | ThinkingContent | ToolCallContent | ImageContent


@dataclass(frozen=True)
class Message(JsonSerializable):
    """Unified message type supporting all providers.

    Content can be:
    - str: Simple text message (most common)
    - list[ContentBlock]: Structured message with text/thinking/tools/images

    Role can be:
    - "user": User input
    - "assistant": Model response
    - "tool": Tool execution result
    """
    role: str
    content: str | list[ContentBlock] | None
    # Provider metadata for message transformation
    provider: str | None = None  # e.g., "anthropic", "openai", "google"
    api: str | None = None  # e.g., "anthropic-messages", "openai-completions", "openai-responses"
    model: str | None = None  # e.g., "claude-3-5-sonnet-20241022", "gpt-4o"
    # For tool role messages: which tool call this is responding to
    tool_call_id: str | None = None
    # UI-only structured data (stripped before LLM)
    details: dict[str, Any] | None = None

    def get_tool_calls(self) -> list[ToolCall]:
        """Extract tool calls from ContentBlocks.

        Tiger Style: Helper for common operation, makes migration easier.
        """
        if not isinstance(self.content, list):
            return []

        tool_calls = []
        for block in self.content:
            if isinstance(block, ToolCallContent):
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    args=block.arguments
                ))
        return tool_calls

    def __repr__(self) -> str:
        """Tiger Style: Bounded repr, truncate large content.

        Vision messages can contain base64 images (100KB+).
        Always truncate to prevent terminal spam.
        """
        # Truncate content for display
        if isinstance(self.content, str):
            content_preview = self.content[:100] + "..." if len(self.content) > 100 else self.content
        elif isinstance(self.content, list):
            # Show ContentBlock types
            block_types = [b.type for b in self.content if hasattr(b, 'type')]
            content_preview = f"[{len(self.content)} blocks: {', '.join(block_types)}]"
        else:
            content_preview = str(self.content)

        return f"Message(role={self.role!r}, content={content_preview!r})"


@dataclass(frozen=True)
class Cost(JsonSerializable):
    """Cost breakdown in USD. Immutable.

    Following IMMUTABILITY_AND_FP: frozen dataclass for data that doesn't change.
    """

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0

    @property
    def total(self) -> float:
        return self.input + self.output + self.cache_read + self.cache_write


@dataclass(frozen=True)
class Usage(JsonSerializable):
    """Token usage with cost tracking. Immutable.

    Following IMMUTABILITY_AND_FP: state changes are explicit via replace().
    Following SSA: each transformation creates a new binding.

    Example:
        # SSA style - named intermediate values
        raw_usage = Usage(input_tokens=100, output_tokens=50)
        usage_with_cost = replace(raw_usage, cost=calculated_cost)
    """

    # Token counts (primary fields)
    input_tokens: int = 0  # Non-cached input tokens
    output_tokens: int = 0  # Output/completion tokens (excludes reasoning)
    reasoning_tokens: int = 0  # Reasoning/thinking tokens (OpenAI o1/o3, Anthropic thinking)
    cache_read_tokens: int = 0  # Tokens read from cache (Anthropic/OpenAI)
    cache_write_tokens: int = 0  # Tokens written to cache (Anthropic)

    # Cost breakdown (computed by provider after API response)
    cost: Cost = field(default_factory=Cost)

    # Computed properties
    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.reasoning_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    # Legacy aliases for backwards compatibility (don't break userspace)
    @property
    def prompt_tokens(self) -> int:
        """Legacy alias: input_tokens + cache_read_tokens"""
        return self.input_tokens + self.cache_read_tokens

    @property
    def completion_tokens(self) -> int:
        """Legacy alias: output_tokens + reasoning_tokens (rolled together for compat)"""
        return self.output_tokens + self.reasoning_tokens


@dataclass(frozen=True)
class Logprob(JsonSerializable):
    token: str
    logprob: float
    bytes: list[int]
    top_logprobs: list[float]


@dataclass(frozen=True)
class Logprobs(JsonSerializable):
    content: list[Logprob] = field(default_factory=list)


@dataclass(frozen=True)
class Choice(JsonSerializable):
    index: int
    message: Message
    finish_reason: str
    logprobs: Logprobs | None = None
    stop_reason: Any | None = None


@dataclass(frozen=True)
class TokenInfo(JsonSerializable):
    logprob: float
    rank: int
    decoded_token: str


PromptLogprob = Optional[dict[str, TokenInfo]]
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
    kv_transfer_params: Any | None = None
    choices: list[Choice] = field(default_factory=list)
    prompt_logprobs: list[PromptLogprob] | None = None


@dataclass
class Trajectory(JsonSerializable):
    completions: list[ChatCompletion] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)   # debugging only
    rewards: float = 0.0
    group: int = 0
    replica: int = 0
    advantages: float = 0.0     # scalar; broadcast later if needed
    metadata: dict[str, Any] = field(default_factory=dict)  # For dataset-specific info (e.g., ground truth)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Trajectory":
        """Rebuild nested dataclasses so type hints stay correct."""
        assert data is not None
        assert isinstance(data, dict)

        comps: list[ChatCompletion] = []
        for comp in data.get("completions", []):
            assert comp is not None
            assert isinstance(comp, dict)
            usage_dict = comp.get("usage", {})
            assert "prompt_tokens" in usage_dict
            assert "completion_tokens" in usage_dict
            assert "total_tokens" in usage_dict
            usage = Usage(
                prompt_tokens=usage_dict["prompt_tokens"],
                completion_tokens=usage_dict["completion_tokens"],
                total_tokens=usage_dict["total_tokens"],
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
            metadata=data.get("metadata", {}),
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
    def to_jsonl(trajectories: list["Trajectory"]) -> str:
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
    def from_jsonl(jsonl_str: str) -> list["Trajectory"]:
        assert jsonl_str is not None
        assert isinstance(jsonl_str, str)
        result = [Trajectory.from_json(line) for line in jsonl_str.strip().splitlines() if line]
        assert isinstance(result, list)
        return result

    # ---------- disk I/O ---------------------------------------------------
    @staticmethod
    def save_jsonl(trajectories: list["Trajectory"], filepath: str) -> None:
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
    def load_jsonl(filepath: str) -> list["Trajectory"]:
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

        with open(filepath, encoding='utf-8') as f:
            for line in f:
                line_stripped = line.strip()
                if line_stripped:  # Skip empty lines
                    yield Trajectory.from_json(line_stripped)

    # ---------- helpers that work pre-/post-serialisation ------------------
    @staticmethod
    def _usage_total(usage: Usage | dict[str, Any], key: str) -> int:
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
    properties: dict[str, Any]
    type: str = "object"


@dataclass(frozen=True)
class ToolFunction(JsonSerializable):
    name: str
    description: str
    parameters: ToolFunctionParameter
    required: list[str] = field(default_factory=list)


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
    ABORTED = "ABORTED"


@dataclass(frozen=True)
class ToolResult(JsonSerializable):
    tool_call_id: str = ""
    is_error: bool = False
    content: str = ""
    error: str | None = None
    stop_reason: StopReason | None = None
    # UI-only structured data (stripped before LLM)
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolConfirmResult(JsonSerializable):
    """Result of tool confirmation"""
    proceed: bool
    tool_result: ToolResult | None = None
    user_message: str | None = None

# ── Core Agent Framework Types ────────────────────────────────────────────────


@runtime_checkable
class Environment(Protocol):
    """Protocol that all environments must satisfy for composition over inheritance."""

    def get_tools(self) -> list[Tool]:
        """Return available tools for this environment."""
        ...

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: 'AgentState',
        run_config: 'RunConfig',
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult:
        """Execute a tool call in this environment.

        Args:
            tool_call: The tool call to execute
            current_state: Current agent state
            run_config: Run configuration
            cancel_scope: Optional Trio cancel scope for graceful cancellation
        """
        ...

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        """Check if tool requires confirmation."""
        ...

    def get_tool_formatter(self, tool_name: str) -> 'ToolFormatter | None':
        """Return optional TUI formatter for this tool.

        Args:
            tool_name: Name of the tool to format

        Returns:
            A formatter function or None to use the default formatter.
            The formatter receives (tool_name, args, result, expanded) and returns
            a formatted string for display in the TUI.
        """
        ...

    def get_status_info(self) -> dict[str, str] | None:
        """Return key-value pairs to display in TUI status line.

        Optional method - environments can return None or not implement this.
        Example: {"cwd": "~/research", "branch": "main"}
        """
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
    oauth_token: str = ""  # OAuth bearer token (takes precedence over api_key for Anthropic)
    max_tokens: int = 8192
    temperature: float = 1.0
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    reasoning_effort: str | None = None  # for openai
    max_completion_tokens: int | None = None  # for openai
    thinking: dict[str, Any] | None = None  # for anthropic
    # Retry configuration
    max_retries: int = 3  # Number of retries for rate limits/transient errors
    timeout: float = 120.0  # Timeout in seconds for API calls
    # Extra params merged into the raw chat request for custom servers
    extra_params: dict[str, Any] | None = None

    def __post_init__(self):
        """Validate endpoint configuration.

        Tiger Style: Crash loud on invalid config, explicit error messages.
        """
        # Validate Claude thinking budget (Anthropic requires >= 1024 tokens)
        if self.thinking is not None and self.provider == "anthropic":
            assert isinstance(self.thinking, dict), f"thinking must be dict, got {type(self.thinking)}"
            if self.thinking.get("type") == "enabled":
                budget = self.thinking.get("budget_tokens", 0)
                assert isinstance(budget, int), f"budget_tokens must be int, got {type(budget)}"
                assert budget >= 1024, (
                    f"Claude thinking budget_tokens must be >= 1024, got {budget}. "
                    "Anthropic API requirement for extended thinking mode."
                )
                # max_tokens must be greater than thinking budget
                assert self.max_tokens > budget, (
                    f"max_tokens ({self.max_tokens}) must be greater than thinking.budget_tokens ({budget}). "
                    f"Anthropic requires max_tokens > budget_tokens to allow space for both thinking and response. "
                    f"See https://docs.claude.com/en/docs/build-with-claude/extended-thinking#max-tokens-and-context-window-size"
                )
                # Anthropic requires temperature=1.0 when thinking is enabled
                assert self.temperature == 1.0, (
                    f"Anthropic requires temperature=1.0 when thinking is enabled, got {self.temperature}. "
                    "See https://docs.claude.com/en/docs/build-with-claude/extended-thinking"
                )


@dataclass(frozen=True)
class Actor(JsonSerializable):
    trajectory: Trajectory
    endpoint: Endpoint
    tools: list[Tool] = field(default_factory=list)


@dataclass(frozen=True)
class AgentState:
    actor: Actor
    environment: Environment | None
    stop: StopReason | None = None
    turn_idx: int = 0
    pending_tool_calls: list[ToolCall] = field(default_factory=list)
    next_tool_idx: int = 0  # Which tool we're about to process
    timestamp: str = datetime.now(timezone.utc).isoformat() + 'Z'
    session_id: str | None = None  # Session ID for persistence (set by run_agent)
    # For forking: when resuming with different config, create child session
    parent_session_id: str | None = None  # Parent session to branch from
    branch_point: int | None = None  # Message index where branching from parent
    confirm_tools: bool = False  # Whether tool confirmation is required


# Forward declarations for RunConfig (needs to be after AgentState but before default handlers)
async def default_stdin_handler(prompt: str) -> str:
    """Default input handler using trio.to_thread.run_sync for non-blocking input."""
    return await trio.to_thread.run_sync(input, prompt)


async def default_confirm_tool(tc: ToolCall, state: 'AgentState', run_config: 'RunConfig') -> tuple['AgentState', ToolConfirmResult]:
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
    on_chunk: Callable[[StreamEvent], Awaitable[None]]
    on_input: Callable[[str], Awaitable[str]] = field(default_factory=lambda: default_stdin_handler)
    confirm_tool: Callable[[ToolCall, 'AgentState', 'RunConfig'], Awaitable[tuple['AgentState', ToolConfirmResult]]] = field(default_factory=lambda: default_confirm_tool)
    handle_tool_error: Callable[[ToolResult, 'AgentState'], 'AgentState'] = lambda tr, s: s
    on_step_start: Callable[['AgentState'], 'AgentState'] = lambda s: s
    handle_stop: Callable[['AgentState'], 'AgentState'] = lambda s: s
    handle_no_tool: Callable[['AgentState', 'RunConfig'], Awaitable['AgentState']] = field(default_factory=lambda: default_no_tool_handler)
    user_message_for_thinking: str | None = None
    inline_thinking: str | None = None
    show_progress: bool = False  # Enable turn-level progress tracking
    cancel_scope: trio.CancelScope | None = None  # Optional Trio cancel scope for graceful cancellation. When cancel_scope.cancel() is called, any in-flight HTTP request is immediately cancelled and trio.Cancelled is raised. The agent loop catches this and sets stop=StopReason.ABORTED.
    # Session persistence
    session_store: Any | None = None  # SessionStore instance for persistence (session_id is on AgentState)


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
        ...     run_config=RunConfig(handle_stop=handle_stop_max_turns(10)),
        ...     max_concurrent=4,
        ... )
    """
    # Required: how to compute rewards
    reward_fn: RewardFunction

    # Agent execution
    run_config: RunConfig | None = None  # If None, use silent default

    # Dataset control
    max_samples: int | None = None  # If None, evaluate all
    sample_id_fn: Callable[[int, dict[str, Any]], str] = field(
        default_factory=lambda: lambda i, _: f"sample_{i:04d}"
    )

    # Parallelization
    max_concurrent: int = 1

    # Output
    output_dir: Path | None = None
    eval_name: str = "evaluation"

    # Logging
    verbose: bool = True
    show_progress: bool = False  # Enable sample-level progress tracking
    stream_tokens: bool = False  # Stream LLM tokens to stdout (used if run_config is None)


# ── Session Types ──────────────────────────────────────────────────────────────
# Types for persisting agent sessions (trajectories, config, environment state).
# See docs/design/rollouts-session-design.md for design details.


class SessionStatus(Enum):
    """Session status."""

    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"


@dataclass
class EndpointConfig:
    """LLM endpoint configuration.

    Stored in session for reproducibility. This is the serializable subset
    of Endpoint - excludes api_key and other runtime-only fields.
    """

    model: str
    provider: str = "anthropic"
    temperature: float = 0.0
    max_tokens: int | None = None
    # Extended thinking
    thinking: bool = False
    thinking_budget: int | None = None
    # Additional params
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "thinking": self.thinking,
            "thinking_budget": self.thinking_budget,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EndpointConfig":
        return cls(
            model=data["model"],
            provider=data.get("provider", "anthropic"),
            temperature=data.get("temperature", 0.0),
            max_tokens=data.get("max_tokens"),
            thinking=data.get("thinking", False),
            thinking_budget=data.get("thinking_budget"),
            extra=data.get("extra", {}),
        )


@dataclass
class EnvironmentConfig:
    """Environment configuration.

    Stored in session for reproducibility.
    """

    type: str  # e.g., "gpumode", "localfs"
    # Environment-specific config
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvironmentConfig":
        return cls(
            type=data["type"],
            config=data.get("config", {}),
        )


@dataclass
class SessionMessage:
    """A message in the session trajectory.

    This is a simplified serializable version of Message for session storage.
    """

    role: str  # user, assistant, tool
    content: str | list[dict[str, Any]]  # text or content blocks
    tool_call_id: str | None = None  # for tool results
    # Additional metadata
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.timestamp:
            result["timestamp"] = self.timestamp
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionMessage":
        return cls(
            role=data["role"],
            content=data["content"],
            tool_call_id=data.get("tool_call_id"),
            timestamp=data.get("timestamp"),
        )


@dataclass
class AgentSession:
    """A persisted agent session.

    This is the record stored in ~/.rollouts/sessions/<session_id>/
    """

    # Identity
    session_id: str
    parent_id: str | None = None  # None for root sessions
    branch_point: int | None = None  # message index where branched from parent

    # Config (serializable, stored in session.json)
    endpoint: EndpointConfig = field(default_factory=lambda: EndpointConfig(model=""))
    environment: EnvironmentConfig = field(default_factory=lambda: EnvironmentConfig(type=""))

    # Trajectory
    messages: list[SessionMessage] = field(default_factory=list)

    # Environment state (opaque, env-specific)
    environment_state: dict[str, Any] | None = None

    # Outcome
    status: SessionStatus = SessionStatus.PENDING
    reward: float | dict[str, float] | None = None

    # Metadata
    tags: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "session_id": self.session_id,
            "parent_id": self.parent_id,
            "branch_point": self.branch_point,
            "endpoint": self.endpoint.to_dict(),
            "environment": self.environment.to_dict(),
            # messages are stored separately in messages.jsonl
            "environment_state": self.environment_state,
            "status": self.status.value,
            "reward": self.reward,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], messages: list[SessionMessage] | None = None) -> "AgentSession":
        """Deserialize from dict."""
        return cls(
            session_id=data["session_id"],
            parent_id=data.get("parent_id"),
            branch_point=data.get("branch_point"),
            endpoint=EndpointConfig.from_dict(data["endpoint"]),
            environment=EnvironmentConfig.from_dict(data["environment"]),
            messages=messages or [],
            environment_state=data.get("environment_state"),
            status=SessionStatus(data.get("status", "pending")),
            reward=data.get("reward"),
            tags=data.get("tags", {}),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )
