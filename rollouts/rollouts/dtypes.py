from __future__ import annotations

import json
import os
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    Self,
    runtime_checkable,
)

if TYPE_CHECKING:
    from .core.eval import Score

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

# TUI formatter type - receives (tool_name, args, result, detail_level, theme) and returns formatted string
# Theme is optional to allow headless/non-TUI usage
ToolFormatter = Callable[[str, dict[str, Any], dict[str, Any] | None, "DetailLevel", Any], str]


class DetailLevel(IntEnum):
    """Detail level for displaying content in TUI.

    Controls how much information is shown for tool outputs, messages, and errors.
    Use +/- keys to cycle through levels globally.
    """

    COMPACT = 0  # Header + 1-2 line summary only
    STANDARD = 1  # Default preview (5-10 lines)
    EXPANDED = 2  # Full output, no truncation
    # Future levels (not yet implemented):
    # VERBOSE = 3   # Full output + metadata
    # DEBUG = 4     # Everything including internal state


@dataclass(frozen=True)
class ToolRenderConfig:
    """Rendering config for tool output in TUI.

    Environments can optionally provide this to customize how tools render.
    All fields have sensible defaults, so most tools need no config at all.

    For simple tools: just override header_fn and maybe summaries
    For complex tools: provide custom_formatter for full control

    Example:
        # Simple tool - just customize header
        ToolRenderConfig(
            header_fn=lambda name, args: f"bash(command={repr(args.get('command', '...'))})",
            success_summary="Command completed",
        )

        # Complex tool - full control
        ToolRenderConfig(custom_formatter=my_custom_format_fn)
    """

    # How to build the header line
    # If None, uses default: tool_name(arg1=..., arg2=...)
    header_fn: Callable[[str, dict[str, Any]], str] | None = None

    # Output display settings - lines shown at each detail level
    # COMPACT: minimal preview, STANDARD: default, EXPANDED: full output
    lines_compact: int = 2  # Header + 1-2 lines
    lines_standard: int = 10  # Default preview
    lines_expanded: int = -1  # -1 = unlimited

    # Legacy field for backward compatibility (maps to lines_standard)
    max_lines: int = 10

    style_fn: str = "diff_context_fg"  # Theme method name for styling output lines

    # Summary lines (shown after header, before output)
    # If None, no summary shown - tool name is usually self-documenting
    success_summary: str | None = None
    error_summary: str | None = None

    # For complex tools that need full control over rendering
    # If provided, all other fields are ignored
    # Signature: (tool_name, args, result, detail_level, theme) -> str
    custom_formatter: ToolFormatter | None = None

    def get_max_lines(self, level: DetailLevel) -> int:
        """Get the max lines for a given detail level.

        Args:
            level: The detail level to get lines for

        Returns:
            Max lines to display (-1 for unlimited)
        """
        if level == DetailLevel.COMPACT:
            return self.lines_compact
        elif level == DetailLevel.STANDARD:
            return self.lines_standard
        else:  # EXPANDED or higher
            return self.lines_expanded


# Verbose function for debugging
def verbose(level: int = 1) -> bool:
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
    except json.JSONDecodeError:
        pass
    else:
        # Model might return non-object JSON (e.g., "8" instead of {"result": 8})
        # Return empty dict rather than crashing - caller will handle via ToolCallError
        if not isinstance(result, dict):
            return {}
        return result

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
        cleaned + '"}',  # Close incomplete string value
        cleaned + "]",  # Close incomplete array
        cleaned + "}",  # Close incomplete object
        cleaned + '"}]',  # Close string in array
        cleaned + '"}}',  # Close string in nested object
    ]

    # Also try removing trailing incomplete key/value
    if "," in cleaned:
        # Remove everything after the last comma (incomplete key-value pair)
        last_comma = cleaned.rfind(",")
        truncated = cleaned[:last_comma]
        attempts.extend([truncated + "}", truncated + "]}", truncated + "}}"])

    # If there's a colon without a value, remove the incomplete pair
    if ":" in cleaned:
        # Find the last complete comma before incomplete value
        parts = cleaned.split(",")
        for i in range(len(parts) - 1, -1, -1):
            # Check if this part has both key and value
            truncated = ",".join(parts[:i])
            if truncated:
                attempts.extend([truncated + "}", truncated + "]}", truncated + "}}"])

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

    TODO(cleanup): Delete this base class. Replace with standalone functions:
        def to_json(obj: Any) -> str: return json.dumps(asdict(obj), ensure_ascii=False)
        def from_json(cls: type[T], s: str) -> T: return dacite.from_dict(cls, json.loads(s))
    ~50 classes inherit from this but gain nothing that two functions wouldn't provide.
    tinygrad would reject: inheritance for serialization is enterprise disease.
    """

    def to_json(self) -> str:
        """Serialize to JSON string"""
        assert self is not None
        result = json.dumps(asdict(self), ensure_ascii=False)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        return result

    @classmethod
    def from_json(cls, json_str: str) -> Self:
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
    # Parse error info - set when tool call JSON was malformed
    parse_error: str | None = None


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
class SemaphoreWaitStart(JsonSerializable):
    """Emitted when waiting to acquire a semaphore (api_limiter or tool_limiter).

    Enables distinguishing "waiting (queue)" from "waiting (api)" in status display.
    """

    limiter_type: Literal["api", "tool"]
    type: Literal["semaphore_wait_start"] = "semaphore_wait_start"
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class SemaphoreAcquired(JsonSerializable):
    """Emitted when semaphore is acquired (waiting is over)."""

    limiter_type: Literal["api", "tool"]
    wait_duration_ms: float  # How long we waited for the semaphore
    type: Literal["semaphore_acquired"] = "semaphore_acquired"
    timestamp: float = field(default_factory=time.time)


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
class FirstToken(JsonSerializable):
    """Emitted when first content token arrives (for TTFT tracking)"""

    ttft_ms: float  # Time from request start to first token
    type: Literal["first_token"] = "first_token"
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
class ToolExecutionStart(JsonSerializable):
    """Emitted when a tool begins execution (after confirmation, before result)"""

    tool_call_id: str
    tool_name: str
    type: Literal["tool_execution_start"] = "tool_execution_start"
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ToolResultReceived(JsonSerializable):
    """Emitted when a tool execution result is received"""

    tool_call_id: str
    content: str | list[ContentBlock]  # Forward ref - ContentBlock defined later
    is_error: bool = False
    error: str | None = None
    details: dict[str, Any] | None = None  # UI-only structured data (e.g., diff for edit tool)
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


@dataclass(frozen=True)
class RetryStart(JsonSerializable):
    """Emitted when starting a retry attempt after a transient error.

    Allows TUI to show retry status (e.g., "Retrying (1/3) in 2s... (esc to cancel)")
    instead of raw print() statements that mess up the display.
    """

    attempt: int  # Current attempt number (1-indexed)
    max_attempts: int  # Total number of attempts allowed
    delay_seconds: float  # Seconds until retry
    error_message: str  # What error triggered the retry
    provider: str  # "anthropic", "openai", etc.
    type: Literal["retry_start"] = "retry_start"
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class RetryEnd(JsonSerializable):
    """Emitted when retry completes (either success or final failure).

    Allows TUI to clean up retry display and show appropriate status.
    """

    success: bool  # True if retry succeeded, False if all attempts exhausted
    attempt: int  # Final attempt number
    final_error: str | None = None  # Error message if failed
    type: Literal["retry_end"] = "retry_end"
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class LLMCallEnd(JsonSerializable):
    """Emitted after LLM API call completes (success or error).

    Wide event: includes all context needed for profiling without correlation.
    """

    duration_ms: float
    provider: str  # "anthropic", "openai", etc.
    model: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost: float | None = None  # Total cost in USD
    ttft_ms: float | None = None  # Time to first token
    status: Literal["success", "error"] = "success"
    error: str | None = None
    type: Literal["llm_call_end"] = "llm_call_end"
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ToolExecutionEnd(JsonSerializable):
    """Emitted after tool execution completes (success or error).

    Wide event: includes result summary for profiling without correlation.
    """

    tool_call_id: str
    tool_name: str
    duration_ms: float
    status: Literal["success", "error"] = "success"
    is_error: bool = False  # Tool returned error result
    # Result summary (tool-specific, optional)
    result_summary: dict[str, Any] | None = None
    type: Literal["tool_execution_end"] = "tool_execution_end"
    timestamp: float = field(default_factory=time.time)


# Union type for all streaming events
StreamEvent = (
    SemaphoreWaitStart
    | SemaphoreAcquired
    | LLMCallStart
    | LLMCallEnd
    | StreamStart
    | FirstToken
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
    | ToolExecutionStart
    | ToolExecutionEnd
    | ToolResultReceived
    | StreamDone
    | StreamError
    | RetryStart
    | RetryEnd
    | StreamChunk  # DEPRECATED: Included for backwards compatibility, will be removed
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
        actor: Actor,
        on_chunk: Callable[[StreamEvent], Awaitable[None]],
        **kwargs: Any,
    ) -> Actor:
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
    thinking_signature: str | None = (
        None  # Provider-specific identifier (e.g., GPT-5 Codex reasoning item ID)
    )


@dataclass(frozen=True)
class ToolCallContent(JsonSerializable):
    """Tool call content block in a message."""

    type: Literal["toolCall"] = "toolCall"
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    thought_signature: str | None = None  # Google-specific opaque context
    # Parse error info - set when tool call JSON was malformed
    parse_error: str | None = None
    raw_arguments: str | None = None  # Original malformed JSON string


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
    # Session storage timestamp (optional, only set when persisting)
    timestamp: str | None = None

    def get_tool_calls(self) -> list[ToolCall]:
        """Extract tool calls from ContentBlocks.

        Tiger Style: Helper for common operation, makes migration easier.
        """
        if not isinstance(self.content, list):
            return []

        tool_calls = []
        for block in self.content:
            if isinstance(block, ToolCallContent):
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        args=block.arguments,
                        parse_error=block.parse_error,
                    )
                )
        return tool_calls

    def __repr__(self) -> str:
        """Tiger Style: Bounded repr, truncate large content.

        Vision messages can contain base64 images (100KB+).
        Always truncate to prevent terminal spam.
        """
        # Truncate content for display
        if isinstance(self.content, str):
            content_preview = (
                self.content[:100] + "..." if len(self.content) > 100 else self.content
            )
        elif isinstance(self.content, list):
            # Show ContentBlock types
            block_types = [b.type for b in self.content if hasattr(b, "type")]
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
class RequestSpan(JsonSerializable):
    """Per-request metrics for cost/latency analysis. Persisted to spans.jsonl.

    Captures everything needed to analyze request performance:
    - Timing: when it started, how long it took
    - Tokens: input/output/cache breakdown
    - Cost: USD breakdown by token type
    - Provider: which provider/model served the request
    """

    # Timing
    started_at: str  # ISO timestamp
    duration_ms: float  # Total request duration

    # Provider info
    provider: str  # e.g., "openrouter", "anthropic"
    model: str  # e.g., "moonshotai/kimi-k2.5"
    api_base: str | None = None  # e.g., "https://openrouter.ai/api/v1"

    # Time to first token (network + queue + model warmup)
    ttft_ms: float | None = None

    # Token counts
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    # Cost breakdown (USD)
    cost_input: float = 0.0
    cost_output: float = 0.0
    cost_cache_read: float = 0.0
    cost_cache_write: float = 0.0

    # Request metadata
    request_id: str | None = None  # Provider's request ID if available
    finish_reason: str | None = None  # e.g., "stop", "tool_calls", "length"
    error: str | None = None  # Error message if request failed

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.reasoning_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    @property
    def cost_total(self) -> float:
        return self.cost_input + self.cost_output + self.cost_cache_read + self.cost_cache_write


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
    bytes: list[int] = field(default_factory=list)
    top_logprobs: list[float] = field(default_factory=list)
    token_id: int | None = None  # Token ID from response (for TI/TO)


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
    token_ids: tuple[int, ...] | None = None  # Generated token IDs for TI/TO


@dataclass(frozen=True)
class TokenInfo(JsonSerializable):
    logprob: float
    rank: int
    decoded_token: str


PromptLogprob = dict[str, TokenInfo] | None
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
    prompt_token_ids: tuple[int, ...] | None = None  # Prompt token IDs for TI/TO (from server)


@dataclass(frozen=True)
class TrajectoryAnnotations(JsonSerializable):
    """Optional rollout/training annotations attached to a trajectory.

    These fields are currently duplicated in legacy top-level Trajectory fields.
    The nested bundle is the migration target; the top-level fields remain for
    compatibility until callers are moved over.
    """

    reward: float | dict[str, float] | None = None
    group: int | None = None
    replica: int | None = None
    advantage: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrajectoryAnnotations:
        assert data is not None
        assert isinstance(data, dict)
        return cls(
            reward=data.get("reward"),
            group=data.get("group"),
            replica=data.get("replica"),
            advantage=data.get("advantage"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.reward is not None:
            result["reward"] = self.reward
        if self.group is not None:
            result["group"] = self.group
        if self.replica is not None:
            result["replica"] = self.replica
        if self.advantage is not None:
            result["advantage"] = self.advantage
        return result


@dataclass(frozen=True)
class TrajectorySession(JsonSerializable):
    """Session/branching metadata attached to a trajectory."""

    session_id: str | None = None
    parent_id: str | None = None
    branch_point: int | None = None
    endpoint: Endpoint | None = None
    status: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    vcs: dict[str, str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrajectorySession:
        assert data is not None
        assert isinstance(data, dict)
        raw_tags = data.get("tags", {})
        tags = raw_tags if isinstance(raw_tags, dict) else {}
        status = data.get("status")
        if status is not None:
            status = str(status)
        return cls(
            session_id=data.get("session_id"),
            parent_id=data.get("parent_id"),
            branch_point=data.get("branch_point"),
            endpoint=(
                Endpoint.from_dict(data["endpoint"])
                if isinstance(data.get("endpoint"), dict)
                else None
            ),
            status=status,
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            tags=tags,
            vcs=data.get("vcs"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.session_id is not None:
            result["session_id"] = self.session_id
        if self.parent_id is not None:
            result["parent_id"] = self.parent_id
        if self.branch_point is not None:
            result["branch_point"] = self.branch_point
        if self.endpoint is not None:
            result["endpoint"] = self.endpoint.to_dict(exclude_secrets=True)
        if self.status is not None:
            result["status"] = self.status
        if self.created_at is not None:
            result["created_at"] = self.created_at
        if self.updated_at is not None:
            result["updated_at"] = self.updated_at
        if self.tags:
            result["tags"] = self.tags
        if self.vcs is not None:
            result["vcs"] = self.vcs
        return result


class EnvironmentResumeMode(Enum):
    """Resumability guarantee for serialized environment data."""

    COLD = "cold"
    WARM = "warm"
    NONE = "none"


@dataclass(frozen=True)
class TrajectoryEnvironment(JsonSerializable):
    """Environment bundle attached to a trajectory.

    `config` mirrors the durable environment selection/configuration.
    `state` is an optional serialized checkpoint snapshot.
    `resume_mode` states whether `state` is cold-resumable, warm-resumable,
    or informational only.
    """

    kind: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] | None = None
    resume_mode: EnvironmentResumeMode = EnvironmentResumeMode.COLD

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrajectoryEnvironment:
        assert data is not None
        assert isinstance(data, dict)
        raw_mode = data.get("resume_mode", EnvironmentResumeMode.COLD.value)
        if isinstance(raw_mode, EnvironmentResumeMode):
            resume_mode = raw_mode
        else:
            resume_mode = EnvironmentResumeMode(str(raw_mode))
        raw_config = data.get("config", {})
        config = raw_config if isinstance(raw_config, dict) else {}
        raw_state = data.get("state")
        state = raw_state if isinstance(raw_state, dict) else None
        return cls(
            kind=data.get("kind", ""),
            config=config,
            state=state,
            resume_mode=resume_mode,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind,
            "config": self.config,
            "resume_mode": self.resume_mode.value,
        }
        if self.state is not None:
            result["state"] = self.state
        return result

    @classmethod
    def from_session_parts(
        cls,
        environment: EnvironmentConfig,
        state: dict[str, Any] | None = None,
    ) -> TrajectoryEnvironment:
        assert environment is not None
        resume_mode = EnvironmentResumeMode.COLD
        if isinstance(state, dict) and "_env_ref" in state:
            # Warm-only checkpoints depend on a live in-process environment.
            resume_mode = EnvironmentResumeMode.WARM
        return cls(
            kind=environment.type,
            config=environment.config,
            state=state,
            resume_mode=resume_mode,
        )


@dataclass(frozen=True)
class Trajectory(JsonSerializable):
    completions: list[ChatCompletion] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)  # debugging only
    rewards: float = 0.0
    group: int = 0
    replica: int = 0
    advantages: float = 0.0  # scalar; broadcast later if needed
    metadata: dict[str, Any] = field(
        default_factory=dict
    )  # For dataset-specific info (e.g., ground truth)
    annotations: TrajectoryAnnotations = field(default_factory=TrajectoryAnnotations)
    session: TrajectorySession = field(default_factory=TrajectorySession)
    environment: TrajectoryEnvironment | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "completions": [asdict(completion) for completion in self.completions],
            "messages": [asdict(message) for message in self.messages],
            "rewards": self.rewards,
            "group": self.group,
            "replica": self.replica,
            "advantages": self.advantages,
            "metadata": self.metadata,
        }
        if self.annotations.to_dict():
            result["annotations"] = self.annotations.to_dict()
        if self.session.to_dict():
            result["session"] = self.session.to_dict()
        if self.environment is not None:
            result["environment"] = self.environment.to_dict()
        return result

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Trajectory:
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
                input_tokens=usage_dict["prompt_tokens"],
                output_tokens=usage_dict["completion_tokens"],
            )
            assert usage is not None
            # Construct ChatCompletion with explicit parameters for type safety
            comps.append(
                ChatCompletion(
                    id=comp.get("id", "unknown"),
                    object=comp.get("object", "chat.completion"),
                    created=comp.get("created", 0),
                    model=comp.get("model", "unknown"),
                    usage=usage,
                    kv_transfer_params=comp.get("kv_transfer_params"),
                    choices=comp.get("choices", []),
                    prompt_logprobs=comp.get("prompt_logprobs"),
                )
            )

        annotations_data = data.get("annotations", {})
        if annotations_data:
            annotations = TrajectoryAnnotations.from_dict(annotations_data)
        else:
            annotations = TrajectoryAnnotations(
                reward=data.get("reward", data.get("rewards")),
                group=data.get("group"),
                replica=data.get("replica"),
                advantage=data.get("advantage", data.get("advantages")),
            )

        session_data = data.get("session", {})
        session = TrajectorySession.from_dict(session_data) if session_data else TrajectorySession()

        environment_data = data.get("environment")
        environment = (
            TrajectoryEnvironment.from_dict(environment_data)
            if isinstance(environment_data, dict)
            else None
        )

        result = Trajectory(
            completions=comps,
            messages=data.get("messages", []),
            rewards=data.get("rewards", 0.0),
            group=data.get("group", 0),
            replica=data.get("replica", 0),
            advantages=data.get("advantages", 0.0),
            metadata=data.get("metadata", {}),
            annotations=annotations,
            session=session,
            environment=environment,
        )
        assert result is not None
        return result

    def session_reward(self) -> float | dict[str, float] | None:
        """Return the persisted session-level reward annotation."""
        if self.annotations.reward is not None:
            return self.annotations.reward
        if self.rewards != 0.0:
            return self.rewards
        return None

    def session_status(self) -> SessionStatus:
        """Return the persisted session status with a safe default."""
        from .core.session import SessionStatus

        status_str = self.session.status or SessionStatus.PENDING.value
        try:
            return SessionStatus(status_str)
        except ValueError:
            return SessionStatus[status_str.upper()]

    def endpoint_or_default(self) -> Endpoint:
        """Return the session endpoint or an empty placeholder."""
        return self.session.endpoint or Endpoint(model="", base_url="", api_format="")

    def environment_config(self) -> EnvironmentConfig:
        """Return the durable environment configuration for the trajectory."""
        from .core.session import EnvironmentConfig

        if self.environment is None:
            return EnvironmentConfig(type="")
        return EnvironmentConfig(type=self.environment.kind, config=dict(self.environment.config))

    def environment_state(self) -> dict[str, Any] | None:
        """Return the serialized environment state, if any."""
        return self.environment.state if self.environment is not None else None

    def to_session_record(self) -> dict[str, Any]:
        """Serialize the trajectory using the session store's durable shape."""
        return {
            "session_id": self.session.session_id,
            "parent_id": self.session.parent_id,
            "branch_point": self.session.branch_point,
            "endpoint": self.endpoint_or_default().to_dict(exclude_secrets=True),
            "environment": self.environment_config().to_dict(),
            "environment_state": self.environment_state(),
            "status": self.session_status().value,
            "reward": self.session_reward(),
            "tags": dict(self.session.tags),
            "created_at": self.session.created_at or datetime.now().isoformat(),
            "updated_at": self.session.updated_at or datetime.now().isoformat(),
            "vcs": self.session.vcs,
        }

    @classmethod
    def from_session_record(
        cls, data: dict[str, Any], messages: list[Message] | None = None
    ) -> Trajectory:
        """Deserialize the session store's durable shape into a canonical trajectory."""
        from .core.session import EnvironmentConfig, SessionStatus

        environment = TrajectoryEnvironment.from_session_parts(
            EnvironmentConfig.from_dict(data["environment"]),
            data.get("environment_state"),
        )
        reward = data.get("reward")
        reward_scalar = reward if isinstance(reward, (int, float)) else 0.0
        return cls(
            messages=list(messages or []),
            rewards=reward_scalar,
            annotations=TrajectoryAnnotations(reward=reward),
            session=TrajectorySession(
                session_id=data["session_id"],
                parent_id=data.get("parent_id"),
                branch_point=data.get("branch_point"),
                endpoint=Endpoint.from_dict(data["endpoint"]),
                status=data.get("status", SessionStatus.PENDING.value),
                created_at=data.get("created_at", datetime.now().isoformat()),
                updated_at=data.get("updated_at", datetime.now().isoformat()),
                tags=data.get("tags", {}),
                vcs=data.get("vcs"),
            ),
            environment=environment,
        )

    def to_session_summary(self, message_count: int | None = None) -> SessionSummary:
        """Build a lightweight summary for listing/index views."""
        from .core.session import SessionSummary

        return SessionSummary(
            session_id=self.session.session_id or "",
            parent_id=self.session.parent_id,
            branch_point=self.session.branch_point,
            endpoint=self.endpoint_or_default(),
            environment=self.environment_config(),
            status=self.session_status(),
            reward=self.session_reward(),
            tags=dict(self.session.tags),
            created_at=self.session.created_at or datetime.now().isoformat(),
            updated_at=self.session.updated_at or datetime.now().isoformat(),
            vcs=self.session.vcs,
            message_count=message_count,
        )

    # ---------- JSONL convenience layer -----------------------------------
    def to_json(self) -> str:
        assert self is not None
        result = json.dumps(self.to_dict(), ensure_ascii=False)
        assert result is not None
        assert isinstance(result, str)
        return result

    @staticmethod
    def to_jsonl(trajectories: list[Trajectory]) -> str:
        assert trajectories is not None
        assert isinstance(trajectories, list)
        result = "\n".join(t.to_json() for t in trajectories)
        assert isinstance(result, str)
        return result

    @staticmethod
    def from_json(json_str: str) -> Trajectory:
        assert json_str is not None
        assert isinstance(json_str, str)
        assert len(json_str) > 0
        data = json.loads(json_str)
        result = Trajectory.from_dict(data)
        assert result is not None
        return result

    @staticmethod
    def from_jsonl(jsonl_str: str) -> list[Trajectory]:
        assert jsonl_str is not None
        assert isinstance(jsonl_str, str)
        result = [Trajectory.from_json(line) for line in jsonl_str.strip().splitlines() if line]
        assert isinstance(result, list)
        return result

    # ---------- disk I/O ---------------------------------------------------
    @staticmethod
    def save_jsonl(trajectories: list[Trajectory], filepath: str) -> None:
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
    def load_jsonl(filepath: str) -> list[Trajectory]:
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
    def load_jsonl_streaming(filepath: str) -> Iterator[Trajectory]:
        assert filepath is not None
        assert len(filepath) > 0
        path_obj = Path(filepath)
        assert path_obj.exists(), f"File not found: {filepath}"
        assert path_obj.is_file()

        with open(filepath, encoding="utf-8") as f:
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
    def get_completion_tokens(traj: Trajectory) -> int:
        assert traj is not None
        assert isinstance(traj, Trajectory)
        result = sum(
            Trajectory._usage_total(c.usage, "completion_tokens") for c in traj.completions
        )
        assert result >= 0
        return result

    @staticmethod
    def get_total_tokens(traj: Trajectory) -> int:
        assert traj is not None
        assert isinstance(traj, Trajectory)
        result = sum(
            Trajectory._usage_total(c.usage, "total_tokens") for c in traj.completions[-1:]
        )
        assert result >= 0
        return result

    @staticmethod
    def hash(trajectory: Trajectory) -> str:
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
    NEEDS_INPUT = "NEEDS_INPUT"  # Agent waiting for user input (interactive mode)
    INTERRUPTED = "INTERRUPTED"  # User interrupted (Escape) - can resume with driver_session_id
    ERROR = "ERROR"  # General driver error (e.g. invalid state, unexpected condition)
    END_TURN = "END_TURN"  # Claude driver: model finished its turn normally
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"  # Token/cost budget exceeded


@dataclass(frozen=True)
class ToolResult(JsonSerializable):
    tool_call_id: str = ""
    is_error: bool = False
    content: str | list[ContentBlock] = ""
    error: str | None = None
    stop_reason: StopReason | None = None
    # UI-only structured data (stripped before LLM)
    details: dict[str, Any] | None = None

    def to_summary(self, tool_call: ToolCall) -> dict[str, Any] | None:
        """Build observability summary for wide event emission.

        Extracts key fields from tool_call.args and self.details that are useful
        for debugging without re-running the tool.
        """
        summary: dict[str, Any] = {}

        # Tool-specific args worth capturing
        if tool_call.name == "bash" and "command" in tool_call.args:
            summary["command"] = tool_call.args["command"]
            if self.content:
                summary["output"] = str(self.content)
        elif tool_call.name == "write" and "path" in tool_call.args:
            summary["path"] = tool_call.args["path"]

        # Extract key metrics from details (e.g., compiled, correct for kernelbench)
        if self.details:
            for k in (
                "compiled",
                "correct",
                "speedup",
                "runtime_us",
                "error",
                "exit_code",
                "output_file",
            ):
                if k in self.details:
                    summary[k] = self.details[k]

        # Always include error info when is_error
        if self.is_error:
            if self.error:
                summary["error"] = self.error
            elif self.content:
                summary["error"] = str(self.content)

        return summary if summary else None


@dataclass(frozen=True)
class ToolConfirmResult(JsonSerializable):
    """Result of tool confirmation"""

    proceed: bool
    tool_result: ToolResult | None = None
    user_message: str | None = None


# ── Core Agent Framework Types ────────────────────────────────────────────────


@runtime_checkable
class Environment(Protocol):
    """Protocol that all environments must satisfy for composition over inheritance.

    An Environment has two coupled responsibilities that belong together:
      1. Resource setup — spin up sandboxes, fetch tasks, write files, start containers
      2. Tool contract — define what tools the agent sees and route calls to those resources

    These are coupled because the tool contract is meaningless without the resources behind it.

    ## Intended usage pattern

    A rollout/eval/training run is fully determined by three things:

        (Environment, Endpoint, DatasetRow) → Trajectory → Score

    Adding a new task means adding a new DatasetRow — not a new Environment class.
    The environment knows how to read its row's columns and do the right thing.

    The construction path is:

        row_to_state(row: dict) -> dict        # pure, defined in eval config
        Environment.deserialize(state: dict)   # I/O, defined on the environment

    `row_to_state` is a pure function in the eval config that transforms a dataset row
    into the serialized state dict the environment expects. It can be as thin as
    `{"task_id": row["task_id"]}` or as fat as cloning a repo and writing files —
    whatever the environment needs to reconstruct itself without the original row.

    `deserialize` is the single construction path for live environments, whether the
    state came from a fresh row or a mid-run checkpoint. Environments that today raise
    NotImplementedError in deserialize should instead implement it as the general
    construction path from data → live resources.

    ## Scoring

    Environments that own a verification oracle (test runner, benchmark, LLM judge)
    implement `score(trajectory) -> Score`. The eval runner calls this after the agent
    loop, while resources are still live, before calling `close()`.

    Environments without built-in scoring leave `score` unimplemented. The eval runner
    falls back to an externally-injected `score_fn(trajectory, row)`, or records no
    score for open-ended/SFT use cases.
    """

    def get_tools(self) -> list[Tool]:
        """Return available tools for this environment."""
        ...

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: AgentState,
        run_config: RunConfig,
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

    def get_tool_formatter(self, tool_name: str) -> ToolFormatter | None:
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

    def get_system_prompt(self) -> str | None:
        """Return environment-specific system prompt content.

        Optional method - environments can return None or not implement this.
        The returned string will be appended to the base system prompt.

        Use this to explain environment-specific concepts, tools, or strategies
        that the model needs to understand.

        Example (REPL environment):
            return '''
            The input context is stored in a Python variable called `context`.
            Use the repl tool to explore it programmatically.
            '''
        """
        ...

    async def initialize(self, session_id: str | None = None) -> None:
        """Eagerly initialize required environment resources before first model call.

        Optional method. Environments that need required external resources should
        prefer this over lazy setup so impossible episodes fail before spending
        model tokens.
        """
        ...

    async def on_session_start(self, session_id: str) -> None:
        """Called when an agent session starts, before any tools execute.

        Optional method - environments can use this to initialize session-specific
        resources (e.g., git worktrees, temp directories, etc.).
        Prefer `initialize()` for new environments; this hook remains as a
        compatibility shim for older implementations.

        Args:
            session_id: The session ID for this agent run
        """
        ...

    async def on_assistant_message(self, message: Message, state: AgentState) -> AgentState:
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

    async def close(self) -> None:
        """Release environment resources.

        Called by the eval runner after score() completes. Environments that own
        containers, sandboxes, or other external resources must implement this.

        Optional - environments with no external resources can omit it.
        """
        ...

    async def score(self, trajectory: Trajectory) -> Score:
        """Score the agent's trajectory using this environment's resources.

        Called after the agent loop completes, before close(). The environment
        still has access to its live resources (sandbox, container, etc.) here,
        so scoring can run tests, call verification oracles, or query the sandbox.

        Optional - environments that don't own scoring should not implement this.
        The eval runner falls back to an externally-injected score_fn when this
        method is absent, and records no score when neither is present.
        """
        ...

    async def serialize(self) -> dict:
        """Serialize environment state to dictionary.

        Must include:
            - env_kind: str (e.g., "calculator", "code_exec", "browser")
            - version: str (e.g., "1.0.0", "2.0.0")
            - ...rest of environment-specific state

        The env_kind and version enable safe restore validation:
        - Prevents restoring snapshots into wrong environment types
        - Prevents restoring incompatible versions (schema changes)

        Example:
            >>> async def serialize(self) -> dict:
            ...     return {
            ...         "env_kind": self.ENV_KIND,  # Class constant
            ...         "version": self.VERSION,    # Class constant
            ...         "history": self._history,
            ...         "state": self._state,
            ...     }
        """
        ...

    @staticmethod
    async def deserialize(data: dict) -> Environment:
        """Construct a live environment from a state dictionary.

        This is the single construction path for environments — used both for
        initial setup from a dataset row and for resuming a mid-run checkpoint.

        The intended flow from the eval config:

            row_to_state(row: dict) -> dict          # pure function in eval config
            MyEnvironment.deserialize(state: dict)   # does all I/O here

        `row_to_state` transforms a dataset row into the state dict this method
        expects. It can be a thin pointer (e.g. {"task_id": row["task_id"]}) or
        a fat snapshot that includes everything needed to reconstruct the environment
        without network calls (Dockerfile contents, test scripts, repo worktree path).

        Some row_to_state functions do real work themselves — clone a repo, create a
        worktree, run uv sync — and return the resulting paths as the fat state.
        Others are nearly identity and let deserialize do the fetching (registry model).
        Both are valid; the invariant is that deserialize receives enough state to
        construct the live environment without the original row.

        Should validate env_kind and version before restoring state from a checkpoint.
        """
        ...


# TODO: Add provider-specific max_tokens limits and validate in Endpoint.__post_init__
# Article quote: "Some providers have lower max_tokens than advertised, resulting in cut-off
# responses, even though a higher limit was set via the API request. This affects SiliconFlow,
# Friendly and Cerebras."
#
# Article quote: "Some providers have max_tokens limits which are lower than needed to evaluate
# the corresponding model. These providers were dropped completely for the given eval."
#
# Problem: No validation that max_tokens is within provider/model limits. This causes silent
# truncation where responses are cut off but we don't detect it.
#
# Fix: Add max_output_tokens to ModelInfo in models.py, then validate in __post_init__:
#     model_meta = get_model(self.provider, self.model)
#     if model_meta and model_meta.max_output_tokens:
#         assert self.max_tokens <= model_meta.max_output_tokens


@dataclass(frozen=True)
class Endpoint(JsonSerializable):
    """Endpoint configuration for model calls.

    The model string uses "provider/model-id" format (e.g., "anthropic/claude-3-5-sonnet").
    This decouples routing (base_url + api_format) from model identity.

    Examples:
        # Direct Anthropic API
        Endpoint(
            model="anthropic/claude-3-5-sonnet-20241022",
            base_url="https://api.anthropic.com/v1",
            api_format="anthropic-messages",
            api_key="sk-...",
        )

        # Same model via OpenRouter
        Endpoint(
            model="openrouter/anthropic/claude-3.5-sonnet",
            base_url="https://openrouter.ai/api/v1",
            api_format="openai-completions",
            api_key="sk-or-...",
        )

        # Vercel AI Gateway
        Endpoint(
            model="vercel/claude-3-5-sonnet",
            base_url="https://gateway.vercel.ai/v1",
            api_format="openai-completions",
            api_key="...",
        )
    """

    model: str  # "provider/model-id" format
    base_url: str  # API endpoint (e.g., "https://api.anthropic.com/v1")
    api_format: str  # Wire protocol: "openai-chat", "openai-responses", "anthropic-messages", "google-generative-ai"
    api_key: str = ""
    oauth_token: str = ""  # OAuth bearer token (takes precedence over api_key for Anthropic)
    is_claude_code_api_key: bool = (
        False  # API key created via Claude Code OAuth (requires special headers)
    )
    max_tokens: int = 8192
    # TODO: Document temperature choice for evaluations
    # Article quote: "Evaluators must also decide on the sampling parameters that models will
    # be run with, in particular the temperature... Default temperature is 0.0 for API-based
    # models [lm-evaluation-harness]... Default temperature is 0.5 [simple-evals]...
    # Default temperature is 1.0 (when invoking the script via command line) [gpt-oss]"
    #
    # Problem: Different temperature defaults make results incomparable across frameworks.
    # Our default of 1.0 increases variance. For reproducible evals, consider 0.0.
    #
    # Decision needed: Document why we chose 1.0, or change to 0.0 for deterministic evals.
    temperature: float = 1.0
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    reasoning_effort: str | None = None  # for openai
    max_completion_tokens: int | None = None  # for openai
    thinking: dict[str, Any] | None = None  # for anthropic
    # Retry configuration
    max_retries: int = 10  # Number of retries for rate limits/transient errors
    timeout: float = 120.0  # Timeout in seconds for API calls
    # Extra params merged into the raw chat request for custom servers
    extra_params: dict[str, Any] | None = None

    @property
    def provider(self) -> str:
        """Extract provider from model string (first segment before /)."""
        return self.model.split("/")[0] if "/" in self.model else self.model

    @property
    def model_id(self) -> str:
        """Extract model ID from model string (everything after first /).

        This is what should be sent to the API (e.g., 'gpt-4o', not 'openai/gpt-4o').
        """
        return self.model.split("/", 1)[1] if "/" in self.model else self.model

    @property
    def api_base(self) -> str:
        """Compatibility alias for base_url.

        DEPRECATED: Use base_url directly.
        """
        return self.base_url

    def __post_init__(self) -> None:
        """Validate endpoint configuration.

        Tiger Style: Crash loud on invalid config, explicit error messages.
        """
        # Allow empty placeholder endpoints (for dataclass defaults)
        if not self.model and not self.base_url and not self.api_format:
            return

        # Validate model format
        assert "/" in self.model, (
            f"model must be in 'provider/model-id' format, got: {self.model}\n"
            f"Examples: 'anthropic/claude-3-5-sonnet', 'openai/gpt-4o', 'openrouter/kimi-k2.5'"
        )
        # Validate base_url
        assert self.base_url, (
            "base_url is required. Examples:\n"
            "  Anthropic: https://api.anthropic.com/v1\n"
            "  OpenAI: https://api.openai.com/v1\n"
            "  OpenRouter: https://openrouter.ai/api/v1"
        )
        # Validate api_format
        valid_formats = {
            "openai-completions",
            "openai-responses",
            "anthropic-messages",
            "google-generative-ai",
        }
        assert self.api_format in valid_formats, (
            f"api_format must be one of {valid_formats}, got: {self.api_format}"
        )
        # Validate Claude thinking budget (Anthropic requires >= 1024 tokens)
        if self.thinking is not None and self.api_format == "anthropic-messages":
            assert isinstance(self.thinking, dict), (
                f"thinking must be dict, got {type(self.thinking)}"
            )
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

    def to_dict(self, exclude_secrets: bool = True) -> dict[str, Any]:
        """Serialize to dict for storage.

        Args:
            exclude_secrets: If True (default), omits api_key and oauth_token.
        """
        d = asdict(self)
        if exclude_secrets:
            d.pop("api_key", None)
            d.pop("oauth_token", None)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any], api_key: str = "", oauth_token: str = "") -> Endpoint:
        """Deserialize from dict, injecting secrets at runtime.

        Args:
            data: Dict from to_dict()
            api_key: API key to inject (not stored in session)
            oauth_token: OAuth token to inject (not stored in session)
        """
        # Remove secrets if present (they shouldn't be, but be safe)
        data = data.copy()
        data.pop("api_key", None)
        data.pop("oauth_token", None)

        # Handle legacy format: separate provider and model fields, no base_url/api_format
        if "provider" in data and "/" not in data.get("model", ""):
            # Convert legacy format to new format
            legacy_provider = data.pop("provider")
            legacy_model = data.get("model", "")
            legacy_api_base = data.pop("api_base", "")

            # Import here to avoid circular dependency
            from .models import get_api_type, get_model

            # Build new model string
            data["model"] = f"{legacy_provider}/{legacy_model}"

            # Get base_url from registry or legacy api_base
            if legacy_api_base:
                data["base_url"] = legacy_api_base
            else:
                model_meta = get_model(legacy_provider, legacy_model)
                if model_meta:
                    data["base_url"] = model_meta.base_url
                else:
                    # Fallback defaults
                    default_urls = {
                        "anthropic": "https://api.anthropic.com/v1",
                        "openai": "https://api.openai.com/v1",
                        "google": "https://generativelanguage.googleapis.com/v1beta",
                        "openrouter": "https://openrouter.ai/api/v1",
                        "groq": "https://api.groq.com/openai/v1",
                    }
                    data["base_url"] = default_urls.get(legacy_provider, "")

            # Get api_format from registry
            if "api_format" not in data:
                data["api_format"] = get_api_type(legacy_provider, legacy_model)

        return cls(**data, api_key=api_key, oauth_token=oauth_token)

    @classmethod
    def from_legacy(
        cls,
        provider: str,
        model: str,
        api_base: str = "",
        api_key: str = "",
        oauth_token: str = "",
        **kwargs: Any,
    ) -> Endpoint:
        """Create Endpoint from legacy provider+model format.

        DEPRECATED: Use Endpoint() directly with the new format.

        Args:
            provider: Legacy provider name (e.g., "anthropic", "openai")
            model: Legacy model ID (e.g., "claude-3-5-sonnet-20241022")
            api_base: Optional base URL override
            api_key: API key
            oauth_token: OAuth token
            **kwargs: Other Endpoint fields (max_tokens, temperature, etc.)

        Returns:
            Endpoint configured with derived base_url and api_format
        """
        from .models import get_api_type, get_model

        # Build new model string
        new_model = f"{provider}/{model}"

        # Get base_url
        if api_base:
            base_url = api_base
        else:
            model_meta = get_model(provider, model)
            if model_meta:
                base_url = model_meta.base_url
            else:
                default_urls = {
                    "anthropic": "https://api.anthropic.com/v1",
                    "openai": "https://api.openai.com/v1",
                    "google": "https://generativelanguage.googleapis.com/v1beta",
                    "openrouter": "https://openrouter.ai/api/v1",
                    "groq": "https://api.groq.com/openai/v1",
                    "fireworks": "https://api.fireworks.ai/inference/v1",
                    "together": "https://api.together.xyz/v1",
                    "cerebras": "https://api.cerebras.ai/v1",
                    "xai": "https://api.x.ai/v1",
                }
                base_url = default_urls.get(provider, "")

        # Get api_format
        api_format = get_api_type(provider, model)

        return cls(
            model=new_model,
            base_url=base_url,
            api_format=api_format,
            api_key=api_key,
            oauth_token=oauth_token,
            **kwargs,
        )
