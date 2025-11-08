import json
from pathlib import Path
from enum import Enum
import os
import asyncio
from abc import ABC
from dataclasses import dataclass, field, asdict, fields
from typing import Any, Dict, List, Optional, Mapping, Union, TypeVar, Type, Iterator, Callable, Awaitable, Tuple, Protocol, runtime_checkable
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

T = TypeVar('T', bound='SerialDataclass')

class SerialDataclass:
    """Base class for dataclasses with JSON serialization support"""
    
    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(asdict(self), ensure_ascii=False) #type:ignore
    
    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Deserialize from JSON string using dacite"""
        data = json.loads(json_str)
        return dacite.from_dict(data_class=cls, data=data)
    
    def to_path(self, path: str | Path) -> None:
        """Save to file as JSON"""
        Path(path).write_text(self.to_json(), encoding="utf-8")
    
    @classmethod
    def from_path(cls: Type[T], path: str | Path) -> T:
        """Load from JSON file"""
        json_str = Path(path).read_text(encoding="utf-8")
        return cls.from_json(json_str)

class ToolCall(SerialDataclass):
    id: str
    name: str
    args: Mapping[str, Any]

@dataclass(frozen=True)
class StreamChunk(SerialDataclass):
    """A chunk of data emitted during streaming"""
    kind: str  # "token", "tool_call_complete", "tool_result", etc.
    data: Mapping[str, Any]

@dataclass(frozen=True)
class Message(SerialDataclass):
    role: str
    content: Optional[str | List[Dict[str, Any]]]  # str for text, List[Dict] for vision (OpenAI format)
    reasoning_content: Optional[Any] = None
    thinking_content: Optional[str] = None
    thinking_signature: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None

@dataclass(frozen=True)
class Usage(SerialDataclass):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[Any] = None

@dataclass(frozen=True)
class Logprob(SerialDataclass):
    token: str
    logprob: float
    bytes: List[int]
    top_logprobs: List[float]

@dataclass(frozen=True)
class Logprobs(SerialDataclass):
    content: List[Logprob] = field(default_factory=list)

@dataclass(frozen=True)
class Choice(SerialDataclass):
    index: int
    message: Message
    finish_reason: str
    logprobs: Optional[Logprobs] = None
    stop_reason: Optional[Any] = None


@dataclass(frozen=True)
class TokenInfo(SerialDataclass):
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
class ChatCompletion(SerialDataclass):
    id: str
    object: str
    created: int
    model: str
    usage: Usage
    kv_transfer_params: Optional[Any] = None
    choices: List[Choice] = field(default_factory=list)
    prompt_logprobs: Optional[List[PromptLogprob]] = None

@dataclass
class Trajectory: # TODO: Port to serial
    completions: List[ChatCompletion] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)   # debugging only
    rewards: float = 0.0
    group: int = 0
    replica: int = 0
    advantages: float = 0.0     # scalar; broadcast later if needed

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Trajectory":
        """Rebuild nested dataclasses so type hints stay correct."""
        comps: List[ChatCompletion] = []
        for comp in data.get("completions", []):
            usage_dict = comp.get("usage", {})
            usage = Usage(
                prompt_tokens       = usage_dict["prompt_tokens"],
                completion_tokens   = usage_dict["completion_tokens"],
                total_tokens        = usage_dict["total_tokens"],
            )
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

        return Trajectory(
            completions=comps,
            messages=data.get("messages", []),
            rewards=data.get("rewards", 0.0),
            group=data.get("group", 0),
            replica=data.get("replica", 0),
            advantages=data.get("advantages", 0.0),
        )

    # ---------- JSONL convenience layer -----------------------------------
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def to_jsonl(trajectories: List["Trajectory"]) -> str:
        return "\n".join(t.to_json() for t in trajectories)

    @staticmethod
    def from_json(json_str: str) -> "Trajectory":
        return Trajectory.from_dict(json.loads(json_str))

    @staticmethod
    def from_jsonl(jsonl_str: str) -> List["Trajectory"]:
        return [Trajectory.from_json(line) for line in jsonl_str.strip().splitlines() if line]

    # ---------- disk I/O ---------------------------------------------------
    @staticmethod
    def save_jsonl(trajectories: List["Trajectory"], filepath: str) -> None:
        Path(filepath).write_text(Trajectory.to_jsonl(trajectories), encoding="utf-8")

    @staticmethod
    def load_jsonl(filepath: str) -> List["Trajectory"]:
        return Trajectory.from_jsonl(Path(filepath).read_text(encoding="utf-8"))

    @staticmethod
    def load_jsonl_streaming(filepath: str) -> Iterator["Trajectory"]:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    yield Trajectory.from_json(line)

    # ---------- helpers that work pre-/post-serialisation ------------------
    @staticmethod
    def _usage_total(usage: Union[Usage, Dict[str, Any]], key: str) -> int:
        if isinstance(usage, Usage):
            return getattr(usage, key, 0)
        return usage.get(key, 0)

    @staticmethod
    def get_completion_tokens(traj: "Trajectory") -> int:
        return sum(Trajectory._usage_total(c.usage, "completion_tokens") for c in traj.completions)

    @staticmethod
    def get_total_tokens(traj: "Trajectory") -> int:
        return sum(Trajectory._usage_total(c.usage, "total_tokens") for c in traj.completions[-1:])

    @staticmethod
    def hash(trajectory: "Trajectory") -> str:
        import hashlib
        """Generate a hash for a single trajectory."""
        traj_dict = asdict(trajectory)
        traj_str = json.dumps(traj_dict, sort_keys=True)
        return hashlib.sha256(traj_str.encode()).hexdigest()[:16]

@dataclass(frozen=True)
class ToolFunctionParameter(SerialDataclass):
    properties: Dict[str, Any]
    type: str = "object"

@dataclass(frozen=True)
class ToolFunction(SerialDataclass):
    name: str
    description: str
    parameters: ToolFunctionParameter
    required: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class Tool(SerialDataclass):
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
class ToolResult(SerialDataclass):
    call_id: str = ""
    ok: bool = False
    content: str = ""
    error: Optional[str] = None
    stop_reason: Optional[StopReason] = None

@dataclass(frozen=True)
class ToolConfirmResult(SerialDataclass):
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

    async def serialize(self) -> dict:
        """Serialize environment state to dictionary."""
        ...

    @staticmethod
    async def deserialize(data: dict) -> 'Environment':
        """Deserialize environment from dictionary."""
        ...

@dataclass(frozen=True)
class Endpoint(SerialDataclass):
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
    # Extra params merged into the raw chat request for custom servers
    extra_params: Optional[Dict[str, Any]] = None

@dataclass(frozen=True)
class Actor(SerialDataclass):
    trajectory: Trajectory
    endpoint: Endpoint
    tools: List[Tool] = field(default_factory=list)

@dataclass(frozen=True)
class AgentState:
    actor: Actor
    environment: Environment
    max_turns: int
    stop: Optional[StopReason] = None
    turn_idx: int = 0
    pending_tool_calls: List[ToolCall] = field(default_factory=list)
    next_tool_idx: int = 0  # Which tool we're about to process
    timestamp: str = datetime.now(timezone.utc).isoformat() + 'Z'

# Forward declarations for RunConfig (needs to be after AgentState but before default handlers)
async def default_stdin_handler(prompt: str) -> str:
    """Default input handler using asyncio.to_thread for non-blocking input."""
    return await asyncio.to_thread(input, prompt)

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
