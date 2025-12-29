"""
Environment composition for combining multiple environments.

Allows combining tools from different environments into a single environment.
For example, combining LocalFilesystemEnvironment with REPLEnvironment to get
both file tools and RLM capabilities.

Usage:
    env = compose(
        LocalFilesystemEnvironment(working_dir=Path.cwd()),
        REPLEnvironment(context=huge_doc, sub_endpoint=endpoint),
    )
    # Tools: read, write, edit, bash, repl, llm_query, final_answer
"""

from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

import trio

from ..dtypes import (
    AgentState,
    Environment,
    Message,
    RunConfig,
    Tool,
    ToolCall,
    ToolResult,
)

# Environment registry: maps env_kind -> deserialize function
# Lazily populated to avoid circular imports
_ENVIRONMENT_REGISTRY: dict[str, Callable[[dict], Coroutine[Any, Any, Environment]]] = {}


def _get_environment_registry() -> dict[str, Callable[[dict], Coroutine[Any, Any, Environment]]]:
    """Get the environment registry, populating it lazily on first call."""
    if not _ENVIRONMENT_REGISTRY:
        # Import all environments and register them
        from .ask_user import AskUserQuestionEnvironment
        from .binary_search import BinarySearchEnvironment
        from .calculator import CalculatorEnvironment
        from .coding import LocalFilesystemEnvironment
        from .git_worktree import GitWorktreeEnvironment
        from .no_tools import BasicEnvironment
        from .repl import REPLEnvironment

        _ENVIRONMENT_REGISTRY.update({
            "coding": LocalFilesystemEnvironment.deserialize,
            "git_worktree": GitWorktreeEnvironment.deserialize,
            "repl": REPLEnvironment.deserialize,
            "calculator": CalculatorEnvironment.deserialize,
            "basic": BasicEnvironment.deserialize,
            "binary_search": BinarySearchEnvironment.deserialize,
            "ask_user": AskUserQuestionEnvironment.deserialize,
        })

        # Optional environments with heavy dependencies (lazy import)
        try:
            from .browsing import BrowsingEnvironment

            _ENVIRONMENT_REGISTRY["browsing"] = BrowsingEnvironment.deserialize
        except ImportError:
            pass

        try:
            from .chess_puzzle import ChessPuzzleEnvironment

            _ENVIRONMENT_REGISTRY["chess_puzzle"] = ChessPuzzleEnvironment.deserialize
        except ImportError:
            pass

    return _ENVIRONMENT_REGISTRY


@dataclass
class ComposedEnvironment:
    """Compose multiple environments into one.

    This merges tools from all environments and routes exec_tool calls
    to the appropriate environment based on tool name.

    TODO: Environment compatibility validation
    ------------------------------------
    Not all environments compose well. Known issues to handle:

    1. STATE CONFLICTS: Environments may have conflicting state assumptions.
       - Example: Two environments both trying to manage working_dir
       - Example: REPLEnvironment's namespace vs another env's namespace
       - Mitigation: Add explicit conflict detection in __post_init__

    2. ON_ASSISTANT_MESSAGE ORDERING: When multiple environments implement
       on_assistant_message, the order matters. Currently we chain them,
       but this may cause issues:
       - Example: REPLEnvironment parses ```repl blocks, another env parses ```sql
       - Example: One env stops early, preventing others from running
       - Mitigation: Consider parallel execution or explicit ordering config

    3. SERIALIZATION: Each environment serializes independently, but there's
       no guarantee the combined state is coherent on restore.
       - Mitigation: Add version/compatibility checks in deserialize

    4. TOOL NAME COLLISIONS: Currently we raise on collision, but some tools
       might intentionally want to override others.
       - Mitigation: Add override=True option or namespacing (env.tool_name)

    5. CONFIRMATION LOGIC: requires_confirmation may conflict if environments
       have different security models.
       - Mitigation: Union (any env requiring confirmation) vs intersection

    For now, this is experimental and works for simple cases like
    LocalFilesystemEnvironment + REPLEnvironment.
    """

    environments: list[Environment]
    _tool_to_env: dict[str, Environment] = field(default_factory=dict, repr=False)
    _initialized: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        # Build routing table: tool_name -> environment
        for env in self.environments:
            for tool in env.get_tools():
                name = tool.function.name
                if name in self._tool_to_env:
                    existing_env = self._tool_to_env[name]
                    raise ValueError(
                        f"Tool name collision: '{name}' is provided by both "
                        f"{type(existing_env).__name__} and {type(env).__name__}. "
                        f"Consider using different environments or namespacing tools."
                    )
                self._tool_to_env[name] = env

    def get_name(self) -> str:
        names = [env.get_name() for env in self.environments if hasattr(env, "get_name")]
        return "+".join(names) if names else "composed"

    def get_status_info(self) -> dict[str, str] | None:
        """Merge status info from all environments."""
        combined: dict[str, str] = {}
        for env in self.environments:
            if hasattr(env, "get_status_info"):
                info = env.get_status_info()
                if info:
                    # TODO: Handle key collisions in status info
                    combined.update(info)
        return combined if combined else None

    def get_system_prompt(self) -> str | None:
        """Combine system prompts from all environments."""
        prompts = []
        for env in self.environments:
            if hasattr(env, "get_system_prompt"):
                prompt = env.get_system_prompt()
                if prompt:
                    prompts.append(prompt)
        return "\n\n".join(prompts) if prompts else None

    def get_tools(self) -> list[Tool]:
        """Return union of all tools from composed environments."""
        tools = []
        for env in self.environments:
            tools.extend(env.get_tools())
        return tools

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        """Check if any environment requires confirmation for this tool.

        Uses union semantics: if ANY environment requires confirmation, confirm.
        """
        env = self._tool_to_env.get(tool_call.name)
        if env is None:
            return False
        return env.requires_confirmation(tool_call)

    def get_tool_formatter(self, tool_name: str) -> Any:
        """Get formatter from the environment that owns this tool."""
        env = self._tool_to_env.get(tool_name)
        if env is None:
            return None
        if hasattr(env, "get_tool_formatter"):
            return env.get_tool_formatter(tool_name)
        return None

    async def on_session_start(self, session_id: str) -> None:
        """Notify all environments of session start."""
        for env in self.environments:
            if hasattr(env, "on_session_start"):
                await env.on_session_start(session_id)

    async def on_assistant_message(self, message: Message, state: AgentState) -> AgentState:
        """Chain on_assistant_message through all environments.

        TODO: This chains sequentially. Consider:
        - Parallel execution for independent environments
        - Explicit ordering configuration
        - Early-exit vs continue-after-stop semantics
        """
        current_state = state
        for env in self.environments:
            if hasattr(env, "on_assistant_message"):
                current_state = await env.on_assistant_message(message, current_state)
                # TODO: Should we continue after one env signals stop?
                # Current behavior: stop chaining, return immediately
                if current_state.stop:
                    break
        return current_state

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: AgentState,
        run_config: RunConfig,
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult:
        """Route tool execution to the appropriate environment."""
        env = self._tool_to_env.get(tool_call.name)
        if env is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Unknown tool: {tool_call.name}",
            )
        return await env.exec_tool(tool_call, current_state, run_config, cancel_scope)

    async def serialize(self) -> dict[str, Any]:
        """Serialize all composed environments."""
        return {
            "env_kind": "composed",
            "version": "1.0.0",
            "environments": [await env.serialize() for env in self.environments],
        }

    @staticmethod
    async def deserialize(data: dict[str, Any]) -> "ComposedEnvironment":
        """Deserialize composed environment.

        Uses the environment registry to deserialize each sub-environment
        based on its env_kind field.
        """
        registry = _get_environment_registry()
        environments = []

        for env_data in data["environments"]:
            env_kind = env_data.get("env_kind")
            if env_kind is None:
                raise ValueError(f"Environment data missing 'env_kind' field: {env_data.keys()}")

            deserialize_fn = registry.get(env_kind)
            if deserialize_fn is None:
                raise ValueError(
                    f"Unknown environment kind '{env_kind}'. Known kinds: {list(registry.keys())}"
                )

            env = await deserialize_fn(env_data)
            environments.append(env)

        return ComposedEnvironment(environments=environments)


def compose(*environments: Environment) -> ComposedEnvironment:
    """Compose multiple environments into one.

    Example:
        env = compose(
            LocalFilesystemEnvironment(working_dir=Path.cwd()),
            REPLEnvironment(context=huge_doc),
        )

    Args:
        *environments: Environments to compose

    Returns:
        ComposedEnvironment with merged tools

    Raises:
        ValueError: If tool names collide between environments
    """
    if len(environments) == 0:
        raise ValueError("compose() requires at least one environment")
    if len(environments) == 1:
        # No need to wrap single environment
        return environments[0]  # type: ignore
    return ComposedEnvironment(environments=list(environments))
