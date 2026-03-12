# Creating An Environment

An environment is a protocol, not a framework base class.

Treat it as:
- plain state
- injected dependencies
- a few required methods

Do not build around inheritance, globals, or hidden singletons.

## Required Interface

Implement:
- `get_tools()`
- `exec_tool()`
- `serialize()`
- `deserialize()`

That is the real contract.

`serialize()` / `deserialize()` matter even for simple environments. They force you to be explicit about state and side effects.

## Mental Model

Prefer:

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rollouts.agents import AgentState, RunConfig
from rollouts.core import Tool, ToolCall, ToolFunction, ToolFunctionParameter, ToolResult


@dataclass
class SearchEnvironment:
    corpus_path: Path
    search_fn: Any
    max_results: int = 20
    _submitted: bool = field(default=False, init=False)

    def get_tools(self) -> list[Tool]:
        return [
            Tool(
                function=ToolFunction(
                    name="search",
                    description="Search the corpus",
                    parameters=ToolFunctionParameter(
                        properties={"query": {"type": "string"}},
                    ),
                    required=["query"],
                )
            )
        ]

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: AgentState,
        run_config: RunConfig,
        cancel_scope=None,
    ) -> ToolResult:
        if tool_call.name != "search":
            return ToolResult(tool_call_id=tool_call.id, is_error=True, error="unknown tool")
        result = await self.search_fn(tool_call.args["query"], self.max_results)
        return ToolResult(tool_call_id=tool_call.id, content=result)

    async def serialize(self) -> dict[str, Any]:
        return {
            "env_kind": "search",
            "version": "1",
            "corpus_path": str(self.corpus_path),
            "max_results": self.max_results,
            "submitted": self._submitted,
        }

    @staticmethod
    async def deserialize(data: dict[str, Any]) -> "SearchEnvironment":
        assert data["env_kind"] == "search"
        env = SearchEnvironment(
            corpus_path=Path(data["corpus_path"]),
            search_fn=build_search_fn(Path(data["corpus_path"])),
            max_results=int(data.get("max_results", 20)),
        )
        env._submitted = bool(data.get("submitted", False))
        return env
```

The class is just a container for state and injected behavior.

## Optional Hooks

Only add these if you need them:
- `get_system_prompt()`
- `on_assistant_message()`
- `on_session_start()`
- `requires_confirmation()`
- `get_status_info()`
- `get_tool_formatter()`

These are hooks, not the core abstraction.

## Guidelines

- Inject dependencies through fields or a small factory.
- Keep tool behavior in small helper functions when possible.
- Keep `serialize()` honest. If something cannot be restored, make that explicit.
- Avoid reading env vars or constructing API clients deep inside `exec_tool()`.
- Avoid hidden module-level mutable state.
- Prefer dataclasses over inheritance-heavy object models.

## Good Signs

- You can construct the environment in a test with fake dependencies.
- `serialize()` shows all important state.
- `deserialize()` makes restore behavior obvious.
- Most logic is understandable without reading the runtime.

## Bad Signs

- The environment reaches into global singletons.
- Tool execution creates clients/processes from ambient state every time.
- Serialization omits important mutable state.
- UI formatting and product concerns dominate the environment logic.
