"""
REPL Environment for Recursive Language Models (RLM).

Implements the RLM paradigm where:
- Large context is stored as a Python variable, not in message history
- Model interacts via code execution (REPL) rather than seeing full context
- Recursive LLM calls (llm_query) allow semantic processing of context chunks

Two variants:
- REPLEnvironment: Tool-based interface (repl, llm_query, final_answer tools)
- MessageParsingREPLEnvironment: Parses ```repl blocks from assistant messages

Reference: https://github.com/alexzhang13/rlm-minimal
Paper: "Recursive Language Models" (Zhang & Khattab, 2025)
"""

import contextlib
import io
import re
from dataclasses import dataclass, field, replace
from typing import Any

import trio

from ..dtypes import (
    AgentState,
    Endpoint,
    Message,
    RunConfig,
    StopReason,
    Tool,
    ToolCall,
    ToolFunction,
    ToolFunctionParameter,
    ToolResult,
)

# Safe builtins for REPL execution
SAFE_BUILTINS = {
    # Types
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bytes": bytes,
    # Functions
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "isinstance": isinstance,
    "type": type,
    "repr": repr,
    "print": print,  # Captured via redirect_stdout
    # String methods accessed via str
    "chr": chr,
    "ord": ord,
    # Iteration
    "iter": iter,
    "next": next,
    # None/True/False
    "None": None,
    "True": True,
    "False": False,
}

# Explicitly blocked - dangerous operations
BLOCKED_BUILTINS = {"eval", "exec", "compile", "open", "input", "__import__"}

MAX_OUTPUT_SIZE = 50_000  # 50KB max stdout capture


def _create_namespace(
    context: str,
    llm_query_fn: Any,
    rlm_query_fn: Any,
) -> dict[str, Any]:
    """Create a sandboxed namespace for REPL execution."""
    namespace = dict(SAFE_BUILTINS)
    namespace["context"] = context
    namespace["llm_query"] = llm_query_fn
    namespace["rlm_query"] = rlm_query_fn
    # Allow re for regex operations on context
    namespace["re"] = re
    return namespace


def _exec_code(code: str, namespace: dict[str, Any]) -> tuple[str, bool]:
    """Execute code in namespace, return (stdout, had_error).

    Returns:
        Tuple of (captured stdout, whether an error occurred)
    """
    stdout = io.StringIO()
    had_error = False

    try:
        with contextlib.redirect_stdout(stdout):
            # Handle imports separately (they need to go in globals)
            lines = code.strip().split("\n")
            import_lines = []
            other_lines = []

            for line in lines:
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    import_lines.append(line)
                else:
                    other_lines.append(line)

            # Execute imports (limited to safe modules)
            if import_lines:
                import_code = "\n".join(import_lines)
                try:
                    exec(import_code, namespace)
                except ImportError as e:
                    print(f"Import error: {e}")
                    had_error = True

            # Execute rest of code
            if other_lines:
                main_code = "\n".join(other_lines)
                # Try as expression first (for auto-print of last value)
                try:
                    result = eval(main_code, namespace)
                    if result is not None:
                        print(repr(result))
                except SyntaxError:
                    # Not an expression, execute as statements
                    exec(main_code, namespace)

    except Exception as e:
        stdout.write(f"Error: {type(e).__name__}: {e}\n")
        had_error = True

    output = stdout.getvalue()
    # Truncate if too long
    if len(output) > MAX_OUTPUT_SIZE:
        output = output[:MAX_OUTPUT_SIZE] + f"\n... (truncated, {len(output)} total chars)"

    return output, had_error


@dataclass
class REPLEnvironment:
    """RLM-style environment with tool-based interface.

    The model uses three tools:
    - repl: Execute Python code with `context` variable available
    - llm_query: Query a sub-LLM for semantic processing of text chunks
    - final_answer: Submit the final answer (stops the agent)

    Args:
        context: The large input context (stored as Python variable, not in messages)
        sub_endpoint: Endpoint for llm_query sub-calls
        recursive: If True, llm_query spawns full RLM; if False, simple LM call
        max_depth: Maximum recursion depth for nested RLM calls
    """

    context: str
    sub_endpoint: Endpoint | None = None
    recursive: bool = False
    max_depth: int = 2
    _current_depth: int = 0

    # Internal state
    _namespace: dict[str, Any] = field(default_factory=dict)
    _final_answer: str | None = None
    _initialized: bool = False

    def __post_init__(self) -> None:
        if not self._initialized:
            self._namespace = _create_namespace(
                context=self.context,
                llm_query_fn=self._sync_llm_query,
                rlm_query_fn=self._sync_rlm_query,
            )
            self._initialized = True

    def get_name(self) -> str:
        return "repl"

    def get_status_info(self) -> dict[str, str] | None:
        return {
            "context_size": f"{len(self.context):,} chars",
            "depth": f"{self._current_depth}/{self.max_depth}",
        }

    def get_tools(self) -> list[Tool]:
        return [
            Tool(
                type="function",
                function=ToolFunction(
                    name="repl",
                    description=(
                        "Execute Python code in a REPL environment. "
                        "The variable `context` contains the full input text. "
                        "Use this to peek at context (context[:1000]), search (re.findall), "
                        "slice, filter, or process the context programmatically."
                    ),
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "code": {
                                "type": "string",
                                "description": "Python code to execute. Output is captured from print() and expression results.",
                            },
                        },
                    ),
                    required=["code"],
                ),
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="llm_query",
                    description=(
                        "Query a language model with a prompt. Use this for semantic tasks "
                        "that require understanding text, like 'classify this paragraph' or "
                        "'extract the key entities from this chunk'. The prompt should be "
                        "self-contained - include any context the LLM needs."
                    ),
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "prompt": {
                                "type": "string",
                                "description": "The prompt to send to the LLM. Include all necessary context.",
                            },
                        },
                    ),
                    required=["prompt"],
                ),
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="final_answer",
                    description=(
                        "Submit your final answer. Call this when you have determined "
                        "the answer to the original query."
                    ),
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "answer": {
                                "type": "string",
                                "description": "Your final answer to the query.",
                            },
                        },
                    ),
                    required=["answer"],
                ),
            ),
        ]

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        return False

    def get_tool_formatter(self, tool_name: str) -> None:
        # Could add custom formatters for REPL output
        return None

    async def on_session_start(self, session_id: str) -> None:
        pass

    async def on_assistant_message(self, message: Message, state: AgentState) -> AgentState:
        """No message parsing in tool-based variant."""
        return state

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: AgentState,
        run_config: RunConfig,
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult:
        """Execute REPL tools."""
        try:
            if tool_call.name == "repl":
                return await self._exec_repl(tool_call)
            elif tool_call.name == "llm_query":
                return await self._exec_llm_query(tool_call, run_config)
            elif tool_call.name == "final_answer":
                return self._exec_final_answer(tool_call)
            else:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content="",
                    error=f"Unknown tool: {tool_call.name}",
                )
        except trio.Cancelled:
            # Re-raise cancellation so agent loop can handle it
            raise
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"{type(e).__name__}: {e}",
            )

    async def _exec_repl(self, tool_call: ToolCall) -> ToolResult:
        """Execute Python code in the REPL namespace."""
        code = tool_call.args.get("code", "")
        if not code.strip():
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error="No code provided",
            )

        output, had_error = _exec_code(code, self._namespace)

        return ToolResult(
            tool_call_id=tool_call.id,
            content=output or "(no output)",
            is_error=had_error,
        )

    async def _exec_llm_query(self, tool_call: ToolCall, run_config: RunConfig) -> ToolResult:
        """Execute sub-LLM query."""
        prompt = tool_call.args.get("prompt", "")
        if not prompt.strip():
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error="No prompt provided",
            )

        if self.sub_endpoint is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error="No sub_endpoint configured for llm_query",
            )

        # Make the sub-call
        try:
            if self.recursive and self._current_depth < self.max_depth:
                # Recursive RLM call
                result = await self._async_rlm_query(prompt, run_config)
            else:
                # Simple LM call
                result = await self._async_llm_query(prompt)

            return ToolResult(
                tool_call_id=tool_call.id,
                content=result,
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"LLM query failed: {e}",
            )

    def _exec_final_answer(self, tool_call: ToolCall) -> ToolResult:
        """Store final answer and signal completion."""
        answer = tool_call.args.get("answer", "")
        self._final_answer = answer

        return ToolResult(
            tool_call_id=tool_call.id,
            content=f"Final answer submitted: {answer}",
            stop_reason=StopReason.TASK_COMPLETED,
        )

    # Sync wrappers for use inside exec'd code
    def _sync_llm_query(self, prompt: str) -> str:
        """Synchronous llm_query for use inside REPL code.

        Note: This uses trio.from_thread which requires being called from
        a thread spawned by trio. For now, returns placeholder.
        """
        # TODO: Implement proper async bridge
        # Could use trio.from_thread.run or similar
        return "[llm_query not available inside exec - use llm_query tool instead]"

    def _sync_rlm_query(self, prompt: str) -> str:
        """Synchronous rlm_query for use inside REPL code."""
        return (
            "[rlm_query not available inside exec - use llm_query tool with recursive=True instead]"
        )

    async def _async_llm_query(self, prompt: str) -> str:
        """Make a simple LLM call (not recursive)."""
        from ..dtypes import Actor, Trajectory
        from ..providers import get_provider_function

        assert self.sub_endpoint is not None

        actor = Actor(
            trajectory=Trajectory(messages=[Message(role="user", content=prompt)]),
            endpoint=self.sub_endpoint,
            tools=[],
        )

        # Collect response
        response_text = ""

        async def collect_response(event: Any) -> None:
            nonlocal response_text
            from ..dtypes import TextDelta

            if isinstance(event, TextDelta):
                response_text += event.delta

        provider_func = get_provider_function(
            self.sub_endpoint.provider,
            self.sub_endpoint.model,
        )

        await provider_func(actor, collect_response)

        return response_text

    async def _async_rlm_query(self, prompt: str, run_config: RunConfig) -> str:
        """Make a recursive RLM call (spawns new REPLEnvironment)."""
        from ..agents import run_agent
        from ..dtypes import Actor, Trajectory

        assert self.sub_endpoint is not None

        # Create child RLM environment
        child_env = REPLEnvironment(
            context=prompt,  # The prompt becomes the context for child
            sub_endpoint=self.sub_endpoint,
            recursive=self.recursive,
            max_depth=self.max_depth,
            _current_depth=self._current_depth + 1,
        )

        # Create agent state
        actor = Actor(
            trajectory=Trajectory(
                messages=[
                    Message(role="user", content="Process the context and provide your answer."),
                ]
            ),
            endpoint=self.sub_endpoint,
            tools=child_env.get_tools(),
        )

        child_state = AgentState(
            actor=actor,
            environment=child_env,
        )

        # Run child agent
        child_run_config = replace(
            run_config,
            session_store=None,  # Don't persist child sessions
        )

        await run_agent(child_state, child_run_config)

        # Return final answer from child
        return child_env._final_answer or "(no answer from recursive RLM)"

    async def serialize(self) -> dict:
        return {
            "env_kind": "repl",
            "version": "1.0.0",
            "context": self.context,
            "sub_endpoint": self.sub_endpoint.to_json() if self.sub_endpoint else None,
            "recursive": self.recursive,
            "max_depth": self.max_depth,
            "current_depth": self._current_depth,
            "final_answer": self._final_answer,
            # Note: namespace not serialized (contains lambdas)
        }

    @staticmethod
    async def deserialize(data: dict) -> "REPLEnvironment":
        assert data.get("env_kind") == "repl"

        sub_endpoint = None
        if data.get("sub_endpoint"):
            sub_endpoint = Endpoint.from_json(data["sub_endpoint"])

        env = REPLEnvironment(
            context=data["context"],
            sub_endpoint=sub_endpoint,
            recursive=data.get("recursive", False),
            max_depth=data.get("max_depth", 2),
            _current_depth=data.get("current_depth", 0),
        )
        env._final_answer = data.get("final_answer")
        return env


@dataclass
class MessageParsingREPLEnvironment(REPLEnvironment):
    """RLM environment that parses ```repl blocks from assistant messages.

    Instead of using formal tool calls, the model writes code in markdown
    code blocks which are automatically extracted and executed.

    This variant uses on_assistant_message to:
    1. Extract ```repl or ```python code blocks
    2. Execute them in the REPL namespace
    3. Inject output as a user message
    4. Check for FINAL(answer) markers

    The model should be prompted to use:
    - ```repl or ```python blocks for code execution
    - FINAL(answer) to submit final answer
    - llm_query("prompt") inside code for sub-queries
    """

    def get_tools(self) -> list[Tool]:
        """No formal tools - we parse from message content."""
        return []

    async def on_assistant_message(self, message: Message, state: AgentState) -> AgentState:
        """Parse code blocks from assistant message and execute them."""
        content = message.content if isinstance(message.content, str) else ""

        if not content:
            return state

        # Check for FINAL(answer) or FINAL_VAR(varname)
        final_match = re.search(r"FINAL\(([^)]+)\)", content)
        if final_match:
            self._final_answer = final_match.group(1).strip().strip("\"'")
            return replace(state, stop=StopReason.TASK_COMPLETED)

        final_var_match = re.search(r"FINAL_VAR\((\w+)\)", content)
        if final_var_match:
            var_name = final_var_match.group(1)
            self._final_answer = str(
                self._namespace.get(var_name, f"Variable '{var_name}' not found")
            )
            return replace(state, stop=StopReason.TASK_COMPLETED)

        # Extract ```repl or ```python code blocks
        code_blocks = re.findall(r"```(?:repl|python)\n(.*?)```", content, re.DOTALL)

        if not code_blocks:
            return state

        # Execute all code blocks
        all_output = []
        for code in code_blocks:
            output, _had_error = _exec_code(code.strip(), self._namespace)
            if output.strip():
                all_output.append(output)

        if not all_output:
            return state

        # Inject REPL output as user message
        combined_output = "\n".join(all_output)
        feedback = Message(role="user", content=f"[REPL Output]\n{combined_output}")

        new_trajectory = replace(
            state.actor.trajectory,
            messages=[*state.actor.trajectory.messages, feedback],
        )

        return replace(
            state,
            actor=replace(state.actor, trajectory=new_trajectory),
        )

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: AgentState,
        run_config: RunConfig,
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult:
        """No tools in message-parsing variant."""
        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=True,
            content="",
            error="MessageParsingREPLEnvironment does not use tools. Use ```repl code blocks instead.",
        )
