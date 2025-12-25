"""
REPL Environment for Recursive Language Models (RLM).

Implements the RLM paradigm where:
- Large context is stored as a Python variable, not in message history
- Model interacts via code execution (REPL) rather than seeing full context
- Recursive LLM calls (llm_query) allow semantic processing of context chunks
- Recursive agent calls (agent) spawn sub-agents for complex exploration

Two variants:
- REPLEnvironment: Tool-based interface (repl, llm_query, agent, final_answer tools)
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


# ── Tool Formatters ───────────────────────────────────────────────────────────


def _get_text_output(result: dict | None) -> str:
    """Extract text output from tool result."""
    if not result:
        return ""
    content = result.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_blocks = [c for c in content if isinstance(c, dict) and c.get("type") == "text"]
        return "\n".join(c.get("text", "") for c in text_blocks if c.get("text"))
    return ""


def format_final_answer(
    tool_name: str, args: dict, result: dict | None, expanded: bool, theme=None
) -> str:
    """Format final_answer tool - show answer prominently."""
    answer = args.get("answer", "")

    # Show a nice header
    text = "✅ final_answer()"

    if answer:
        # Show full answer, nicely formatted
        text += "\n⎿ "
        lines = answer.strip().split("\n")
        if len(lines) == 1:
            text += lines[0]
        else:
            # Multi-line answer - indent nicely
            max_lines = len(lines) if expanded else 20
            for i, line in enumerate(lines[:max_lines]):
                if i == 0:
                    text += line
                else:
                    text += "\n  " + line
            if len(lines) > max_lines:
                text += f"\n  ... ({len(lines) - max_lines} more lines)"

    return text


def format_repl(
    tool_name: str, args: dict, result: dict | None, expanded: bool, theme=None
) -> str:
    """Format repl tool - show code and output."""
    code = args.get("code", "")

    # Truncate long code for display
    code_lines = code.strip().split("\n")
    max_code_lines = 10 if not expanded else len(code_lines)
    display_code = "\n".join(code_lines[:max_code_lines])
    if len(code_lines) > max_code_lines:
        display_code += f"\n# ... ({len(code_lines) - max_code_lines} more lines)"

    text = f"repl()\n```python\n{display_code}\n```"

    if result:
        output = _get_text_output(result).strip()
        if output:
            is_error = result.get("isError", False)
            lines = output.split("\n")
            max_lines = len(lines) if expanded else 10
            display_lines = lines[:max_lines]

            summary = "Error" if is_error else "Output"
            text += f"\n⎿ {summary}:"
            for line in display_lines:
                if theme and is_error:
                    text += "\n  " + (theme.diff_removed_fg(line) if hasattr(theme, 'diff_removed_fg') else line)
                else:
                    text += "\n  " + line
            if len(lines) > max_lines:
                text += f"\n  ... ({len(lines) - max_lines} more lines)"

    return text


def format_agent(
    tool_name: str, args: dict, result: dict | None, expanded: bool, theme=None
) -> str:
    """Format agent tool - show task and result."""
    task = args.get("task", "")
    context = args.get("context", "")

    # Truncate task/context for display
    task_preview = task[:100] + "..." if len(task) > 100 else task
    context_len = len(context)

    text = f"agent(task={repr(task_preview)}, context=<{context_len:,} chars>)"

    if result:
        output = _get_text_output(result).strip()
        if output:
            is_error = result.get("isError", False)
            lines = output.split("\n")
            max_lines = len(lines) if expanded else 15

            summary = "Sub-agent failed" if is_error else "Sub-agent result"
            text += f"\n⎿ {summary}:"
            for line in lines[:max_lines]:
                text += "\n  " + line
            if len(lines) > max_lines:
                text += f"\n  ... ({len(lines) - max_lines} more lines)"

    return text


def format_llm_query(
    tool_name: str, args: dict, result: dict | None, expanded: bool, theme=None
) -> str:
    """Format llm_query tool - show prompt summary and response."""
    prompt = args.get("prompt", "")

    # Show truncated prompt
    prompt_preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
    prompt_preview = prompt_preview.replace("\n", "\\n")
    text = f"llm_query({repr(prompt_preview)})"

    if result:
        output = _get_text_output(result).strip()
        if output:
            is_error = result.get("isError", False)
            lines = output.split("\n")
            max_lines = len(lines) if expanded else 10

            summary = "Error" if is_error else "Response"
            text += f"\n⎿ {summary}:"
            for line in lines[:max_lines]:
                text += "\n  " + line
            if len(lines) > max_lines:
                text += f"\n  ... ({len(lines) - max_lines} more lines)"

    return text


def _create_namespace(
    context: str,
    llm_query_fn: Any,
    agent_fn: Any,
) -> dict[str, Any]:
    """Create a sandboxed namespace for REPL execution."""
    namespace = dict(SAFE_BUILTINS)
    namespace["context"] = context
    namespace["llm_query"] = llm_query_fn
    namespace["agent"] = agent_fn
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

    The model uses these tools:
    - repl: Execute Python code with `context` variable available
    - llm_query: Quick LLM call for semantic tasks on chunks (no tools)
    - agent: Spawn a sub-agent with full tool access for complex exploration
    - final_answer: Submit the final answer (stops the agent)

    Args:
        context: The large input context (stored as Python variable, not in messages)
        sub_endpoint: Endpoint for llm_query and agent sub-calls
        max_depth: Maximum recursion depth for nested agent calls
        max_agent_turns: Maximum turns for sub-agents (default 15)
    """

    context: str
    sub_endpoint: Endpoint | None = None
    max_depth: int = 3
    max_agent_turns: int = 15
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
                agent_fn=self._sync_agent,
            )
            self._initialized = True

    def get_name(self) -> str:
        return "repl"

    def get_status_info(self) -> dict[str, str] | None:
        return {
            "context_size": f"{len(self.context):,} chars",
            "depth": f"{self._current_depth}/{self.max_depth}",
        }

    def get_system_prompt(self) -> str | None:
        """Return RLM-specific system prompt explaining the REPL paradigm."""
        return """## REPL Environment for Large Context Processing

The input context is stored in a Python variable called `context`. It may be very large (millions of characters). You NEVER see the full context in your messages - instead, you explore it programmatically.

### Available Tools

**repl** - Execute Python code to explore the context:
```python
context              # the full input text as a string
len(context)         # get the size
context[:1000]       # peek at the beginning
re.findall(pattern, context)  # search with regex
[l for l in context.split('\\n') if 'keyword' in l]  # filter lines
```

Inside `repl`, you also have access to:
- `llm_query(prompt)` - Quick LLM call for semantic tasks (classification, extraction, summarization). Include the text to analyze in the prompt.
- `agent(task, context)` - Spawn a sub-agent with its own REPL to explore a subset of the context. Use for complex tasks requiring multiple steps.

**llm_query** - Direct tool call for semantic tasks (same as calling from repl)

**agent** - Spawn a sub-agent for complex exploration:
- The sub-agent gets its own `context` variable (what you pass)
- It can use repl, llm_query, and even spawn its own sub-agents
- Use when a task requires exploration, not just a quick answer

**final_answer** - Submit your answer when done

### When to use what

| Task | Tool |
|------|------|
| Check size, peek at structure | `repl` with Python |
| Find patterns, filter lines | `repl` with regex |
| Classify/summarize a chunk | `llm_query(prompt + chunk)` |
| Complex multi-step exploration | `agent(task, subset)` |

### Example workflow

```python
# 1. Peek at structure
print(f"Size: {len(context)}, First 1000 chars:")
print(context[:1000])

# 2. Find relevant sections
matches = re.findall(r'## (.+)', context)
print(f"Sections: {matches}")

# 3. Quick semantic task on a chunk
chunk = context[5000:10000]
summary = llm_query(f"Summarize this:\\n{chunk}")

# 4. Delegate complex exploration to sub-agent
auth_code = context[20000:50000]
findings = agent("Find security vulnerabilities and explain each", auth_code)
```

### Important

- Start by peeking at the context structure before diving deep
- Use Python for structural tasks, llm_query for semantic tasks
- Use agent() when a subtask needs its own exploration
- Always call final_answer when you have the answer"""

    def get_tools(self) -> list[Tool]:
        tools = [
            Tool(
                type="function",
                function=ToolFunction(
                    name="repl",
                    description=(
                        "Execute Python code in a REPL environment. "
                        "The variable `context` contains the full input text. "
                        "You also have `llm_query(prompt)` for semantic tasks and "
                        "`agent(task, context)` to spawn sub-agents for complex exploration."
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
                        "Quick LLM call for semantic tasks (classify, extract, summarize). "
                        "The prompt should be self-contained - include any text to analyze. "
                        "For complex tasks requiring exploration, use agent() instead."
                    ),
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "prompt": {
                                "type": "string",
                                "description": "The prompt including any text to analyze.",
                            },
                        },
                    ),
                    required=["prompt"],
                ),
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="agent",
                    description=(
                        "Spawn a sub-agent with its own REPL environment for complex exploration. "
                        "The sub-agent gets the context you provide, can use repl/llm_query/agent, "
                        "and returns its final answer. Use for tasks requiring multiple steps."
                    ),
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "task": {
                                "type": "string",
                                "description": "What you want the sub-agent to do.",
                            },
                            "context": {
                                "type": "string",
                                "description": "The context for the sub-agent to explore (e.g., a subset of your context).",
                            },
                        },
                    ),
                    required=["task", "context"],
                ),
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="final_answer",
                    description="Submit your final answer when you have solved the task.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "answer": {
                                "type": "string",
                                "description": "Your final answer.",
                            },
                        },
                    ),
                    required=["answer"],
                ),
            ),
        ]
        return tools

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        return False

    def get_tool_formatter(self, tool_name: str):
        """Return custom formatters for REPL tools."""
        formatters = {
            "repl": format_repl,
            "llm_query": format_llm_query,
            "agent": format_agent,
            "final_answer": format_final_answer,
        }
        return formatters.get(tool_name)

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
                return await self._exec_repl(tool_call, run_config)
            elif tool_call.name == "llm_query":
                return await self._exec_llm_query(tool_call)
            elif tool_call.name == "agent":
                return await self._exec_agent(tool_call, run_config)
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

    async def _exec_repl(self, tool_call: ToolCall, run_config: RunConfig) -> ToolResult:
        """Execute Python code in the REPL namespace."""
        code = tool_call.args.get("code", "")
        if not code.strip():
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error="No code provided",
            )

        # Get trio token so we can call back from thread
        trio_token = trio.lowlevel.current_trio_token()

        # Create sync llm_query that bridges back to async
        def sync_llm_query(prompt: str) -> str:
            """Quick LLM call for semantic tasks."""
            try:
                return trio.from_thread.run(
                    self._async_llm_query,
                    prompt,
                    trio_token=trio_token,
                )
            except Exception as e:
                return f"[llm_query error: {e}]"

        # Create sync agent that bridges back to async
        def sync_agent(task: str, context: str | None = None) -> str:
            """Spawn a sub-agent for complex exploration."""
            if self._current_depth >= self.max_depth:
                return f"[max depth {self.max_depth} reached - use llm_query for simpler tasks]"
            if context is None:
                context = self._namespace.get("context", "")
            try:
                return trio.from_thread.run(
                    self._async_agent,
                    task,
                    context,
                    run_config,
                    trio_token=trio_token,
                )
            except Exception as e:
                return f"[agent error: {e}]"

        # Update namespace with working functions
        self._namespace["llm_query"] = sync_llm_query
        self._namespace["agent"] = sync_agent

        # Run exec in a thread so trio.from_thread.run works
        output, had_error = await trio.to_thread.run_sync(_exec_code, code, self._namespace)

        return ToolResult(
            tool_call_id=tool_call.id,
            content=output or "(no output)",
            is_error=had_error,
        )

    async def _exec_llm_query(self, tool_call: ToolCall) -> ToolResult:
        """Execute sub-LLM query (tool interface)."""
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

        try:
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

    async def _exec_agent(self, tool_call: ToolCall, run_config: RunConfig) -> ToolResult:
        """Execute agent sub-call (tool interface)."""
        task = tool_call.args.get("task", "")
        context = tool_call.args.get("context", "")

        if not task.strip():
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error="No task provided",
            )

        if not context.strip():
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error="No context provided for sub-agent",
            )

        if self._current_depth >= self.max_depth:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Max depth {self.max_depth} reached - use llm_query instead",
            )

        if self.sub_endpoint is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error="No sub_endpoint configured for agent",
            )

        try:
            result = await self._async_agent(task, context, run_config)
            return ToolResult(
                tool_call_id=tool_call.id,
                content=result,
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Agent failed: {e}",
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

    # Sync placeholders - replaced at runtime in _exec_repl with thread-bridged versions
    def _sync_llm_query(self, prompt: str) -> str:
        """Placeholder - replaced with thread-bridged version in _exec_repl."""
        return "[llm_query: call from within repl tool execution]"

    def _sync_agent(self, task: str, context: str | None = None) -> str:
        """Placeholder - replaced with thread-bridged version in _exec_repl."""
        return "[agent: call from within repl tool execution]"

    async def _async_llm_query(self, prompt: str) -> str:
        """Make a simple LLM call (no tools, no recursion)."""
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

    async def _async_agent(self, task: str, context: str, run_config: RunConfig) -> str:
        """Spawn a child agent with full tool access."""
        from ..agents import (
            compose_handlers,
            handle_stop_max_turns,
            handle_stop_on_empty_message,
            run_agent,
        )
        from ..dtypes import Actor, Trajectory

        assert self.sub_endpoint is not None

        # Create child RLM environment with the provided context
        child_env = REPLEnvironment(
            context=context,
            sub_endpoint=self.sub_endpoint,
            max_depth=self.max_depth,
            max_agent_turns=self.max_agent_turns,
            _current_depth=self._current_depth + 1,
        )

        # Create agent state
        actor = Actor(
            trajectory=Trajectory(
                messages=[
                    Message(role="system", content=child_env.get_system_prompt()),
                    Message(role="user", content=task),
                ]
            ),
            endpoint=self.sub_endpoint,
            tools=child_env.get_tools(),
        )

        child_state = AgentState(
            actor=actor,
            environment=child_env,
        )

        # Build run config for child agent
        child_run_config = RunConfig(
            on_chunk=run_config.on_chunk,  # Inherit streaming (or could silence)
            handle_stop=compose_handlers([
                handle_stop_max_turns(self.max_agent_turns),
                handle_stop_on_empty_message(),
            ]),
            session_store=None,  # Don't persist child sessions
        )

        await run_agent(child_state, child_run_config)

        # Return final answer from child
        if child_env._final_answer:
            return child_env._final_answer
        else:
            return "(sub-agent did not provide a final answer)"

    async def serialize(self) -> dict:
        return {
            "env_kind": "repl",
            "version": "1.1.0",  # Bumped for agent() addition
            "context": self.context,
            "sub_endpoint": self.sub_endpoint.to_json() if self.sub_endpoint else None,
            "max_depth": self.max_depth,
            "max_agent_turns": self.max_agent_turns,
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
            max_depth=data.get("max_depth", 3),
            max_agent_turns=data.get("max_agent_turns", 15),
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
    - llm_query("prompt") inside code for quick semantic tasks
    - agent(task, context) inside code for complex exploration
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

        # Setup thread-bridged functions for code execution
        trio_token = trio.lowlevel.current_trio_token()

        def sync_llm_query(prompt: str) -> str:
            """Quick LLM call for semantic tasks."""
            try:
                return trio.from_thread.run(
                    self._async_llm_query,
                    prompt,
                    trio_token=trio_token,
                )
            except Exception as e:
                return f"[llm_query error: {e}]"

        def sync_agent(task: str, context: str | None = None) -> str:
            """Spawn a sub-agent for complex exploration."""
            if self._current_depth >= self.max_depth:
                return f"[max depth {self.max_depth} reached]"
            if context is None:
                context = self._namespace.get("context", "")
            try:
                # Note: We don't have run_config here, so we build a minimal one
                from ..agents import (
                    compose_handlers,
                    handle_stop_max_turns,
                    handle_stop_on_empty_message,
                )
                minimal_config = RunConfig(
                    handle_stop=compose_handlers([
                        handle_stop_max_turns(self.max_agent_turns),
                        handle_stop_on_empty_message(),
                    ]),
                )
                return trio.from_thread.run(
                    self._async_agent,
                    task,
                    context,
                    minimal_config,
                    trio_token=trio_token,
                )
            except Exception as e:
                return f"[agent error: {e}]"

        self._namespace["llm_query"] = sync_llm_query
        self._namespace["agent"] = sync_agent

        # Execute all code blocks in a thread (so trio.from_thread.run works)
        all_output = []
        for code in code_blocks:
            output, _had_error = await trio.to_thread.run_sync(
                _exec_code, code.strip(), self._namespace
            )
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
