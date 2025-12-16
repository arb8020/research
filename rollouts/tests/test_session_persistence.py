"""
Integration test for session persistence.

Tests the full flow:
1. run_agent creates session automatically
2. Tool calls are persisted per-message
3. Resume with different model works
4. Resume with disabled tools - model adapts
5. Full trajectory preserved across resume

This is an "integration test" per grugbrain: high level enough to test
correctness, low level enough to debug when it breaks.
"""

import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import trio

from rollouts import (
    Actor,
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
    Trajectory,
)
from rollouts.agents import resume_session, run_agent
from rollouts.store import FileSessionStore

# --- Test Environment: Calculator with disableable tools ---


@dataclass
class TestCalculatorEnvironment:
    """Calculator with configurable tool availability."""

    current_value: float = 0.0
    disabled_tools: set[str] | None = None

    def __post_init__(self):
        if self.disabled_tools is None:
            self.disabled_tools = set()

    async def serialize(self) -> dict:
        return {
            "current_value": self.current_value,
            "disabled_tools": list(self.disabled_tools),
        }

    @staticmethod
    async def deserialize(data: dict) -> "TestCalculatorEnvironment":
        return TestCalculatorEnvironment(
            current_value=data.get("current_value", 0.0),
            disabled_tools=set(data.get("disabled_tools", [])),
        )

    def get_tools(self) -> list[Tool]:
        all_tools = [
            Tool(
                type="function",
                function=ToolFunction(
                    name="add",
                    description="Add a number to the current value",
                    parameters=ToolFunctionParameter(
                        type="object", properties={"value": {"type": "number"}}
                    ),
                    required=["value"],
                ),
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="multiply",
                    description="Multiply the current value by a number",
                    parameters=ToolFunctionParameter(
                        type="object", properties={"value": {"type": "number"}}
                    ),
                    required=["value"],
                ),
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="get_value",
                    description="Get the current value",
                    parameters=ToolFunctionParameter(type="object", properties={}),
                    required=[],
                ),
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="complete",
                    description="Mark task complete with final answer",
                    parameters=ToolFunctionParameter(
                        type="object", properties={"answer": {"type": "number"}}
                    ),
                    required=["answer"],
                ),
            ),
        ]
        # Filter out disabled tools
        return [t for t in all_tools if t.function.name not in self.disabled_tools]

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        return False

    async def on_assistant_message(self, message: Message, state: AgentState) -> AgentState:
        return state

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: AgentState,
        run_config: RunConfig,
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult:
        name = tool_call.name
        args = tool_call.args

        if name in self.disabled_tools:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                error=f"Tool '{name}' is not available",
            )

        if name == "add":
            self.current_value += args["value"]
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Added {args['value']}. Value is now {self.current_value}",
            )
        elif name == "multiply":
            self.current_value *= args["value"]
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Multiplied by {args['value']}. Value is now {self.current_value}",
            )
        elif name == "get_value":
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Current value is {self.current_value}",
            )
        elif name == "complete":
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Task complete. Answer: {args['answer']}",
                stop_reason=StopReason.TASK_COMPLETED,
            )
        else:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                error=f"Unknown tool: {name}",
            )

    def get_tool_formatter(self, tool_name: str):
        return None


# --- Mock LLM that follows a script ---


@dataclass
class ScriptedResponse:
    """A scripted LLM response."""

    text: str | None = None
    tool_calls: list[tuple[str, dict]] | None = None  # [(name, args), ...]


class ScriptedLLM:
    """Mock LLM that returns scripted responses in order."""

    def __init__(self, responses: list[ScriptedResponse]):
        self.responses = list(responses)
        self.call_count = 0

    def get_next_response(self) -> ScriptedResponse:
        if self.call_count >= len(self.responses):
            # Default: just complete
            return ScriptedResponse(tool_calls=[("complete", {"answer": 0})])
        response = self.responses[self.call_count]
        self.call_count += 1
        return response


def make_endpoint(provider: str, model: str) -> Endpoint:
    """Create an endpoint for a specific provider/model.

    This creates real endpoint configs that would hit different API paths:
    - anthropic: Anthropic Messages API
    - openai: OpenAI Chat Completions API
    - openai (o1/o3/codex): OpenAI Responses API
    """
    api_bases = {
        "anthropic": "https://api.anthropic.com",
        "openai": "https://api.openai.com/v1",
    }
    return Endpoint(
        provider=provider,
        model=model,
        api_base=api_bases.get(provider, "https://api.openai.com/v1"),
        api_key="test-key",  # Won't actually be used since we mock rollout
    )


# --- Test Fixtures ---


async def run_with_script(
    initial_state: AgentState,
    run_config: RunConfig,
    script: list[ScriptedResponse],
) -> list[AgentState]:
    """Run agent with scripted LLM responses.

    This patches the rollout function to return scripted responses.
    """
    from rollouts import agents
    from rollouts.dtypes import TextContent, ToolCallContent

    llm = ScriptedLLM(script)
    original_rollout = agents.rollout

    async def mock_rollout(actor, on_chunk, *args, **kwargs):
        response = llm.get_next_response()

        # Build content blocks
        content = []
        if response.text:
            content.append(TextContent(text=response.text))
        if response.tool_calls:
            for i, (name, tool_args) in enumerate(response.tool_calls):
                content.append(
                    ToolCallContent(
                        id=f"call_{llm.call_count}_{i}",
                        name=name,
                        arguments=tool_args,
                    )
                )

        # Create assistant message
        new_message = Message(role="assistant", content=content if content else "")
        new_trajectory = replace(
            actor.trajectory, messages=actor.trajectory.messages + [new_message]
        )
        return replace(actor, trajectory=new_trajectory)

    # Patch and run
    agents.rollout = mock_rollout
    try:
        return await run_agent(initial_state, run_config)
    finally:
        agents.rollout = original_rollout


# --- The Integration Test ---


async def test_session_persistence_with_model_swap():
    """
    Full integration test with provider swaps:

    Phase 1: Claude 3.5 Haiku (Anthropic Messages API)
      - Start fresh, run_agent creates session
      - Model calls add(5), multiply(3) -> value = 15
      - Truncate after 3 turns

    Phase 2: GPT-5.1 Codex (OpenAI Responses API)
      - Resume session with different provider
      - Disable 'multiply' tool
      - Model calls add(10) -> value = 25
      - Truncate after 2 turns

    Phase 3: GPT-4o (OpenAI Chat Completions API)
      - Resume again with yet another model
      - Re-enable multiply
      - Model calls multiply(2) -> value = 50, then complete
      - Verify full trajectory preserved across all providers
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        store = FileSessionStore(base_dir=Path(tmp_dir))

        # =======================================================================
        # Phase 1: Claude 3.5 Haiku (Anthropic Messages API)
        # =======================================================================

        print("\n--- Phase 1: Claude 3.5 Haiku (Anthropic) ---")

        env1 = TestCalculatorEnvironment(current_value=0.0)
        endpoint1 = make_endpoint("anthropic", "claude-3-5-haiku-20241022")

        system_msg = Message(role="system", content="You are a calculator assistant.")
        user_msg = Message(role="user", content="Add 5, then multiply by 3.")

        initial_state = AgentState(
            actor=Actor(
                trajectory=Trajectory(messages=[system_msg, user_msg]),
                endpoint=endpoint1,
                tools=env1.get_tools(),
            ),
            environment=env1,
        )

        # Script: add(5), multiply(3), then text response
        script1 = [
            ScriptedResponse(tool_calls=[("add", {"value": 5})]),
            ScriptedResponse(tool_calls=[("multiply", {"value": 3})]),
            ScriptedResponse(text="Done! The value is 15."),
        ]

        def stop_after_3(state: AgentState) -> AgentState:
            if state.turn_idx >= 3:
                return replace(state, stop=StopReason.MAX_TURNS)
            return state

        run_config1 = RunConfig(
            on_chunk=lambda e: trio.lowlevel.checkpoint(),
            session_store=store,
            handle_stop=stop_after_3,
        )

        states1 = await run_with_script(initial_state, run_config1, script1)

        session_id = states1[-1].session_id
        assert session_id is not None
        print(f"✓ Session created: {session_id}")

        final_env1 = states1[-1].environment
        env1_value = (await final_env1.serialize())["current_value"]
        assert env1_value == 15.0, f"Expected 15, got {env1_value}"
        print(f"✓ Value after Haiku: {env1_value}")

        assert states1[-1].actor.endpoint.provider == "anthropic"
        assert states1[-1].actor.endpoint.model == "claude-3-5-haiku-20241022"
        print(
            f"✓ Provider: {states1[-1].actor.endpoint.provider}/{states1[-1].actor.endpoint.model}"
        )

        session1, _ = await store.get(session_id)
        msg_count_after_phase1 = len(session1.messages)
        print(f"✓ Messages persisted: {msg_count_after_phase1}")

        # =======================================================================
        # Phase 2: GPT-5.1 Codex (OpenAI Responses API - reasoning model)
        # =======================================================================

        print("\n--- Phase 2: GPT-5.1 Codex (OpenAI Responses API) ---")

        env2 = TestCalculatorEnvironment(
            current_value=15.0,
            disabled_tools={"multiply"},  # Disable multiply
        )
        endpoint2 = make_endpoint("openai", "gpt-5.1-codex")  # Responses API model

        resumed_state2 = await resume_session(session_id, store, endpoint2, env2)

        assert resumed_state2.actor.endpoint.provider == "openai"
        assert resumed_state2.actor.endpoint.model == "gpt-5.1-codex"
        print(
            f"✓ Swapped to: {resumed_state2.actor.endpoint.provider}/{resumed_state2.actor.endpoint.model}"
        )

        tool_names = [t.function.name for t in env2.get_tools()]
        assert "multiply" not in tool_names
        print(f"✓ multiply disabled, tools: {tool_names}")

        # Add user message
        user_msg2 = Message(role="user", content="Add 10 to the current value.")
        resumed_state2 = replace(
            resumed_state2,
            actor=replace(
                resumed_state2.actor,
                trajectory=Trajectory(
                    messages=resumed_state2.actor.trajectory.messages + [user_msg2]
                ),
                tools=env2.get_tools(),
            ),
        )

        # Script: add(10), then text
        script2 = [
            ScriptedResponse(tool_calls=[("add", {"value": 10})]),
            ScriptedResponse(text="Added 10. Value is now 25."),
        ]

        def stop_after_2(state: AgentState) -> AgentState:
            if state.turn_idx >= 2:
                return replace(state, stop=StopReason.MAX_TURNS)
            return state

        run_config2 = RunConfig(
            on_chunk=lambda e: trio.lowlevel.checkpoint(),
            session_store=store,
            handle_stop=stop_after_2,
        )

        states2 = await run_with_script(resumed_state2, run_config2, script2)

        final_env2 = states2[-1].environment
        env2_value = (await final_env2.serialize())["current_value"]
        assert env2_value == 25.0, f"Expected 25, got {env2_value}"
        print(f"✓ Value after Codex: {env2_value}")

        assert states2[-1].session_id == session_id
        print(f"✓ Session ID preserved: {states2[-1].session_id}")

        session2, _ = await store.get(session_id)
        msg_count_after_phase2 = len(session2.messages)
        assert msg_count_after_phase2 > msg_count_after_phase1
        print(f"✓ Messages persisted: {msg_count_after_phase2}")

        # =======================================================================
        # Phase 3: GPT-4o (OpenAI Chat Completions API)
        # =======================================================================

        print("\n--- Phase 3: GPT-4o (OpenAI Chat Completions API) ---")

        env3 = TestCalculatorEnvironment(
            current_value=25.0,
            disabled_tools=set(),  # Re-enable all tools
        )
        endpoint3 = make_endpoint("openai", "gpt-4o")  # Chat Completions model

        resumed_state3 = await resume_session(session_id, store, endpoint3, env3)

        assert resumed_state3.actor.endpoint.provider == "openai"
        assert resumed_state3.actor.endpoint.model == "gpt-4o"
        print(
            f"✓ Swapped to: {resumed_state3.actor.endpoint.provider}/{resumed_state3.actor.endpoint.model}"
        )

        tool_names3 = [t.function.name for t in env3.get_tools()]
        assert "multiply" in tool_names3
        print(f"✓ multiply re-enabled, tools: {tool_names3}")

        # Add user message
        user_msg3 = Message(role="user", content="Multiply by 2 and complete the task.")
        resumed_state3 = replace(
            resumed_state3,
            actor=replace(
                resumed_state3.actor,
                trajectory=Trajectory(
                    messages=resumed_state3.actor.trajectory.messages + [user_msg3]
                ),
                tools=env3.get_tools(),
            ),
        )

        # Script: multiply(2), complete(50)
        script3 = [
            ScriptedResponse(tool_calls=[("multiply", {"value": 2})]),
            ScriptedResponse(tool_calls=[("complete", {"answer": 50})]),
        ]

        run_config3 = RunConfig(
            on_chunk=lambda e: trio.lowlevel.checkpoint(),
            session_store=store,
        )

        states3 = await run_with_script(resumed_state3, run_config3, script3)

        final_state = states3[-1]
        assert final_state.stop == StopReason.TASK_COMPLETED
        print(f"✓ Task completed with stop reason: {final_state.stop}")

        final_env3 = final_state.environment
        env3_value = (await final_env3.serialize())["current_value"]
        assert env3_value == 50.0, f"Expected 50, got {env3_value}"
        print(f"✓ Final value: {env3_value}")

        assert final_state.session_id == session_id
        print(f"✓ Session ID preserved through all phases: {final_state.session_id}")

        # =======================================================================
        # Final verification: full trajectory
        # =======================================================================

        print("\n--- Final Verification ---")

        final_session, _ = await store.get(session_id)
        print(f"✓ Total messages: {len(final_session.messages)}")

        roles = [m.role for m in final_session.messages]
        print(f"✓ Role sequence: {roles}")

        # Count user messages (should be 3: initial + 2 added during resumes)
        user_count = roles.count("user")
        assert user_count == 3, f"Expected 3 user messages, got {user_count}"
        print(f"✓ User messages: {user_count}")

        # Verify structure
        assert roles[0] == "system"
        assert roles[1] == "user"
        assert "assistant" in roles
        assert "tool" in roles

        # Summary of the math: 0 + 5 = 5, * 3 = 15, + 10 = 25, * 2 = 50
        print(f"\n✓ Math verified: 0 → +5 → *3 → +10 → *2 = {env3_value}")

        print("\n✅ All assertions passed!")
        print(f"""
Summary:
  - Session: {session_id}
  - Phase 1: claude-3-5-haiku-20241022 (Anthropic) → value = 15
  - Phase 2: gpt-5.1-codex (OpenAI Responses) → value = 25
  - Phase 3: gpt-4o (OpenAI Chat) → value = 50
  - Total messages: {len(final_session.messages)}
  - Providers used: anthropic, openai (2 different APIs)
""")
        return True


# --- Run the test ---

if __name__ == "__main__":
    result = trio.run(test_session_persistence_with_model_swap)
    if result:
        print("\n🎉 Integration test passed!")
    else:
        print("\n❌ Integration test failed!")
        exit(1)
