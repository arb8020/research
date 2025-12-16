import argparse

import trio

from ..agents import (
    confirm_tool_with_feedback,
    handle_stop_max_turns,
    handle_tool_error,
    inject_tool_reminder,
    inject_turn_warning,
    run_agent,
    stdout_handler,
)
from ..dtypes import (
    Actor,
    AgentState,
    Endpoint,
    Environment,
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


class BinarySearchEnvironment(Environment):
    def __init__(
        self,
        range_min: int = 0,
        range_max: int = 7,
        space_size: int = 8,
        answer: int = 0,
        _turns=0,
        _correct=False,
    ):
        self.range_min: int = range_min
        self.range_max: int = range_max
        self.answer: int = answer
        self.space_size: int = space_size

        # managed during runtime
        self._turns: int = _turns
        self._correct: bool = _correct

        assert abs(range_min - range_max) + 1 == space_size, (
            f"[{range_min},{range_max}] is not {space_size}"
        )
        assert (answer >= range_min) & (answer <= range_max)

    async def serialize(self):
        return {k: v for k, v in self.__dict__.items()}

    @staticmethod
    async def deserialize(data: dict) -> "BinarySearchEnvironment":
        return BinarySearchEnvironment(**data)

    def get_tools(self) -> list[Tool]:
        return [
            Tool(
                function=ToolFunction(
                    name="guess_answer",
                    description="Guess the hidden number. You'll be told if your guess is too high or too low.",
                    parameters=ToolFunctionParameter(
                        properties={"number": {"type": "number", "description": "Your guess"}},
                    ),
                    required=["number"],
                )
            )
        ]

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        return False

    async def on_assistant_message(self, message: Message, state: AgentState) -> AgentState:
        """No feedback needed for binary search environment."""
        return state

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: AgentState,
        run_config: RunConfig,
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult:
        try:
            if tool_call.name == "guess_answer":
                guess = int(tool_call.args["number"])
                self._correct = guess == self.answer
                if self._correct:
                    return ToolResult(
                        tool_call_id=tool_call.id,
                        is_error=False,
                        content=f"CONGRATS!!!! {guess} is correct!",
                        stop_reason=StopReason.TASK_COMPLETED,
                    )
                else:
                    hint = "too high" if guess > self.answer else "too low"
                    return ToolResult(
                        tool_call_id=tool_call.id,
                        is_error=False,
                        content=f"Wrong! {guess} is {hint}. Try again!",
                    )
            else:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content=f"{tool_call.name} is not a valid tool",
                )
        except Exception as e:
            return ToolResult(tool_call_id=tool_call.id, is_error=True, content="", error=str(e))


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Binary Search Agent")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use")
    args = parser.parse_args()

    # Create run config
    run_config = RunConfig(
        on_chunk=stdout_handler,
        confirm_tool=confirm_tool_with_feedback,  # type: ignore
        handle_tool_error=handle_tool_error,
        handle_stop=handle_stop_max_turns,
        handle_no_tool=inject_tool_reminder,
        on_step_start=inject_turn_warning,
    )

    # Create the initial environment
    environment = BinarySearchEnvironment()

    # Create the initial actor state
    sys_msg = Message(
        role="system",
        content=f"You are helpful tool use agent. Your job is to guess a number in the range {environment.range_min} and {environment.range_max} inclusive.",
    )
    user_msg = Message(
        role="user",
        content="I'll take a backset while you do this task. Have fun!",
    )

    trajectory = Trajectory(messages=[sys_msg, user_msg])
    endpoint = Endpoint(provider="openai", model=args.model, api_base="https://api.openai.com/v1")
    actor = Actor(trajectory=trajectory, endpoint=endpoint)
    environment = BinarySearchEnvironment()

    # Create the initial agent state
    initial_state = AgentState(
        actor=actor,
        environment=environment,
        turn_idx=0,
        max_turns=10,
    )

    states = await run_agent(initial_state, run_config)

    print("\n" + "=" * 80)
    print("📊 Conversation Summary")
    print("=" * 80)

    final_state = states[-1]
    print(f"✅ Total turns: {final_state.turn_idx}")
    print(f"💬 Total messages: {len(final_state.actor.trajectory.messages)}")

    # Count tool calls
    tool_calls = sum(len(msg.get_tool_calls()) for msg in final_state.actor.trajectory.messages)
    print(f"🔧 Total tool calls: {tool_calls}")

    # (optionally) save trajectory to disk
    Trajectory.save_jsonl([final_state.actor.trajectory], "./trajectory.jsonl")
    if final_state.stop:
        print(f"🛑 Stopped due to: {final_state.stop.value}")


if __name__ == "__main__":
    trio.run(main)
