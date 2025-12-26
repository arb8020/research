from dataclasses import dataclass

import trio

from ..dtypes import (
    AgentState,
    Message,
    RunConfig,
    StopReason,
    Tool,
    ToolCall,
    ToolFunction,
    ToolFunctionParameter,
    ToolResult,
)


@dataclass
class CalculatorEnvironment:
    """Calculator environment with numeric operations."""

    current_value: float = 0.0

    async def serialize(self) -> dict:
        return {"env_kind": "calculator", "current_value": self.current_value}

    @staticmethod
    async def deserialize(data: dict) -> "CalculatorEnvironment":
        return CalculatorEnvironment(current_value=data["current_value"])

    def get_tools(self) -> list[Tool]:
        # Calculator is accumulator-style: starts at 0, operations modify current value
        return [
            Tool(
                type="function",
                function=ToolFunction(
                    name="add",
                    description="Add a number to the current value. The calculator starts at 0, so use add(x) to set an initial value.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={"value": {"type": "number", "description": "Number to add"}},
                    ),
                    required=["value"],
                ),
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="subtract",
                    description="Subtract a number from the current value.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "value": {"type": "number", "description": "Number to subtract"}
                        },
                    ),
                    required=["value"],
                ),
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="multiply",
                    description="Multiply the current value by a number. Note: if current value is 0, result will be 0.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "value": {"type": "number", "description": "Number to multiply by"}
                        },
                    ),
                    required=["value"],
                ),
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="divide",
                    description="Divide the current value by a number.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "value": {"type": "number", "description": "Number to divide by"}
                        },
                    ),
                    required=["value"],
                ),
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="clear",
                    description="Reset the current value to zero.",
                    parameters=ToolFunctionParameter(type="object", properties={}),
                    required=[],
                ),
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="complete_task",
                    description="Submit your final answer when the calculation is complete.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "summary": {
                                "type": "string",
                                "description": "Summary of calculations performed",
                            },
                            "final_result": {
                                "type": "number",
                                "description": "The final numerical answer",
                            },
                        },
                    ),
                    required=["summary", "final_result"],
                ),
            ),
        ]

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        # e.g. only confirm "divide" calls:
        return tool_call.name == "divide"

    async def on_assistant_message(self, message: Message, state: AgentState) -> AgentState:
        """No feedback needed for calculator environment."""
        return state

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: "AgentState",
        run_config: "RunConfig",
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult:
        """Execute tool call, mutating environment state"""
        try:
            if tool_call.name == "add":
                value = tool_call.args["value"]
                self.current_value += value
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=False,
                    content=f"Added {value}. Current value: {self.current_value}",
                )

            elif tool_call.name == "subtract":
                value = tool_call.args["value"]
                self.current_value -= value
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=False,
                    content=f"Subtracted {value}. Current value: {self.current_value}",
                )

            elif tool_call.name == "multiply":
                value = tool_call.args["value"]
                self.current_value *= value
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=False,
                    content=f"Multiplied by {value}. Current value: {self.current_value}",
                )

            elif tool_call.name == "divide":
                value = tool_call.args["value"]
                if value == 0:
                    return ToolResult(
                        tool_call_id=tool_call.id,
                        is_error=True,
                        content="",
                        error="Cannot divide by zero",
                    )
                self.current_value /= value
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=False,
                    content=f"Divided by {value}. Current value: {self.current_value}",
                )

            elif tool_call.name == "clear":
                self.current_value = 0.0
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=False,
                    content="Reset to zero. Current value: 0.0",
                )

            elif tool_call.name == "complete_task":
                summary = tool_call.args["summary"]
                final_result = tool_call.args.get("final_result", self.current_value)

                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=False,
                    content=f"Calculation task completed: {summary}. Final result: {final_result}",
                    stop_reason=StopReason.TASK_COMPLETED,  # This will stop the agent!
                )

            else:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content="",
                    error=f"Unknown operation: {tool_call.name}",
                )

        except trio.Cancelled:
            # Re-raise cancellation so agent loop can handle it
            raise
        except Exception as e:
            return ToolResult(tool_call_id=tool_call.id, is_error=True, content="", error=str(e))


# ── Entry point ───────────────────────────────────────────────────────────────


async def main() -> None:
    """Simple demo main function"""
    print("Use the simple_calculator.py example in examples/ instead!")


if __name__ == "__main__":
    trio.run(main)
