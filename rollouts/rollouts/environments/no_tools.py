"""
BasicEnvironment - Simple environment with no tools for clean AI conversations.

This is useful for single-shot analysis tasks where you want the AI to provide
text responses without access to calculator, search, or other tools that might
confuse the conversation.

Example usage:
    from rollouts.environments import BasicEnvironment
    environment = BasicEnvironment()
"""

from dataclasses import dataclass

from ..dtypes import AgentState, Message, RunConfig, Tool, ToolCall, ToolFormatter, ToolResult


@dataclass
class BasicEnvironment:
    """
    Simple environment with no tools - just for clean AI responses.
    
    This environment provides no tools to the AI agent, making it suitable for:
    - Analysis tasks
    - Text generation
    - Single-shot conversations
    - Any scenario where tools would be distracting
    
    The AI will receive prompts and generate text responses without access to
    external tools or functions.
    """
    
    def get_tools(self) -> list[Tool]:
        """Return empty tool list - no tools available."""
        return []
    
    async def serialize(self) -> dict:
        """Serialize environment state (empty for this simple environment)."""
        return {}
    
    @staticmethod
    async def deserialize(data: dict) -> 'BasicEnvironment':
        """Deserialize environment state."""
        return BasicEnvironment()

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        """No tools, so no confirmation needed."""
        return False

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: AgentState,
        run_config: RunConfig,
        checkpoint_store=None,
        cancel_scope=None,
    ) -> ToolResult:
        """No tools available in basic environment."""
        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=True,
            content="No tools available in basic environment"
        )

    async def on_assistant_message(self, message: Message, state: AgentState) -> AgentState:
        """No feedback needed for basic environment."""
        return state

    def get_tool_formatter(self, tool_name: str) -> ToolFormatter | None:
        """Return optional TUI formatter for tools.

        BasicEnvironment has no tools, so returns None.

        Example for custom environments:
            def get_tool_formatter(self, tool_name: str) -> ToolFormatter | None:
                if tool_name == "my_custom_tool":
                    def formatter(tool_name, args, result, expanded):
                        text = f"my_custom_tool(param={args.get('param')})"
                        if result:
                            text += f"\\n⎿ Custom result: {result}"
                        return text
                    return formatter
                return None  # Use default formatter
        """
        return None


# Backward compatibility alias
NoToolsEnvironment = BasicEnvironment