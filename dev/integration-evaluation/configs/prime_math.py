"""Config for evaluating model on math-python with Prime verifiers.

Usage:
    cd ~/research/dev/integration-evaluation
    python local.py configs/prime_math.py

This evaluates models on mathematical reasoning using Python code execution.
The environment provides a Python sandbox where the model can execute code
to solve math problems.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List
import trio_asyncio

from rollouts.dtypes import Endpoint, EvalConfig, Message, Tool, ToolFunction, ToolFunctionParameter


def prepare_messages(sample_data: Dict[str, Any]) -> List[Message]:
    """Prepare messages for math-python environment.

    math-python uses 'prompt' field as a list of message dicts.
    """
    prompt = sample_data.get("prompt", [])
    return [Message(role=msg["role"], content=msg["content"]) for msg in prompt]


@dataclass(frozen=True)
class IntegrationEvalConfig:
    """Configuration for math-python evaluation."""

    # Model configuration
    # Option 1: OpenAI GPT-4.1 Mini (fast and cheap)
    model_name: str = "gpt-4.1-mini"
    provider: str = "openai"
    api_base: str = "https://api.openai.com/v1"
    api_key_env_var: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 2048

    # Option 2: Gemini (10 req/min quota - use max_concurrent=1)
    # model_name: str = "gemini-2.0-flash-exp"
    # provider: str = "openai"
    # api_base: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    # api_key_env_var: str = "GEMINI_API_KEY"

    # Option 3: Anthropic
    # model_name: str = "claude-3-5-sonnet-20241022"
    # provider: str = "anthropic"
    # api_base: str = "https://api.anthropic.com"
    # api_key_env_var: str = "ANTHROPIC_API_KEY"

    # Prime environment configuration
    env_name: str = "math-python"
    num_samples: int = 5  # Start small to test

    # Evaluation configuration
    eval_name: str = "prime_math_eval"
    max_turns: int = 20  # Math problems may need multiple reasoning steps
    max_concurrent: int = 4

    # Output configuration
    output_dir: Path = Path("results/integration-evaluation")
    verbose: bool = True

    def to_endpoint(self) -> Endpoint:
        """Convert to rollouts Endpoint."""
        import os

        api_key = os.getenv(self.api_key_env_var, "")
        if not api_key and self.provider != "sglang":
            print(f"⚠️  Warning: {self.api_key_env_var} not set in environment")

        return Endpoint(
            provider=self.provider,
            model=self.model_name,
            api_base=self.api_base,
            api_key=api_key,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    def to_eval_config(self, reward_fn) -> EvalConfig:
        """Convert to rollouts EvalConfig."""
        return EvalConfig(
            reward_fn=reward_fn,
            max_turns=self.max_turns,
            max_concurrent=self.max_concurrent,
            max_samples=self.num_samples,
            output_dir=self.output_dir,
            eval_name=self.eval_name,
            verbose=self.verbose,
        )


# Export config instance
config = IntegrationEvalConfig()


class MathPythonEnvironment:
    """Adapter for math-python environment.

    Unlike ACEBench which uses text-based actions, math-python uses OpenAI-style
    tool calling with a 'python' tool that executes code in a sandbox.

    The environment provides:
    - A `python(code: str)` tool for code execution
    - Persistent Python REPL state across turns
    - Output/error feedback from code execution
    """

    def __init__(self, prime_env, initial_state):
        """Initialize with Prime environment and initial state.

        Args:
            prime_env: The Prime PythonEnv instance
            initial_state: Initial state dict from prime_env.init_state + setup_state
        """
        self.prime_env = prime_env
        self.state = initial_state
        self.messages = []

    def get_tools(self) -> List[Tool]:
        """Return OpenAI-style tools from Prime environment.

        math-python provides oai_tools which we can convert to rollouts Tools.
        """
        if not hasattr(self.prime_env, 'oai_tools') or not self.prime_env.oai_tools:
            return []

        tools = []
        for oai_tool in self.prime_env.oai_tools:
            if oai_tool['type'] == 'function':
                func = oai_tool['function']

                # Convert to rollouts Tool format
                tool_function = ToolFunction(
                    name=func['name'],
                    description=func['description'],
                    parameters=ToolFunctionParameter(
                        properties=func['parameters'].get('properties', {}),
                        type=func['parameters'].get('type', 'object')
                    ),
                    required=func['parameters'].get('required', [])
                )

                tool = Tool(function=tool_function, type='function')
                tools.append(tool)

        return tools

    async def exec_tool(self, tool_call, current_state, run_config, checkpoint_store=None):
        """Execute a tool call using Prime's env_response.

        Args:
            tool_call: ToolCall object with name and args (dict)
            current_state: Current AgentState
            run_config: RunConfig
            checkpoint_store: Optional checkpoint store

        Returns:
            ToolResult with execution result
        """
        import json
        from rollouts.dtypes import ToolResult

        try:
            # Create an assistant message with the tool call
            # Prime expects OpenAI format with tool_calls
            assistant_msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.args)  # Convert dict to JSON string
                    }
                }]
            }
            self.messages.append(assistant_msg)

            # Call Prime's env_response to execute the tool
            async with trio_asyncio.open_loop():
                updated_messages, updated_state = await trio_asyncio.aio_as_trio(
                    self.prime_env.env_response
                )(self.messages, self.state)

            # Update state
            self.state = updated_state

            # Extract new tool response messages
            # Prime returns tool messages in OpenAI format
            new_messages = updated_messages[len(self.messages):]
            self.messages = updated_messages

            # Return the tool result content
            if new_messages and new_messages[0].get('role') == 'tool':
                content = new_messages[0].get('content', '')
                return ToolResult(
                    call_id=tool_call.id,
                    ok=True,
                    content=content
                )

            return ToolResult(
                call_id=tool_call.id,
                ok=False,
                content="",
                error="No tool response received from environment"
            )
        except Exception as e:
            return ToolResult(
                call_id=tool_call.id,
                ok=False,
                content="",
                error=str(e)
            )

    async def add_message_and_get_response(self, message: Message) -> List[Message]:
        """Add assistant message and get environment response.

        For math-python, this is used when the agent sends a regular message
        without tool calls (rare, usually they just call the python tool).

        Args:
            message: The assistant's message

        Returns:
            List of response messages from the environment
        """
        # Convert to OpenAI format
        oai_msg = {"role": message.role, "content": message.content or ""}
        self.messages.append(oai_msg)

        # Call env_response
        async with trio_asyncio.open_loop():
            updated_messages, updated_state = await trio_asyncio.aio_as_trio(
                self.prime_env.env_response
            )(self.messages, self.state)

        self.state = updated_state

        # Extract new messages
        new_messages = updated_messages[len(self.messages):]
        self.messages = updated_messages

        # Convert back to rollouts format
        return [Message(role=msg["role"], content=msg.get("content"))
                for msg in new_messages]

    async def serialize(self):
        """Serialize environment state.

        For stateful environments like Prime's StatefulToolEnv, we maintain
        the same environment instance across tool calls rather than creating
        fresh instances. So we just return self as the serialized form.
        """
        return {"env_instance": self}

    @staticmethod
    async def deserialize(data):
        """Deserialize environment state.

        Since we're maintaining the same environment instance for stateful
        execution, just return the original instance.
        """
        return data["env_instance"]

    async def cleanup(self):
        """Cleanup sandbox when done."""
        sandbox_id = self.state.get("sandbox_id")
        if sandbox_id:
            # Use trio_asyncio bridge to call Prime's async destroy_sandbox
            async with trio_asyncio.open_loop():
                await trio_asyncio.aio_as_trio(
                    self.prime_env.destroy_sandbox
                )(sandbox_id)


async def create_environment(prime_env, sample_data):
    """Factory function to create environment instances for each sample.

    Args:
        prime_env: The Prime PythonEnv instance
        sample_data: Sample data dict containing prompt, answer, etc.

    Returns:
        MathPythonEnvironment instance initialized for this sample
    """
    # Initialize state using trio_asyncio bridge
    async with trio_asyncio.open_loop():
        state = await trio_asyncio.aio_as_trio(prime_env.init_state)(
            prompt=sample_data.get("prompt", []),
            completion=[],
            answer=sample_data.get("answer", ""),
            task="default",
            info=sample_data,
            example_id=sample_data.get("example_id", 0)
        )

        # Setup environment-specific state (also async)
        state = await trio_asyncio.aio_as_trio(prime_env.setup_state)(state)

    return MathPythonEnvironment(prime_env, state)
