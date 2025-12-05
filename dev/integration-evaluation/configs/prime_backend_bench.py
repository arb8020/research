"""Config for evaluating model on backend-bench with Prime verifiers.

Usage:
    cd ~/research/dev/integration-evaluation
    python local.py configs/prime_backend_bench.py

This evaluates models on GPU kernel generation tasks. The environment tests
the model's ability to generate correct and fast GPU kernels (Triton/CUDA)
that pass PyTorch verification tests.

Configuration:
    - GPU: Explicitly set to 'local' (change to 'A100', 'T4', etc. for Modal)
    - Suite: 'torchbench' (real workload traces)
    - Print interception: Enabled to capture backend-bench print() output
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import trio_asyncio
from rollouts.dtypes import (
    AgentState,
    Endpoint,
    EvalConfig,
    Message,
    Tool,
    ToolFunction,
    ToolFunctionParameter,
)

from shared import intercept_prints

logger = logging.getLogger(__name__)


def prepare_messages(sample_data: dict[str, Any]) -> list[Message]:
    """Prepare messages for backend-bench environment.

    backend-bench uses 'prompt' field as a list of message dicts.
    """
    prompt = sample_data.get("prompt", [])
    return [Message(role=msg["role"], content=msg["content"]) for msg in prompt]


@dataclass(frozen=True)
class IntegrationEvalConfig:
    """Configuration for backend-bench evaluation."""

    # Hardware target (for remote deployment)
    gpu_ranks: list[int] = field(default_factory=lambda: [5])  # Which GPUs to use on remote node
    device_type: str = "cuda"  # cuda|cpu

    # Model configuration
    # Option 1: OpenAI GPT-4.1 Mini (fast and cheap)
    model_name: str = "gpt-4.1-mini"
    provider: str = "openai"
    api_base: str = "https://api.openai.com/v1"
    api_key_env_var: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 8192  # Kernel code can be longer

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
    env_name: str = "siro/backend-bench"
    num_samples: int = 1  # REDUCED TO 1 for debugging! (136 total available)

    # Backend-bench specific parameters (explicit!)
    # These are passed to verifiers.load_environment()
    # Available options from backend_bench.load_environment signature:
    #   - gpu: 'local' | 'T4' | 'L4' | 'A100' | 'H100' | 'H200' | 'B200'
    #   - suite: 'smoke' | 'opinfo' | 'torchbench' | 'facto'
    #   - ops: list[str] | None (specific ops to test, or None for all)
    #   - num_turns: int (feedback loop iterations)
    #   - feedback_loop: 'until_correct' | 'until_max_turns' | 'none'
    backend_bench_gpu: str = "local"  # Use 'local' for remote node's GPU (no Modal)
    backend_bench_suite: str = "torchbench"
    backend_bench_ops: list[str] | None = None
    backend_bench_num_turns: int = 1
    backend_bench_feedback_loop: str = "until_max_turns"

    # Evaluation configuration
    experiment_name: str = "backend_bench_eval"  # For result directory naming
    eval_name: str = "prime_backend_bench_eval"
    max_turns: int = 1  # Backend-bench is single-turn: generate once, test once
    max_concurrent: int = 4

    # Output configuration
    output_dir: Path = Path("results")  # Results saved here on remote (then synced)
    verbose: bool = True
    show_progress: bool = True  # Enable nested progress bars (outer: samples, inner: turns)

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
            show_progress=self.show_progress,
        )


# Export config instance
config = IntegrationEvalConfig()


class BackendBenchEnvironment:
    """Adapter for backend-bench environment with print interception.

    Similar to math-python, backend-bench is a StatefulToolEnv that uses
    OpenAI-style tool calling for code execution and testing.

    The environment provides:
    - Tools for generating and testing GPU kernels (Triton/CUDA)
    - Persistent state across turns for iterative development
    - Feedback on kernel correctness and performance

    Print Interception:
    - backend-bench uses print() instead of logging
    - We intercept these prints and convert to logger.info()
    - This ensures all output is captured in JSONL logs
    """

    def __init__(self, prime_env, initial_state):
        """Initialize with Prime environment and initial state.

        Args:
            prime_env: The Prime BackendBenchEnvironment instance
            initial_state: Initial state dict from prime_env.init_state + setup_state
        """
        self.prime_env = prime_env
        self.state = initial_state
        self.messages = []
        self.logger = logging.getLogger(f"{__name__}.BackendBenchEnvironment")

    def get_tools(self) -> list[Tool]:
        """Return OpenAI-style tools from Prime environment.

        backend-bench provides oai_tools which we convert to rollouts Tools.
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

            # Call Prime's env_response with print interception
            with intercept_prints(self.logger):
                updated_messages, updated_state = await trio_asyncio.run_aio_coroutine(
                    self.prime_env.env_response(self.messages, self.state)
                )

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

    async def on_assistant_message(self, message: Message, state: AgentState) -> AgentState:
        """Called after each assistant message to execute code and provide feedback.

        Backend-bench parses code from assistant messages, executes it, and returns
        feedback on correctness and performance.

        Args:
            message: The assistant's message (contains generated code)
            state: Current agent state

        Returns:
            Updated state with environment feedback injected
        """
        from dataclasses import replace

        # Log state BEFORE env_response (concise)
        self.logger.info("=" * 60)
        self.logger.info("BEFORE env_response:")
        self.logger.info(f"  self.state keys: {list(self.state.keys())}")
        has_results = 'results' in self.state and self.state.get('results')
        has_best = 'best_result' in self.state and self.state.get('best_result')
        self.logger.info(f"  has results: {has_results}, has best_result: {has_best}")
        self.logger.info(f"  messages so far: {len(self.messages)}")

        # Convert to OpenAI format
        oai_msg = {"role": message.role, "content": message.content or ""}
        self.messages.append(oai_msg)

        # Call Prime's env_response with print interception
        with intercept_prints(self.logger):
            async with trio_asyncio.open_loop():
                updated_messages, updated_state = await trio_asyncio.aio_as_trio(
                    self.prime_env.env_response
                )(self.messages, self.state)

        self.state = updated_state

        # Log state AFTER env_response (concise and readable)
        self.logger.info("AFTER env_response:")
        self.logger.info(f"  updated_state keys: {list(updated_state.keys())}")

        if updated_state.get('results'):
            results = updated_state['results']
            self.logger.info(f"  Number of results: {len(results)}")

            # Log summary of last result (not the full object!)
            if results:
                last_result = results[-1]
                self.logger.info("  Last result summary:")
                self.logger.info(f"    - Correctness score: {last_result.correctness_score}")
                self.logger.info(f"    - Performance score: {last_result.performance_score}")
                self.logger.info(f"    - Is correct: {last_result.is_correct}")
                self.logger.info(f"    - Correctness tests: {len(last_result.correctness_results)} tests")
                self.logger.info(f"    - Performance tests: {len(last_result.performance_results)} tests")

                # Log any errors (truncated)
                for i, test in enumerate(last_result.correctness_results):
                    if test.error_msg:
                        error_preview = test.error_msg[:100] + "..." if len(test.error_msg) > 100 else test.error_msg
                        self.logger.info(f"    - Correctness test {i}: ERROR - {error_preview}")
                    else:
                        self.logger.info(f"    - Correctness test {i}: {'PASS' if test.is_correct else 'FAIL'}")
        else:
            self.logger.info("  results: NOT FOUND")

        best_result = updated_state.get('best_result')
        if best_result:
            self.logger.info(f"  best_result: correctness={best_result.correctness_score}, perf={best_result.performance_score}")
        else:
            self.logger.info("  best_result: NOT FOUND")

        self.logger.info("=" * 60)

        # Extract new messages (feedback from environment)
        new_messages = updated_messages[len(self.messages):]
        self.messages = updated_messages

        # Convert back to rollouts format and inject into trajectory
        if new_messages:
            feedback_messages = [Message(role=msg["role"], content=msg.get("content"))
                               for msg in new_messages]

            # Inject feedback into state's trajectory AND store updated backend-bench state
            # The reward function needs access to the state with results/best_result
            updated_metadata = {
                **state.actor.trajectory.metadata,
                "backend_bench_state": self.state  # Store the updated Prime state
            }

            updated_trajectory = replace(
                state.actor.trajectory,
                messages=[*state.actor.trajectory.messages, *feedback_messages],
                metadata=updated_metadata
            )
            updated_actor = replace(state.actor, trajectory=updated_trajectory)
            return replace(state, actor=updated_actor)

        # No feedback, store state anyway for reward function
        updated_metadata = {
            **state.actor.trajectory.metadata,
            "backend_bench_state": self.state
        }
        updated_trajectory = replace(
            state.actor.trajectory,
            metadata=updated_metadata
        )
        updated_actor = replace(state.actor, trajectory=updated_trajectory)
        return replace(state, actor=updated_actor)

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
        """Cleanup resources when done."""
        # Backend-bench may have GPU resources or sandboxes to clean up
        if hasattr(self.prime_env, 'cleanup'):
            with intercept_prints(self.logger):
                await trio_asyncio.run_aio_coroutine(
                    self.prime_env.cleanup()
                )


async def create_environment(prime_env, sample_data):
    """Factory function to create environment instances for each sample.

    Args:
        prime_env: The Prime BackendBenchEnvironment instance
        sample_data: Sample data dict containing prompt, question, etc.

    Returns:
        BackendBenchEnvironment instance initialized for this sample
    """
    # Initialize state using trio_asyncio bridge
    state = await trio_asyncio.run_aio_coroutine(
        prime_env.init_state(
            prompt=sample_data.get("prompt", []),
            completion=[],
            answer=sample_data.get("answer", ""),
            task=sample_data.get("task", "backend_bench"),
            info=sample_data.get("info", {}),  # Pass just the info dict, not all sample_data
            example_id=sample_data.get("example_id", 0)
        )
    )

    # Setup environment-specific state (not async for backend-bench)
    state = prime_env.setup_state(state)

    return BackendBenchEnvironment(prime_env, state)
