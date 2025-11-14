"""Rollouts: Lightweight framework for LLM evaluation and agentic RL.

Tiger Style evaluation framework inspired by llm-workbench/rollouts.
Now with full agent framework for tool-use and multi-turn interactions.
"""

# Core types from dtypes
from .dtypes import (
    Message, ToolCall, ToolResult, Trajectory, Tool, ToolFunction,
    ToolFunctionParameter, StopReason, ToolConfirmResult, StreamChunk,
    Endpoint, Actor, Environment, AgentState, RunConfig, default_confirm_tool,
    Usage, Logprob, Logprobs, Choice, ChatCompletion,
)

# Agent execution from agents
from .agents import (
    stdout_handler, run_agent, rollout,
    confirm_tool_with_feedback, handle_tool_error, inject_turn_warning,
    handle_stop_max_turns, inject_tool_reminder
)

# Environments
from .environments import (
    CalculatorEnvironment,
    BasicEnvironment, NoToolsEnvironment
)

# Checkpoints
from .checkpoints import FileCheckpointStore

# Providers (rollout functions)
from .providers import rollout_openai, rollout_sglang, rollout_anthropic

# Evaluation
from .evaluate import evaluate_dataset, evaluate_sample

# Configuration (new in 0.3.0)
from .config import (
    # Protocols
    HasModelConfig, HasEnvironmentConfig, HasEvaluationConfig, HasOutputConfig,
    # Base configs
    BaseModelConfig, BaseEnvironmentConfig, BaseEvaluationConfig, BaseOutputConfig,
    # Utilities
    load_config_from_file,
)

__all__ = [
    # Core types
    'Endpoint', 'Actor', 'AgentState', 'RunConfig', 'Environment',
    'Usage', 'Logprob', 'Logprobs', 'Choice', 'ChatCompletion',
    # Message types
    'Message', 'ToolCall', 'ToolResult', 'Trajectory',
    # Tool types
    'Tool', 'ToolFunction', 'ToolFunctionParameter', 'StopReason', 'ToolConfirmResult',
    # Stream handling
    'StreamChunk', 'stdout_handler',
    # Agent execution
    'run_agent', 'rollout',
    # Tool handlers
    'confirm_tool_with_feedback', 'handle_tool_error', 'inject_turn_warning',
    'handle_stop_max_turns', 'inject_tool_reminder', 'default_confirm_tool',
    # Environments
    'CalculatorEnvironment',
    'BasicEnvironment', 'NoToolsEnvironment',
    # Checkpoints
    'FileCheckpointStore',
    # Providers
    'rollout_openai', 'rollout_sglang', 'rollout_anthropic',
    # Evaluation
    'evaluate_dataset', 'evaluate_sample',
    # Configuration
    'HasModelConfig', 'HasEnvironmentConfig', 'HasEvaluationConfig', 'HasOutputConfig',
    'BaseModelConfig', 'BaseEnvironmentConfig', 'BaseEvaluationConfig', 'BaseOutputConfig',
    'load_config_from_file',
]

__version__ = "0.3.0"  # Added configuration protocols and base configs
