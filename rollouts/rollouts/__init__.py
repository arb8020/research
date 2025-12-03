"""Rollouts: Lightweight framework for LLM evaluation and agentic RL.

Tiger Style evaluation framework inspired by llm-workbench/rollouts.
Now with full agent framework for tool-use and multi-turn interactions.
"""

# Core types from dtypes
# Agent execution from agents
from .agents import (
    confirm_tool_with_feedback,
    handle_stop_max_turns,
    handle_tool_error,
    inject_tool_reminder,
    inject_turn_warning,
    rollout,
    run_agent,
    stdout_handler,
)

# Checkpoints
from .checkpoints import FileCheckpointStore

# Configuration (new in 0.3.0)
from .config import (
    BaseEnvironmentConfig,
    BaseEvaluationConfig,
    # Base configs
    BaseModelConfig,
    BaseOutputConfig,
    HasEnvironmentConfig,
    HasEvaluationConfig,
    # Protocols
    HasModelConfig,
    HasOutputConfig,
    # Utilities
    load_config_from_file,
)
from .dtypes import (
    Actor,
    AgentState,
    ChatCompletion,
    Choice,
    Endpoint,
    Environment,
    Logprob,
    Logprobs,
    Message,
    RunConfig,
    StopReason,
    StreamChunk,
    Tool,
    ToolCall,
    ToolConfirmResult,
    ToolFunction,
    ToolFunctionParameter,
    ToolResult,
    Trajectory,
    Usage,
    default_confirm_tool,
)

# Environments
from .environments import BasicEnvironment, CalculatorEnvironment, NoToolsEnvironment

# Evaluation
from .evaluate import evaluate_dataset, evaluate_sample

# Model registry (new unified provider API)
from .models import (
    ApiType,
    ModelCost,
    ModelMetadata,
    Provider,
    calculate_cost,
    get_api_type,
    get_model,
    get_models,
    get_providers,
    register_model,
)

# Providers (rollout functions)
from .providers import get_provider_function, rollout_anthropic, rollout_google, rollout_openai, rollout_openai_responses, rollout_sglang

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
    'rollout_openai', 'rollout_sglang', 'rollout_anthropic', 'get_provider_function',
    # Model registry
    'get_providers', 'get_models', 'get_model', 'register_model', 'get_api_type', 'calculate_cost',
    'Provider', 'ApiType', 'ModelMetadata', 'ModelCost',
    # Evaluation
    'evaluate_dataset', 'evaluate_sample',
    # Configuration
    'HasModelConfig', 'HasEnvironmentConfig', 'HasEvaluationConfig', 'HasOutputConfig',
    'BaseModelConfig', 'BaseEnvironmentConfig', 'BaseEvaluationConfig', 'BaseOutputConfig',
    'load_config_from_file',
]

__version__ = "0.3.0"  # Added configuration protocols and base configs
