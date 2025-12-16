"""Rollouts: Lightweight framework for LLM evaluation and agentic RL.

Tiger Style evaluation framework inspired by llm-workbench/rollouts.
Now with full agent framework for tool-use and multi-turn interactions.

Uses lazy imports so submodules (e.g., rollouts.inference) can be imported
without pulling in all dependencies (e.g., trio for agents).
"""

import importlib
from typing import TYPE_CHECKING

# For type checkers, import everything eagerly
if TYPE_CHECKING:
    from .agents import (
        confirm_tool_with_feedback,
        handle_stop_max_turns,
        handle_tool_error,
        inject_tool_reminder,
        inject_turn_warning,
        resume_session,
        rollout,
        run_agent,
        stdout_handler,
    )
    from .config import (
        BaseEnvironmentConfig,
        BaseEvaluationConfig,
        BaseModelConfig,
        BaseOutputConfig,
        HasEnvironmentConfig,
        HasEvaluationConfig,
        HasModelConfig,
        HasOutputConfig,
        load_config_from_file,
    )
    from .dtypes import (
        Actor,
        AgentSession,
        AgentState,
        ChatCompletion,
        Choice,
        ContentBlock,
        Endpoint,
        EndpointConfig,
        Environment,
        EnvironmentConfig,
        ImageContent,
        Logprob,
        Logprobs,
        Message,
        Metric,
        RunConfig,
        Sample,
        Score,
        ScoreFn,
        SessionMessage,
        SessionStatus,
        StopReason,
        StreamChunk,
        TextContent,
        ThinkingContent,
        Tool,
        ToolCall,
        ToolCallContent,
        ToolConfirmResult,
        ToolFormatter,
        ToolFunction,
        ToolFunctionParameter,
        ToolResult,
        Trajectory,
        Usage,
        default_confirm_tool,
    )
    from .environments import BasicEnvironment, CalculatorEnvironment, NoToolsEnvironment
    from .evaluation import EvalReport, EvalSample, evaluate, group_by, summarize
    from .export import session_to_html, session_to_markdown
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
    from .providers import (
        get_provider_function,
        rollout_anthropic,
        rollout_google,
        rollout_openai,
        rollout_openai_responses,
        rollout_sglang,
    )
    from .store import FileSessionStore, SessionStore, generate_session_id
    from .transform_messages import transform_messages


# Lazy import mappings: attribute -> (module, name)
_LAZY_IMPORTS = {
    # agents
    "confirm_tool_with_feedback": (".agents", "confirm_tool_with_feedback"),
    "handle_stop_max_turns": (".agents", "handle_stop_max_turns"),
    "handle_tool_error": (".agents", "handle_tool_error"),
    "inject_tool_reminder": (".agents", "inject_tool_reminder"),
    "inject_turn_warning": (".agents", "inject_turn_warning"),
    "resume_session": (".agents", "resume_session"),
    "rollout": (".agents", "rollout"),
    "run_agent": (".agents", "run_agent"),
    "stdout_handler": (".agents", "stdout_handler"),
    # store
    "FileSessionStore": (".store", "FileSessionStore"),
    "SessionStore": (".store", "SessionStore"),
    "generate_session_id": (".store", "generate_session_id"),
    # config
    "BaseEnvironmentConfig": (".config", "BaseEnvironmentConfig"),
    "BaseEvaluationConfig": (".config", "BaseEvaluationConfig"),
    "BaseModelConfig": (".config", "BaseModelConfig"),
    "BaseOutputConfig": (".config", "BaseOutputConfig"),
    "HasEnvironmentConfig": (".config", "HasEnvironmentConfig"),
    "HasEvaluationConfig": (".config", "HasEvaluationConfig"),
    "HasModelConfig": (".config", "HasModelConfig"),
    "HasOutputConfig": (".config", "HasOutputConfig"),
    "load_config_from_file": (".config", "load_config_from_file"),
    # dtypes
    "Actor": (".dtypes", "Actor"),
    "AgentSession": (".dtypes", "AgentSession"),
    "AgentState": (".dtypes", "AgentState"),
    "ChatCompletion": (".dtypes", "ChatCompletion"),
    "Choice": (".dtypes", "Choice"),
    "ContentBlock": (".dtypes", "ContentBlock"),
    "Endpoint": (".dtypes", "Endpoint"),
    "EndpointConfig": (".dtypes", "EndpointConfig"),
    "Environment": (".dtypes", "Environment"),
    "EnvironmentConfig": (".dtypes", "EnvironmentConfig"),
    "ImageContent": (".dtypes", "ImageContent"),
    "Logprob": (".dtypes", "Logprob"),
    "Logprobs": (".dtypes", "Logprobs"),
    "Message": (".dtypes", "Message"),
    "RunConfig": (".dtypes", "RunConfig"),
    "SessionMessage": (".dtypes", "SessionMessage"),
    "SessionStatus": (".dtypes", "SessionStatus"),
    "StopReason": (".dtypes", "StopReason"),
    "StreamChunk": (".dtypes", "StreamChunk"),
    "TextContent": (".dtypes", "TextContent"),
    "ThinkingContent": (".dtypes", "ThinkingContent"),
    "Tool": (".dtypes", "Tool"),
    "ToolCall": (".dtypes", "ToolCall"),
    "ToolCallContent": (".dtypes", "ToolCallContent"),
    "ToolConfirmResult": (".dtypes", "ToolConfirmResult"),
    "ToolFormatter": (".dtypes", "ToolFormatter"),
    "ToolFunction": (".dtypes", "ToolFunction"),
    "ToolFunctionParameter": (".dtypes", "ToolFunctionParameter"),
    "ToolResult": (".dtypes", "ToolResult"),
    "Trajectory": (".dtypes", "Trajectory"),
    "Usage": (".dtypes", "Usage"),
    "default_confirm_tool": (".dtypes", "default_confirm_tool"),
    # Evaluation types
    "Metric": (".dtypes", "Metric"),
    "Score": (".dtypes", "Score"),
    "Sample": (".dtypes", "Sample"),
    "ScoreFn": (".dtypes", "ScoreFn"),
    # environments
    "BasicEnvironment": (".environments", "BasicEnvironment"),
    "CalculatorEnvironment": (".environments", "CalculatorEnvironment"),
    "NoToolsEnvironment": (".environments", "NoToolsEnvironment"),
    # evaluation
    "evaluate": (".evaluation", "evaluate"),
    "EvalSample": (".evaluation", "EvalSample"),
    "EvalReport": (".evaluation", "EvalReport"),
    "group_by": (".evaluation", "group_by"),
    "summarize": (".evaluation", "summarize"),
    # models
    "ApiType": (".models", "ApiType"),
    "ModelCost": (".models", "ModelCost"),
    "ModelMetadata": (".models", "ModelMetadata"),
    "Provider": (".models", "Provider"),
    "calculate_cost": (".models", "calculate_cost"),
    "get_api_type": (".models", "get_api_type"),
    "get_model": (".models", "get_model"),
    "get_models": (".models", "get_models"),
    "get_providers": (".models", "get_providers"),
    "register_model": (".models", "register_model"),
    # providers
    "get_provider_function": (".providers", "get_provider_function"),
    "rollout_anthropic": (".providers", "rollout_anthropic"),
    "rollout_google": (".providers", "rollout_google"),
    "rollout_openai": (".providers", "rollout_openai"),
    "rollout_openai_responses": (".providers", "rollout_openai_responses"),
    "rollout_sglang": (".providers", "rollout_sglang"),
    # transform_messages
    "transform_messages": (".transform_messages", "transform_messages"),
    # export
    "session_to_markdown": (".export", "session_to_markdown"),
    "session_to_html": (".export", "session_to_html"),
}


def __getattr__(name: str):
    """Lazy import attributes on first access."""
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name, __package__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core types
    "Endpoint",
    "Actor",
    "AgentState",
    "RunConfig",
    "Environment",
    "Usage",
    "Logprob",
    "Logprobs",
    "Choice",
    "ChatCompletion",
    # Message types
    "Message",
    "ToolCall",
    "ToolResult",
    "Trajectory",
    # ContentBlock types
    "ContentBlock",
    "TextContent",
    "ThinkingContent",
    "ToolCallContent",
    "ImageContent",
    # Tool types
    "Tool",
    "ToolFunction",
    "ToolFunctionParameter",
    "ToolFormatter",
    "StopReason",
    "ToolConfirmResult",
    # Stream handling
    "StreamChunk",
    "stdout_handler",
    # Agent execution
    "run_agent",
    "rollout",
    "resume_session",
    # Tool handlers
    "confirm_tool_with_feedback",
    "handle_tool_error",
    "inject_turn_warning",
    "handle_stop_max_turns",
    "inject_tool_reminder",
    "default_confirm_tool",
    # Environments
    "CalculatorEnvironment",
    "BasicEnvironment",
    "NoToolsEnvironment",
    # Sessions
    "SessionStore",
    "FileSessionStore",
    "generate_session_id",
    "AgentSession",
    "SessionMessage",
    "SessionStatus",
    "EndpointConfig",
    "EnvironmentConfig",
    # Providers
    "rollout_openai",
    "rollout_sglang",
    "rollout_anthropic",
    "get_provider_function",
    # Message transformation
    "transform_messages",
    # Model registry
    "get_providers",
    "get_models",
    "get_model",
    "register_model",
    "get_api_type",
    "calculate_cost",
    "Provider",
    "ApiType",
    "ModelMetadata",
    "ModelCost",
    # Evaluation
    "evaluate",
    "EvalSample",
    "EvalReport",
    "group_by",
    "summarize",
    "Metric",
    "Score",
    "Sample",
    "ScoreFn",
    # Configuration
    "HasModelConfig",
    "HasEnvironmentConfig",
    "HasEvaluationConfig",
    "HasOutputConfig",
    "BaseModelConfig",
    "BaseEnvironmentConfig",
    "BaseEvaluationConfig",
    "BaseOutputConfig",
    "load_config_from_file",
    # Export
    "session_to_markdown",
    "session_to_html",
    "rollout_google",
    "rollout_openai_responses",
]

__version__ = "0.3.0"  # Added configuration protocols and base configs
