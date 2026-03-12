from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from ..core import (
    ChatCompletion,
    Endpoint,
    ImageContent,
    Message,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCallContent,
    Trajectory,
    TrajectorySession,
    Usage,
)
from ..dtypes import Cost


def trajectory_to_atif_dict(
    trajectory: Trajectory,
    *,
    agent_name: str = "rollouts",
    agent_version: str | None = None,
    tool_definitions: list[Tool | dict[str, Any]] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    if not trajectory.messages:
        raise ValueError("ATIF export requires at least one message")

    resolved_version = agent_version or _default_agent_version()
    steps = _trajectory_messages_to_steps(trajectory)
    final_metrics = _build_final_metrics(trajectory.completions)
    agent_model = _default_model_name(trajectory)
    session_id = trajectory.session.session_id or _fallback_session_id(trajectory)
    extra = _build_root_extra(trajectory)

    result: dict[str, Any] = {
        "schema_version": "ATIF-v1.6",
        "session_id": session_id,
        "agent": {
            "name": agent_name,
            "version": resolved_version,
        },
        "steps": steps,
    }
    if agent_model is not None:
        result["agent"]["model_name"] = agent_model
    normalized_tool_definitions = _normalize_tool_definitions(tool_definitions)
    if normalized_tool_definitions:
        result["agent"]["tool_definitions"] = normalized_tool_definitions
    if notes is not None:
        result["notes"] = notes
    if final_metrics:
        result["final_metrics"] = final_metrics
    if extra:
        result["extra"] = extra
    return result


def atif_to_trajectory(data: Mapping[str, Any]) -> Trajectory:
    session_id = _get_str(data, "session_id")
    agent = _get_mapping(data, "agent")
    agent_model = _get_optional_str(agent, "model_name")
    steps = data.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("ATIF payload field 'steps' must be a list")

    messages: list[Message] = []
    completions: list[ChatCompletion] = []

    for index, raw_step in enumerate(steps, start=1):
        if not isinstance(raw_step, Mapping):
            raise ValueError(f"ATIF step {index} must be an object")
        source = _get_str(raw_step, "source")
        timestamp = _get_optional_str(raw_step, "timestamp")
        raw_message = raw_step.get("message")

        if source in ("system", "user"):
            messages.append(
                Message(
                    role=source,
                    content=_atif_message_to_rollouts_content(raw_message),
                    timestamp=timestamp,
                )
            )
            continue

        if source != "agent":
            raise ValueError(f"Unsupported ATIF source '{source}'")

        assistant_message = _atif_agent_step_to_message(raw_step, agent_model)
        messages.append(assistant_message)

        observation = raw_step.get("observation")
        if isinstance(observation, Mapping):
            results = observation.get("results", [])
            if isinstance(results, list):
                for raw_result in results:
                    if not isinstance(raw_result, Mapping):
                        continue
                    messages.append(
                        Message(
                            role="tool",
                            content=_atif_message_to_rollouts_content(raw_result.get("content")),
                            tool_call_id=_get_optional_str(raw_result, "source_call_id"),
                            timestamp=timestamp,
                        )
                    )

        metrics = raw_step.get("metrics")
        if isinstance(metrics, Mapping):
            completions.append(
                _atif_metrics_to_completion(
                    metrics=metrics,
                    model_name=_get_optional_str(raw_step, "model_name") or agent_model,
                    index=index,
                )
            )

    metadata = _build_trajectory_metadata_from_atif(data)

    endpoint = _infer_endpoint(agent_model)
    return Trajectory(
        completions=completions,
        messages=messages,
        metadata=metadata,
        session=TrajectorySession(
            session_id=session_id,
            endpoint=endpoint,
        ),
    )


def _default_agent_version() -> str:
    try:
        return version("rollouts")
    except PackageNotFoundError:
        return "unknown"


def _fallback_session_id(trajectory: Trajectory) -> str:
    created_at = trajectory.session.created_at or datetime.now(timezone.utc).isoformat()
    return f"rollouts-{created_at}"


def _default_model_name(trajectory: Trajectory) -> str | None:
    endpoint = trajectory.session.endpoint
    if endpoint is not None and endpoint.model:
        return endpoint.model

    for message in reversed(trajectory.messages):
        if message.model:
            return message.model

    for completion in reversed(trajectory.completions):
        if completion.model:
            return completion.model

    return None


def _infer_endpoint(model_name: str | None) -> Endpoint | None:
    if not model_name:
        return None

    provider = model_name.split("/", 1)[0]
    if provider == "anthropic":
        return Endpoint(
            model=model_name,
            base_url="https://api.anthropic.com/v1",
            api_format="anthropic-messages",
        )
    if provider == "openai":
        return Endpoint(
            model=model_name,
            base_url="https://api.openai.com/v1",
            api_format="openai-completions",
        )
    if provider == "google":
        return Endpoint(
            model=model_name,
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_format="google-generative-ai",
        )
    if provider in {
        "openrouter",
        "groq",
        "cerebras",
        "xai",
        "sglang",
        "vllm",
        "moonshot",
        "zhipu",
        "fireworks",
        "together",
        "bedrock",
    }:
        return Endpoint(
            model=model_name,
            base_url="https://openrouter.ai/api/v1"
            if provider == "openrouter"
            else "https://api.openai.com/v1",
            api_format="openai-completions",
        )
    return None


def _normalize_tool_definitions(
    tool_definitions: list[Tool | dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if not tool_definitions:
        return None

    normalized: list[dict[str, Any]] = []
    for item in tool_definitions:
        if isinstance(item, Mapping):
            normalized.append(dict(item))
            continue
        if isinstance(item, Tool):
            normalized.append(asdict(item))
            continue
        if is_dataclass(item):
            normalized.append(asdict(item))
            continue
        raise TypeError(f"Unsupported tool definition type: {type(item)!r}")
    return normalized


def _trajectory_messages_to_steps(trajectory: Trajectory) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    completion_index = 0
    i = 0

    while i < len(trajectory.messages):
        message = trajectory.messages[i]
        step_id = len(steps) + 1

        if message.role == "assistant":
            assistant_step = _assistant_message_to_step(
                step_id=step_id,
                message=message,
                completion=trajectory.completions[completion_index]
                if completion_index < len(trajectory.completions)
                else None,
            )
            completion_index += 1

            if assistant_step["tool_calls"]:
                observation_results: list[dict[str, Any]] = []
                j = i + 1
                while j < len(trajectory.messages) and trajectory.messages[j].role == "tool":
                    observation_results.append({
                        "source_call_id": trajectory.messages[j].tool_call_id,
                        "content": _rollouts_content_to_atif(trajectory.messages[j].content),
                    })
                    j += 1
                if observation_results:
                    assistant_step["observation"] = {"results": observation_results}
                steps.append(_strip_nones(assistant_step))
                i = j
                continue

            steps.append(_strip_nones(assistant_step))
            i += 1
            continue

        if message.role == "tool":
            steps.append(
                _strip_nones({
                    "step_id": step_id,
                    "timestamp": message.timestamp,
                    "source": "system",
                    "message": "",
                    "observation": {
                        "results": [
                            {
                                "source_call_id": message.tool_call_id,
                                "content": _rollouts_content_to_atif(message.content),
                            }
                        ]
                    },
                })
            )
            i += 1
            continue

        steps.append(
            _strip_nones({
                "step_id": step_id,
                "timestamp": message.timestamp,
                "source": message.role,
                "message": _rollouts_content_to_atif(message.content),
            })
        )
        i += 1

    return steps


def _assistant_message_to_step(
    *,
    step_id: int,
    message: Message,
    completion: ChatCompletion | None,
) -> dict[str, Any]:
    text_parts: list[str] = []
    message_parts: list[dict[str, Any]] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    if isinstance(message.content, str) or message.content is None:
        content = message.content or ""
    else:
        content = message.content
        for block in content:
            if isinstance(block, ThinkingContent):
                reasoning_parts.append(block.thinking)
                continue
            if isinstance(block, ToolCallContent):
                tool_calls.append({
                    "tool_call_id": block.id,
                    "function_name": block.name,
                    "arguments": dict(block.arguments),
                })
                continue
            if isinstance(block, TextContent):
                text_parts.append(block.text)
                message_parts.append({"type": "text", "text": block.text})
                continue
            if isinstance(block, ImageContent):
                message_parts.append({
                    "type": "image",
                    "source": {
                        "media_type": _infer_media_type(block.image_url),
                        "path": block.image_url,
                    },
                })

    step_message = _compress_message_parts(
        text_parts=text_parts,
        message_parts=message_parts,
        fallback=content if isinstance(content, str) else "",
    )

    result: dict[str, Any] = {
        "step_id": step_id,
        "timestamp": message.timestamp,
        "source": "agent",
        "message": step_message,
        "tool_calls": tool_calls or None,
        "reasoning_content": "\n\n".join(part for part in reasoning_parts if part) or None,
        "model_name": message.model or (completion.model if completion is not None else None),
    }

    metrics = _completion_to_metrics(completion)
    if metrics:
        result["metrics"] = metrics
    return result


def _completion_to_metrics(completion: ChatCompletion | None) -> dict[str, Any] | None:
    if completion is None:
        return None

    usage = completion.usage
    metrics: dict[str, Any] = {}
    if usage.prompt_tokens:
        metrics["prompt_tokens"] = usage.prompt_tokens
    if usage.completion_tokens:
        metrics["completion_tokens"] = usage.completion_tokens
    if usage.cache_read_tokens:
        metrics["cached_tokens"] = usage.cache_read_tokens
    if usage.cost.total:
        metrics["cost_usd"] = usage.cost.total

    choice = completion.choices[0] if completion.choices else None
    if choice and choice.token_ids:
        metrics["completion_token_ids"] = list(choice.token_ids)
    if completion.prompt_token_ids:
        metrics["prompt_token_ids"] = list(completion.prompt_token_ids)
    if choice and choice.logprobs is not None:
        metrics["logprobs"] = [entry.logprob for entry in choice.logprobs.content]

    return metrics or None


def _build_final_metrics(completions: list[ChatCompletion]) -> dict[str, Any] | None:
    if not completions:
        return None

    total_prompt_tokens = sum(comp.usage.prompt_tokens for comp in completions)
    total_completion_tokens = sum(comp.usage.completion_tokens for comp in completions)
    total_cached_tokens = sum(comp.usage.cache_read_tokens for comp in completions)
    total_cost_usd = sum(comp.usage.cost.total for comp in completions)

    result: dict[str, Any] = {
        "total_steps": len(completions),
    }
    if total_prompt_tokens:
        result["total_prompt_tokens"] = total_prompt_tokens
    if total_completion_tokens:
        result["total_completion_tokens"] = total_completion_tokens
    if total_cached_tokens:
        result["total_cached_tokens"] = total_cached_tokens
    if total_cost_usd:
        result["total_cost_usd"] = total_cost_usd
    return result


def _build_root_extra(trajectory: Trajectory) -> dict[str, Any] | None:
    extra: dict[str, Any] = {}
    if trajectory.metadata:
        extra["rollouts_metadata"] = trajectory.metadata
    annotations = trajectory.annotations.to_dict()
    if annotations:
        extra["rollouts_annotations"] = annotations
    session = trajectory.session.to_dict()
    if session:
        extra["rollouts_session"] = session
    if trajectory.environment is not None:
        extra["rollouts_environment"] = trajectory.environment.to_dict()
    return extra or None


def _rollouts_content_to_atif(
    content: str | list[Any] | None,
) -> str | list[dict[str, Any]]:
    if isinstance(content, str) or content is None:
        return content or ""

    text_parts: list[str] = []
    message_parts: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextContent):
            text_parts.append(block.text)
            message_parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            message_parts.append({
                "type": "image",
                "source": {
                    "media_type": _infer_media_type(block.image_url),
                    "path": block.image_url,
                },
            })

    return _compress_message_parts(
        text_parts=text_parts,
        message_parts=message_parts,
        fallback="",
    )


def _compress_message_parts(
    *,
    text_parts: list[str],
    message_parts: list[dict[str, Any]],
    fallback: str,
) -> str | list[dict[str, Any]]:
    if not message_parts:
        return fallback
    if len(message_parts) == 1 and message_parts[0]["type"] == "text":
        return message_parts[0]["text"]
    if message_parts and not any(part["type"] == "image" for part in message_parts):
        return "\n".join(text_parts)
    return message_parts


def _atif_message_to_rollouts_content(
    raw_message: Any,
) -> str | list[TextContent | ImageContent] | None:
    if raw_message is None:
        return None
    if isinstance(raw_message, str):
        return raw_message
    if not isinstance(raw_message, list):
        raise ValueError("ATIF message content must be a string or list")

    parts: list[TextContent | ImageContent] = []
    for raw_part in raw_message:
        if not isinstance(raw_part, Mapping):
            raise ValueError("ATIF content parts must be objects")
        part_type = _get_str(raw_part, "type")
        if part_type == "text":
            parts.append(TextContent(text=_get_str(raw_part, "text")))
            continue
        if part_type == "image":
            source = _get_mapping(raw_part, "source")
            parts.append(
                ImageContent(
                    image_url=_get_str(source, "path"),
                    detail=None,
                )
            )
            continue
        raise ValueError(f"Unsupported ATIF content part type '{part_type}'")
    return parts


def _atif_agent_step_to_message(
    step: Mapping[str, Any],
    default_model_name: str | None,
) -> Message:
    raw_message = step.get("message")
    message_content = _atif_message_to_rollouts_content(raw_message)
    tool_calls = step.get("tool_calls")
    reasoning = _get_optional_str(step, "reasoning_content")

    has_tool_calls = isinstance(tool_calls, list) and len(tool_calls) > 0
    has_reasoning = bool(reasoning)
    has_multimodal = isinstance(message_content, list)

    if not (has_tool_calls or has_reasoning or has_multimodal):
        return Message(
            role="assistant",
            content=message_content,
            model=_get_optional_str(step, "model_name") or default_model_name,
            timestamp=_get_optional_str(step, "timestamp"),
        )

    blocks: list[Any] = []
    if reasoning:
        blocks.append(ThinkingContent(thinking=reasoning))

    if isinstance(message_content, str):
        if message_content:
            blocks.append(TextContent(text=message_content))
    elif isinstance(message_content, list):
        blocks.extend(message_content)

    if isinstance(tool_calls, list):
        for raw_tool_call in tool_calls:
            if not isinstance(raw_tool_call, Mapping):
                continue
            blocks.append(
                ToolCallContent(
                    id=_get_str(raw_tool_call, "tool_call_id"),
                    name=_get_str(raw_tool_call, "function_name"),
                    arguments=dict(_get_mapping(raw_tool_call, "arguments")),
                )
            )

    return Message(
        role="assistant",
        content=blocks,
        model=_get_optional_str(step, "model_name") or default_model_name,
        timestamp=_get_optional_str(step, "timestamp"),
    )


def _atif_metrics_to_completion(
    *,
    metrics: Mapping[str, Any],
    model_name: str | None,
    index: int,
) -> ChatCompletion:
    cost_total = _get_optional_float(metrics, "cost_usd") or 0.0
    usage = Usage(
        input_tokens=_get_optional_int(metrics, "prompt_tokens") or 0,
        output_tokens=_get_optional_int(metrics, "completion_tokens") or 0,
        cache_read_tokens=_get_optional_int(metrics, "cached_tokens") or 0,
        cost=Cost(output=cost_total),
    )

    return ChatCompletion(
        id=f"atif-step-{index}",
        object="chat.completion",
        created=0,
        model=model_name or "unknown",
        usage=usage,
        choices=[],
        prompt_token_ids=_tuple_of_ints(metrics.get("prompt_token_ids")),
    )


def _build_trajectory_metadata_from_atif(data: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "harbor_schema_version": _get_optional_str(data, "schema_version"),
    }
    notes = _get_optional_str(data, "notes")
    if notes is not None:
        metadata["harbor_notes"] = notes
    extra = data.get("extra")
    if isinstance(extra, Mapping):
        metadata["harbor_extra"] = dict(extra)
    final_metrics = data.get("final_metrics")
    if isinstance(final_metrics, Mapping):
        metadata["harbor_final_metrics"] = dict(final_metrics)
    agent = data.get("agent")
    if isinstance(agent, Mapping):
        metadata["harbor_agent"] = dict(agent)
    continued = _get_optional_str(data, "continued_trajectory_ref")
    if continued is not None:
        metadata["harbor_continued_trajectory_ref"] = continued
    return {k: v for k, v in metadata.items() if v is not None}


def _tuple_of_ints(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    ints: list[int] = []
    for item in value:
        if isinstance(item, int):
            ints.append(item)
    return tuple(ints)


def _infer_media_type(path: str) -> str:
    lowered = path.lower()
    if lowered.endswith(".jpg") or lowered.endswith(".jpeg"):
        return "image/jpeg"
    if lowered.endswith(".gif"):
        return "image/gif"
    if lowered.endswith(".webp"):
        return "image/webp"
    return "image/png"


def _get_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected object field '{key}'")
    return value


def _get_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Expected string field '{key}'")
    return value


def _get_optional_str(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Expected string field '{key}'")
    return value


def _get_optional_int(data: Mapping[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, int):
        return value
    raise ValueError(f"Expected integer field '{key}'")


def _get_optional_float(data: Mapping[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Expected numeric field '{key}'")


def _strip_nones(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_nones(item)
            for key, item in value.items()
            if item is not None and item != []
        }
    if isinstance(value, list):
        return [_strip_nones(item) for item in value]
    return value
