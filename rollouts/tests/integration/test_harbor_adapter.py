from __future__ import annotations

from rollouts.adapters import atif_to_trajectory, trajectory_to_atif_dict
from rollouts.core import (
    ChatCompletion,
    Endpoint,
    Message,
    TextContent,
    ThinkingContent,
    ToolCallContent,
    Trajectory,
    TrajectorySession,
    Usage,
)
from rollouts.dtypes import Cost


def test_trajectory_to_atif_dict_groups_tool_results() -> None:
    trajectory = Trajectory(
        messages=[
            Message(role="system", content="You are helpful.", timestamp="2026-03-10T10:00:00Z"),
            Message(role="user", content="Read the README", timestamp="2026-03-10T10:00:01Z"),
            Message(
                role="assistant",
                content=[
                    ThinkingContent(thinking="I should inspect the file."),
                    TextContent(text="I'll read the README."),
                    ToolCallContent(
                        id="call-1",
                        name="read",
                        arguments={"path": "README.md"},
                    ),
                ],
                model="anthropic/claude-sonnet",
                timestamp="2026-03-10T10:00:02Z",
            ),
            Message(
                role="tool",
                content="# README\nhello\n",
                tool_call_id="call-1",
                timestamp="2026-03-10T10:00:03Z",
            ),
            Message(
                role="assistant",
                content="The README says hello.",
                model="anthropic/claude-sonnet",
                timestamp="2026-03-10T10:00:04Z",
            ),
        ],
        completions=[
            ChatCompletion(
                id="c1",
                object="chat.completion",
                created=1,
                model="anthropic/claude-sonnet",
                usage=Usage(
                    input_tokens=10,
                    output_tokens=5,
                    cache_read_tokens=2,
                    cost=Cost(input=0.1, output=0.2, cache_read=0.03),
                ),
                choices=[],
            ),
            ChatCompletion(
                id="c2",
                object="chat.completion",
                created=2,
                model="anthropic/claude-sonnet",
                usage=Usage(
                    input_tokens=8,
                    output_tokens=4,
                    cost=Cost(input=0.08, output=0.16),
                ),
                choices=[],
            ),
        ],
        metadata={"task_id": "readme-task"},
        session=TrajectorySession(
            session_id="session-123",
            endpoint=Endpoint(
                model="anthropic/claude-sonnet",
                base_url="https://api.anthropic.com/v1",
                api_format="anthropic-messages",
            ),
        ),
    )

    atif = trajectory_to_atif_dict(trajectory)

    assert atif["schema_version"] == "ATIF-v1.6"
    assert atif["session_id"] == "session-123"
    assert atif["agent"]["name"] == "rollouts"
    assert atif["agent"]["model_name"] == "anthropic/claude-sonnet"
    assert atif["final_metrics"] == {
        "total_steps": 2,
        "total_prompt_tokens": 20,
        "total_completion_tokens": 9,
        "total_cached_tokens": 2,
        "total_cost_usd": 0.5700000000000001,
    }

    assert len(atif["steps"]) == 4
    tool_step = atif["steps"][2]
    assert tool_step["source"] == "agent"
    assert tool_step["message"] == "I'll read the README."
    assert tool_step["reasoning_content"] == "I should inspect the file."
    assert tool_step["tool_calls"] == [
        {
            "tool_call_id": "call-1",
            "function_name": "read",
            "arguments": {"path": "README.md"},
        }
    ]
    assert tool_step["observation"] == {
        "results": [
            {
                "source_call_id": "call-1",
                "content": "# README\nhello\n",
            }
        ]
    }
    assert atif["extra"]["rollouts_metadata"] == {"task_id": "readme-task"}


def test_atif_to_trajectory_restores_messages_and_metrics() -> None:
    atif = {
        "schema_version": "ATIF-v1.6",
        "session_id": "session-123",
        "agent": {
            "name": "rollouts",
            "version": "0.4.0",
            "model_name": "anthropic/claude-sonnet",
        },
        "steps": [
            {
                "step_id": 1,
                "source": "system",
                "message": "You are helpful.",
                "timestamp": "2026-03-10T10:00:00Z",
            },
            {
                "step_id": 2,
                "source": "user",
                "message": "Read the README",
                "timestamp": "2026-03-10T10:00:01Z",
            },
            {
                "step_id": 3,
                "source": "agent",
                "message": "I'll read the README.",
                "reasoning_content": "I should inspect the file.",
                "tool_calls": [
                    {
                        "tool_call_id": "call-1",
                        "function_name": "read",
                        "arguments": {"path": "README.md"},
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "call-1",
                            "content": "# README\nhello\n",
                        }
                    ]
                },
                "metrics": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "cached_tokens": 2,
                    "cost_usd": 0.33,
                },
                "timestamp": "2026-03-10T10:00:02Z",
            },
        ],
        "notes": "exported from Harbor",
        "extra": {"foo": "bar"},
    }

    trajectory = atif_to_trajectory(atif)

    assert trajectory.session.session_id == "session-123"
    assert trajectory.session.endpoint is not None
    assert trajectory.session.endpoint.model == "anthropic/claude-sonnet"
    assert [message.role for message in trajectory.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]

    assistant_message = trajectory.messages[2]
    assert isinstance(assistant_message.content, list)
    assert assistant_message.content[0].type == "thinking"
    assert assistant_message.content[1].type == "text"
    assert assistant_message.content[2].type == "toolCall"
    assert trajectory.messages[3].tool_call_id == "call-1"
    assert trajectory.messages[3].content == "# README\nhello\n"

    assert len(trajectory.completions) == 1
    assert trajectory.completions[0].usage.input_tokens == 10
    assert trajectory.completions[0].usage.output_tokens == 5
    assert trajectory.completions[0].usage.cache_read_tokens == 2
    assert trajectory.metadata["harbor_notes"] == "exported from Harbor"
    assert trajectory.metadata["harbor_extra"] == {"foo": "bar"}
