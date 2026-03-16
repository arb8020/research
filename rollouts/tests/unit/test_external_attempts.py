from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from rollouts.core import Message, Trajectory
from rollouts.drivers.runner import run_driver_to_trajectory
from rollouts.dtypes import StreamChunk, StreamEvent, TextContent, TextDelta, TextEnd, TextStart
from rollouts.eval.external_attempts import (
    ExternalAttemptArtifact,
    _make_raw_driver_line_handler,
    _message_from_openhands_event,
    _parse_openhands_json_events,
    execute_external_attempt,
)
from rollouts.training.types import Status


@pytest.mark.trio
async def test_execute_external_attempt_builds_attempt_row() -> None:
    async def _trajectory_adapter(
        prompt: str,
        sample_id: str,
        sample_data: dict[str, str],
    ) -> ExternalAttemptArtifact:
        assert prompt == "prompt:hello"
        assert sample_id == "sample-1"
        assert sample_data["text"] == "hello"
        return ExternalAttemptArtifact(
            trajectory=Trajectory(
                messages=[Message(role="assistant", content="<reversed_text>olleh</reversed_text>")]
            ),
            metadata={"runtime": "fake"},
            status=Status.COMPLETED,
        )

    sample = await execute_external_attempt(
        {"text": "hello"},
        "sample-1",
        None,
        None,
        prompt_builder=lambda sample: f"prompt:{sample['text']}",
        trajectory_adapter=_trajectory_adapter,
    )

    assert sample.attempt_id == "sample-1"
    assert sample.problem is not None
    assert sample.problem.payload["text"] == "hello"
    assert sample.response == "<reversed_text>olleh</reversed_text>"
    assert sample.metadata["runtime"] == "fake"
    assert sample.status is Status.COMPLETED


@pytest.mark.trio
async def test_execute_external_attempt_passes_run_config_when_adapter_accepts_it() -> None:
    observed_run_config: object | None = None

    async def _trajectory_adapter(
        prompt: str,
        sample_id: str,
        sample_data: dict[str, str],
        *,
        run_config: object,
    ) -> ExternalAttemptArtifact:
        nonlocal observed_run_config
        observed_run_config = run_config
        assert prompt == "prompt:hello"
        assert sample_id == "sample-1"
        assert sample_data["text"] == "hello"
        return ExternalAttemptArtifact(
            trajectory=Trajectory(messages=[Message(role="assistant", content="ok")]),
            metadata={"runtime": "fake"},
            status=Status.COMPLETED,
        )

    run_config = object()
    sample = await execute_external_attempt(
        {"text": "hello"},
        "sample-1",
        None,
        run_config,
        prompt_builder=lambda sample: f"prompt:{sample['text']}",
        trajectory_adapter=_trajectory_adapter,
    )

    assert sample.metadata["runtime"] == "fake"
    assert observed_run_config is run_config


@pytest.mark.trio
async def test_make_raw_driver_line_handler_emits_stream_chunk() -> None:
    observed: list[StreamChunk] = []

    class _RunConfig:
        async def on_chunk(self, event: StreamChunk) -> None:
            observed.append(event)

    handler = _make_raw_driver_line_handler(_RunConfig(), driver="claude")
    assert handler is not None

    await handler('{"type":"assistant"}')

    assert len(observed) == 1
    event = observed[0]
    assert event.type == "raw_driver_line"
    assert event.data["driver"] == "claude"
    assert event.data["raw_line"] == '{"type":"assistant"}'


@pytest.mark.trio
async def test_run_driver_to_trajectory_forwards_live_events() -> None:
    class _FakeDriver:
        async def run(self, prompt: str) -> AsyncIterator[StreamEvent]:
            assert prompt == "Fix it"
            yield TextStart(content_index=0)
            yield TextDelta(content_index=0, delta="hello")
            yield TextEnd(content_index=0, content="hello")

        async def send_input(self, text: str) -> None:
            raise AssertionError(f"unexpected input: {text}")

        async def abort(self) -> None:
            return None

    observed: list[object] = []

    async def _on_event(event: object) -> None:
        observed.append(event)

    trajectory = await run_driver_to_trajectory(
        _FakeDriver(),
        "Fix it",
        sample_id="sample-1",
        on_event=_on_event,
    )

    assert [type(event) for event in observed] == [TextStart, TextDelta, TextEnd]
    assert len(trajectory.messages) == 1
    assert trajectory.messages[0].role == "assistant"
    content = trajectory.messages[0].content
    assert isinstance(content, list)
    assert len(content) == 1
    assert isinstance(content[0], TextContent)
    assert content[0].text == "hello"


def test_parse_openhands_json_events_extracts_pretty_json_blocks() -> None:
    stdout = """
Initializing agent...
--JSON Event--
{
  "kind": "MessageEvent",
  "source": "user",
  "llm_message": {
    "role": "user",
    "content": [{"type": "text", "text": "Reply with hello"}]
  }
}
Agent is working
--JSON Event--
{
  "kind": "MessageEvent",
  "source": "agent",
  "llm_message": {
    "role": "assistant",
    "content": [{"type": "text", "text": "hello"}]
  }
}
"""

    events = _parse_openhands_json_events(stdout)
    assert len(events) == 2
    assert events[0]["source"] == "user"
    assert events[1]["source"] == "agent"


def test_message_from_openhands_event_extracts_text_message() -> None:
    event = {
        "kind": "MessageEvent",
        "source": "agent",
        "llm_message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
        },
    }

    message = _message_from_openhands_event(event)
    assert message is not None
    assert message.role == "assistant"
    assert message.content == "hello"
