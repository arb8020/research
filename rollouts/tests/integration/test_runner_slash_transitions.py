from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import trio

from rollouts.agents import Actor, AgentState
from rollouts.core import Endpoint, EnvironmentConfig, Message, Trajectory
from rollouts.frontends.runner import InteractiveRunner, RunnerConfig
from rollouts.slice import run_slice_command
from rollouts.store import FileSessionStore


@dataclass
class StubFrontend:
    messages: list[str]

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def handle_event(self, event: object) -> None:
        return None

    async def get_input(self, prompt: str = "") -> NoReturn:
        raise RuntimeError("get_input should not be called in this test")

    async def confirm_tool(self, tool_call: object) -> bool:
        return True

    def show_loader(self, text: str) -> None:
        return None

    def hide_loader(self) -> None:
        return None

    def add_system_message(self, text: str) -> None:
        self.messages.append(text)


def test_interactive_runner_switch_session_updates_active_handle(tmp_path: Path) -> None:
    async def _test() -> None:
        session_store = FileSessionStore(base_dir=tmp_path / "sessions")
        parent_endpoint = Endpoint.from_legacy(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
        )

        parent = await session_store.create(
            endpoint=parent_endpoint,
            environment=EnvironmentConfig(type="none"),
        )
        await session_store.append_message(parent.session_id, Message(role="user", content="Hello"))
        await session_store.append_message(
            parent.session_id, Message(role="assistant", content="Hi")
        )
        parent, _ = await session_store.get(parent.session_id)
        assert parent is not None

        child, err = await run_slice_command(
            session=parent,
            spec="0:1",
            endpoint=parent_endpoint,
            session_store=session_store,
        )
        assert not err
        assert child is not None

        runner = InteractiveRunner(
            trajectory=Trajectory(messages=parent.messages),
            endpoint=Endpoint.from_legacy(provider="openai", model="gpt-4o"),
            frontend=StubFrontend(messages=[]),
            config=RunnerConfig(
                session_store=session_store,
                session_id=parent.session_id,
            ),
        )

        switched = await runner.switch_session(child.session_id)
        assert switched
        assert runner.session_id == child.session_id
        assert runner.parent_session_id == parent.session_id
        assert runner.endpoint.provider == child.endpoint.provider
        assert runner.endpoint.model_id == child.endpoint.model_id
        assert runner.trajectory.messages == child.messages

    trio.run(_test)


def test_model_slash_command_forks_child_session(tmp_path: Path) -> None:
    async def _test() -> None:
        session_store = FileSessionStore(base_dir=tmp_path / "sessions")
        parent_endpoint = Endpoint.from_legacy(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
        )

        parent = await session_store.create(
            endpoint=parent_endpoint,
            environment=EnvironmentConfig(type="none"),
        )
        await session_store.append_message(parent.session_id, Message(role="user", content="Hello"))
        parent, _ = await session_store.get(parent.session_id)
        assert parent is not None

        frontend = StubFrontend(messages=[])
        runner = InteractiveRunner(
            trajectory=parent,
            endpoint=parent.endpoint,
            frontend=frontend,
            config=RunnerConfig(
                session_store=session_store,
                session_id=parent.session_id,
            ),
        )

        state = AgentState(
            actor=Actor(
                trajectory=parent,
                endpoint=parent.endpoint,
                tools=[],
            ),
            environment=None,
            session_id=parent.session_id,
        )

        result = await runner._handle_slash_command("model", "openai/gpt-4o", state)

        assert result.handled
        assert runner.session_id != parent.session_id
        assert runner.parent_session_id == parent.session_id
        assert runner.endpoint.provider == "openai"
        assert runner.endpoint.model_id == "gpt-4o"
        assert result.state.session_id == runner.session_id
        assert result.state.actor.endpoint.provider == "openai"
        assert result.state.actor.trajectory.messages == parent.messages
        assert any("Switched to child session" in message for message in frontend.messages)

        children = await session_store.list_children(parent.session_id)
        assert any(
            child.endpoint.provider == "openai" and child.endpoint.model_id == "gpt-4o"
            for child in children
        )

    trio.run(_test)
