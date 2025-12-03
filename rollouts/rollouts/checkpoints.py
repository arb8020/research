# Checkpoint storage for agent states

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from dacite import Config, from_dict

from .dtypes import Actor, AgentState, Environment, StopReason


class CheckpointStore(Protocol):
    async def save(self, checkpoint_id: str, state: AgentState) -> None: ...
    async def load(self, checkpoint_id: str) -> AgentState: ...
    async def list(self) -> list[str]: ...


async def serialize_agent_state(state: AgentState) -> dict[str, Any]:
    """Convert AgentState to JSON-serializable dict"""
    assert state is not None
    assert isinstance(state, AgentState)
    assert state.actor is not None
    assert state.environment is not None
    assert state.turn_idx >= 0

    from dataclasses import asdict
    result: dict[str, Any] = {
        "actor": asdict(state.actor),  # asdict handles nested dataclasses!
        "environment": {
            "class_name": state.environment.__class__.__name__,
            "data": await state.environment.serialize()
        },
        "turn_idx": state.turn_idx,
        "stop": state.stop.value if state.stop else None,
        "pending_tool_calls": [asdict(tc) for tc in state.pending_tool_calls],
        "next_tool_idx": state.next_tool_idx,
    }

    assert "actor" in result
    assert "environment" in result
    env_dict = result["environment"]
    assert isinstance(env_dict, dict)
    assert "class_name" in env_dict
    return result


async def deserialize_agent_state(
        data: dict[str, Any],
        environment_registry: dict[str, type[Environment]],
    ) -> AgentState:
    """Reconstruct AgentState from JSON-serializable dict"""
    assert data is not None
    assert isinstance(data, dict)
    assert "environment" in data
    assert "class_name" in data["environment"]
    assert "actor" in data
    assert "turn_idx" in data
    assert environment_registry is not None
    assert isinstance(environment_registry, dict)

    # Handle environment separately
    env_class_name = data["environment"]["class_name"]
    assert env_class_name in environment_registry, f"Unknown environment class: {env_class_name}"
    env_class = environment_registry[env_class_name]
    environment = await env_class.deserialize(data["environment"]["data"])
    assert environment is not None

    # Use dacite for the rest
    state_data = data.copy()
    state_data["environment"] = environment  # Replace with actual object
    state_data["actor"] = from_dict(Actor, data["actor"], Config(check_types=False))
    state_data["stop"] = StopReason(data["stop"]) if data["stop"] else None

    result = from_dict(AgentState, state_data, Config(check_types=False))
    assert result is not None
    assert isinstance(result, AgentState)
    assert result.turn_idx >= 0
    return result


class FileCheckpointStore:
    """File-based checkpoint storage for agent states"""

    def __init__(
        self,
        environment_registry: dict[str, type[Environment]],
        directory: str = "/tmp/rollouts-agent-checkpoints",
    ):
        assert environment_registry is not None
        assert isinstance(environment_registry, dict)
        assert len(environment_registry) > 0
        assert directory is not None
        assert len(directory) > 0

        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=True, parents=True)
        assert self.directory.exists()
        assert self.directory.is_dir()
        # Registry of environment classes for deserialization
        self.environment_registry = environment_registry
    
    async def save(self, checkpoint_id: str, state: AgentState) -> None:
        """Save state to JSON file"""
        assert checkpoint_id is not None
        assert isinstance(checkpoint_id, str)
        assert len(checkpoint_id) > 0
        assert "/" not in checkpoint_id  # Prevent path traversal
        assert ".." not in checkpoint_id  # Prevent path traversal
        assert state is not None
        assert isinstance(state, AgentState)

        data = await serialize_agent_state(state)
        assert data is not None
        assert isinstance(data, dict)

        # Add metadata
        data["_metadata"] = {
            "checkpoint_id": checkpoint_id,
            "timestamp": time.time(),
            "iso_time": datetime.now().isoformat(),
        }

        path = self.directory / f"{checkpoint_id}.json"
        assert path.parent == self.directory  # Ensure no path traversal

        # Use trio.Path for async file I/O
        import trio
        json_str = json.dumps(data, indent=2)
        await trio.Path(path).write_text(json_str)
        assert path.exists()  # Verify file was written
    
    async def load(self, checkpoint_id: str) -> AgentState:
        """Load state from JSON file"""
        assert checkpoint_id is not None
        assert isinstance(checkpoint_id, str)
        assert len(checkpoint_id) > 0
        assert "/" not in checkpoint_id  # Prevent path traversal
        assert ".." not in checkpoint_id  # Prevent path traversal

        path = self.directory / f"{checkpoint_id}.json"
        assert path.parent == self.directory  # Ensure no path traversal
        assert path.exists(), f"Checkpoint file not found: {path}"
        assert path.is_file()

        # Use trio.Path for async file I/O
        import trio
        json_str = await trio.Path(path).read_text()
        data = json.loads(json_str)

        assert data is not None
        assert isinstance(data, dict)

        # Remove metadata before deserializing
        data.pop("_metadata", None)

        result = await deserialize_agent_state(data, self.environment_registry)
        assert result is not None
        return result
    
    async def list(self) -> list[str]:
        """List all checkpoint IDs"""
        assert self.directory.exists()
        assert self.directory.is_dir()

        checkpoints = []
        for path in self.directory.glob("*.json"):
            assert path.is_file()
            checkpoints.append(path.stem)  # filename without extension

        result = sorted(checkpoints)
        assert isinstance(result, list)
        return result