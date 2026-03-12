from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..core import StopReason, Tool, ToolCall, ToolFunction, ToolFunctionParameter, ToolResult
from .coding import CodingEnvironment
from .resources import (
    CommandExecutionResult,
    CommandRunner,
    TerminalTaskResource,
    TerminalTaskResult,
    TerminalWorkspaceResource,
)

_CWD_MARKER = "__ROLLOUTS_CWD__="


class TerminalSessionCommandRunner:
    def __init__(self, resource: TerminalTaskResource) -> None:
        self.resource = resource

    async def run(
        self,
        command: str,
        *,
        cwd: str,
        timeout: float,
        session_id: str | None = None,
        cancel_scope: Any | None = None,
    ) -> CommandExecutionResult:
        wrapped = f"cd {cwd}\n{command}\nprintf '\\n{_CWD_MARKER}%s\\n' \"$PWD\"\n"
        output = await self.resource.send_keys(
            wrapped,
            is_blocking=True,
            timeout_sec=timeout,
        )
        cleaned_output, updated_cwd = _extract_cwd(output)
        return CommandExecutionResult(
            returncode=0,
            stdout=cleaned_output,
            stderr="",
            cwd=updated_cwd or cwd,
        )


@dataclass
class TerminalBenchCodingEnvironment(CodingEnvironment):
    resource: TerminalWorkspaceResource
    tools: str | list[str] = field(
        default_factory=lambda: ["read", "write", "edit", "bash", "task_complete"]
    )
    interactions: list[dict[str, Any]] = field(default_factory=list)
    _task_completed: bool = False
    _command_runner: CommandRunner | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._command_runner = TerminalSessionCommandRunner(self.resource)
        CodingEnvironment.__init__(
            self,
            self.resource,
            self._command_runner,
            tools=self.tools,
            bash_allowlist=None,
            summarize_web_fetch=False,
            current_working_dir=self.resource.working_dir,
        )

    @property
    def task_id(self) -> str:
        return self.resource.task_id

    @property
    def instruction(self) -> str:
        return self.resource.instruction

    @property
    def logging_dir(self) -> Any | None:
        return getattr(self.resource, "logging_dir", None)

    @property
    def max_agent_timeout_sec(self) -> float | None:
        task = getattr(self.resource, "task", None)
        return getattr(task, "max_agent_timeout_sec", None)

    @property
    def max_test_timeout_sec(self) -> float | None:
        task = getattr(self.resource, "task", None)
        return getattr(task, "max_test_timeout_sec", None)

    async def serialize(self) -> dict[str, Any]:
        return {
            "env_kind": "terminal_bench_coding",
            "task_id": self.task_id,
            "instruction": self.instruction,
            "current_working_dir": self.current_working_dir,
            "interactions": self.interactions,
            "task_completed": self._task_completed,
            "_resource_ref": self.resource,
        }

    @staticmethod
    async def deserialize(data: dict[str, Any]) -> TerminalBenchCodingEnvironment:
        resource = data.get("_resource_ref")
        if resource is None:
            raise NotImplementedError(
                "TerminalBenchCodingEnvironment cannot be deserialized from cold storage "
                "because the live terminal resource is not persistable."
            )
        return TerminalBenchCodingEnvironment(
            resource=resource,
            interactions=data.get("interactions", []),
            _task_completed=data.get("task_completed", False),
        )

    async def cleanup(self) -> None:
        await self.resource.close()

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        return False

    def get_status_info(self) -> dict[str, str] | None:
        return {
            "task": self.task_id,
            "cwd": self.current_working_dir,
            "completed": str(self._task_completed),
        }

    def get_system_prompt(self) -> str | None:
        return f"""You are solving a terminal-bench task in a containerized coding workspace.

## Task
{self.instruction}

## Workspace
- Initial working directory: {self.current_working_dir}
- Use read/write/edit for precise file changes
- Use bash for commands, tests, and navigation
- When the task is solved, call task_complete

## Strategy
1. Inspect the workspace with read and bash
2. Make focused changes with write/edit
3. Use bash to verify intermediate progress
4. Call task_complete once you believe the task is solved"""

    def _get_extra_tools(self) -> list[Tool]:
        return [
            Tool(
                type="function",
                function=ToolFunction(
                    name="task_complete",
                    description="Signal that the task is complete.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "summary": {
                                "type": "string",
                                "description": "Brief summary of the completed work.",
                            }
                        },
                    ),
                    required=["summary"],
                ),
            )
        ]

    async def _exec_extra_tool(self, tool_call: ToolCall) -> ToolResult | None:
        if tool_call.name != "task_complete":
            return None
        summary = tool_call.args.get("summary", "Task completed")
        self._task_completed = True
        self.interactions.append({
            "type": "task_complete",
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        })
        if self.logging_dir is not None:
            (self.logging_dir / "interactions.json").write_text(
                json.dumps(self.interactions, indent=2)
            )
        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content=f"Task marked as complete. Summary: {summary}",
            stop_reason=StopReason.TASK_COMPLETED,
        )

    async def run_tests(self) -> TerminalTaskResult:
        return await self.resource.run_tests()

    def get_trajectory(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "cwd": self.current_working_dir,
            "interactions": self.interactions,
            "success": self._task_completed,
            "failed_reason": "" if self._task_completed else "Task not completed",
        }


def supports_terminal_workspace(resource: TerminalTaskResource) -> bool:
    return isinstance(resource, TerminalWorkspaceResource)


def _extract_cwd(output: str) -> tuple[str, str | None]:
    marker_idx = output.rfind(_CWD_MARKER)
    if marker_idx < 0:
        return output, None
    cwd_line = output[marker_idx + len(_CWD_MARKER) :].splitlines()[0].strip()
    cleaned = output[:marker_idx].rstrip()
    return cleaned, cwd_line or None
