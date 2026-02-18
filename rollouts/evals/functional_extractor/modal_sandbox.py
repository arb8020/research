"""Modal Sandbox with CodingEnvironment tools.

A sandbox environment that runs bash/read/write/edit commands inside a Modal
container with GPU access. The agent interacts with files and runs commands
in the remote sandbox, not locally.

Usage:
    env = ModalSandboxCodingEnvironment(
        sandbox_config=SandboxConfig(gpu_type="H100"),
        workspace_setup=workspace_setup_fn,  # Populates /workspace with task files
    )

    # Agent gets standard coding tools (bash, read, write, edit)
    # All commands execute in the Modal sandbox
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import modal
    import trio

logger = logging.getLogger(__name__)

# Where we put the agent's workspace inside the container
WORKSPACE_PATH = "/workspace"

# Large output handling (matches CodingEnvironment pattern)
MAX_OUTPUT_SIZE = 30_000  # 30KB - write to file if larger


@dataclass
class SandboxConfig:
    """Configuration for Modal sandbox."""

    gpu_type: str = "H100"
    timeout_seconds: int = 1800  # 30 min default
    scaledown_window: int = 300  # Keep warm for 5 min

    # Model to pre-load (optional - for caching weights in image)
    model_name: str | None = None


@dataclass
class ModalSandboxCodingEnvironment:
    """CodingEnvironment that executes in a Modal sandbox.

    Provides bash, read, write, edit tools that all execute remotely
    in a GPU-enabled Modal container.
    """

    sandbox_config: SandboxConfig

    # Function to set up workspace: (sandbox) -> None
    # Called after sandbox creation to populate /workspace
    workspace_setup: Any | None = None

    # Per-sample data (set by for_sample)
    sample_data: dict[str, Any] = field(default_factory=dict)

    # Sandbox state
    _sandbox: modal.Sandbox | None = field(default=None, repr=False)
    _sandbox_id: str | None = field(default=None, repr=False)

    def for_sample(self, sample_data: dict[str, Any]) -> ModalSandboxCodingEnvironment:
        """Create a fresh environment for a sample."""
        return ModalSandboxCodingEnvironment(
            sandbox_config=self.sandbox_config,
            workspace_setup=self.workspace_setup,
            sample_data=sample_data,
            _sandbox=None,
            _sandbox_id=None,
        )

    # ── Sandbox Lifecycle ─────────────────────────────────────────────────────

    async def _ensure_sandbox(self) -> modal.Sandbox:
        """Ensure sandbox exists, creating if needed."""
        if self._sandbox is not None:
            return self._sandbox

        import subprocess
        import sys

        import trio

        logger.info("Creating Modal sandbox...")

        # Run sandbox creation in subprocess to avoid asyncio conflicts
        sandbox_script = self._build_sandbox_creation_script()

        def _create_sandbox_subprocess() -> str:
            result = subprocess.run(
                [sys.executable, "-c", sandbox_script],
                capture_output=True,
                text=True,
                env={**os.environ},
            )
            if result.returncode != 0:
                raise RuntimeError(f"Sandbox creation failed: {result.stderr}")
            sandbox_id = result.stdout.strip().split("\n")[-1]
            return sandbox_id

        sandbox_id = await trio.to_thread.run_sync(_create_sandbox_subprocess)
        self._sandbox_id = sandbox_id

        # Reconnect from this process
        import modal

        self._sandbox = modal.Sandbox.from_id(sandbox_id)
        logger.info(f"Sandbox created: {self._sandbox_id}")

        # Run workspace setup if provided
        if self.workspace_setup:
            logger.info(f"Running workspace setup for {self._sandbox_id}...")
            try:
                await trio.to_thread.run_sync(
                    lambda: self.workspace_setup(self._sandbox, self.sample_data)
                )
                logger.info(f"Workspace setup complete for {self._sandbox_id}")
            except Exception as e:
                logger.error(f"Workspace setup FAILED for {self._sandbox_id}: {e}")
                raise
        else:
            print(f"No workspace_setup function provided for {self._sandbox_id}", file=sys.stderr)

        return self._sandbox

    def _build_sandbox_creation_script(self) -> str:
        """Build script to create sandbox in subprocess."""
        model_name = self.sandbox_config.model_name or ""

        return f'''
import asyncio
import modal

async def create_sandbox():
    app = modal.App.lookup("functional-extractor-sandbox", create_if_missing=True)

    gpu_type = "{self.sandbox_config.gpu_type}"
    if gpu_type in {{"B200", "GB200"}}:
        torch_index = "https://download.pytorch.org/whl/nightly/cu128"
    else:
        torch_index = "https://download.pytorch.org/whl/cu124"

    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04",
            add_python="3.11",
        )
        .apt_install("git", "build-essential")
        .pip_install(
            "torch",
            index_url=torch_index,
            extra_index_url="https://pypi.org/simple",
        )
        .pip_install(
            "transformers>=4.50",
            "accelerate",
            "safetensors",
            "numpy",
        )
        .env({{
            "HF_HOME": "/root/.cache/huggingface",
        }})
        .run_commands("mkdir -p /workspace")
    )

    sandbox = modal.Sandbox.create(
        app=app,
        image=image,
        gpu="{self.sandbox_config.gpu_type}",
        timeout={self.sandbox_config.timeout_seconds},
    )

    print(sandbox.object_id)

asyncio.run(create_sandbox())
'''

    async def close(self) -> None:
        """Terminate sandbox."""
        if self._sandbox is not None:
            logger.info(f"Terminating sandbox: {self._sandbox_id}")
            self._sandbox.terminate()
            self._sandbox = None
            self._sandbox_id = None

    async def __aenter__(self) -> ModalSandboxCodingEnvironment:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def cleanup(self) -> None:
        """Cleanup method expected by rollouts Environment protocol."""
        await self.close()

    # ── Tool Definitions ──────────────────────────────────────────────────────

    def get_tools(self) -> list:
        """Return CodingEnvironment-style tools."""
        from rollouts.dtypes import Tool, ToolFunction, ToolFunctionParameter

        return [
            Tool(
                type="function",
                function=ToolFunction(
                    name="bash",
                    description="Execute a bash command in the sandbox.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "command": {
                                "type": "string",
                                "description": "The command to execute",
                            },
                            "timeout": {
                                "type": "number",
                                "description": "Timeout in seconds (default: 120)",
                            },
                        },
                    ),
                    required=["command"],
                ),
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="read",
                    description="Read a file from the sandbox.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to read",
                            },
                            "offset": {
                                "type": "number",
                                "description": "Line number to start reading from (1-indexed)",
                            },
                            "limit": {
                                "type": "number",
                                "description": "Maximum number of lines to read",
                            },
                        },
                    ),
                    required=["file_path"],
                ),
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="write",
                    description="Write content to a file in the sandbox.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to write",
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file",
                            },
                        },
                    ),
                    required=["file_path", "content"],
                ),
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="edit",
                    description="Edit a file by replacing text.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to edit",
                            },
                            "old_string": {
                                "type": "string",
                                "description": "Text to find and replace",
                            },
                            "new_string": {
                                "type": "string",
                                "description": "Text to replace with",
                            },
                        },
                    ),
                    required=["file_path", "old_string", "new_string"],
                ),
            ),
        ]

    def requires_confirmation(self, tool_call: Any) -> bool:
        return False

    def get_tool_formatter(self, tool_name: str) -> None:
        return None

    async def on_session_start(self, session_id: str) -> None:
        """Create sandbox when session starts."""
        await self._ensure_sandbox()

    async def on_assistant_message(self, message: Any, state: Any) -> Any:
        return state

    def get_status_info(self) -> dict[str, str] | None:
        """Return status info for TUI."""
        if self._sandbox_id:
            return {"sandbox": self._sandbox_id[:12]}
        return None

    def get_system_prompt(self) -> str | None:
        """Return environment-specific system prompt."""
        return None

    # ── Tool Execution ────────────────────────────────────────────────────────

    async def exec_tool(
        self,
        tool_call: Any,
        current_state: Any,
        run_config: Any,
        checkpoint_store: Any = None,
        cancel_scope: trio.CancelScope | None = None,
    ) -> Any:
        """Execute a tool call in the sandbox."""
        from rollouts.dtypes import ToolResult

        import trio

        sandbox = await self._ensure_sandbox()

        try:
            if tool_call.name == "bash":
                return await self._exec_bash(sandbox, tool_call)
            elif tool_call.name == "read":
                return await self._exec_read(sandbox, tool_call)
            elif tool_call.name == "write":
                return await self._exec_write(sandbox, tool_call)
            elif tool_call.name == "edit":
                return await self._exec_edit(sandbox, tool_call)
            else:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content="",
                    error=f"Unknown tool: {tool_call.name}",
                )
        except Exception as e:
            logger.exception(f"Error executing {tool_call.name}")
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=str(e),
            )

    async def _exec_bash(self, sandbox: modal.Sandbox, tool_call: Any) -> Any:
        """Execute bash command in sandbox.

        Large outputs (>30KB) are written to a file in the sandbox instead of
        being returned directly. This prevents context overflow.
        """
        from rollouts.dtypes import ToolResult

        import trio

        command = tool_call.args.get("command", "")
        timeout = tool_call.args.get("timeout", 120)

        def run_command() -> tuple[str, str, int]:
            proc = sandbox.exec("bash", "-c", command, timeout=timeout)
            proc.wait()
            stdout = proc.stdout.read()
            stderr = proc.stderr.read()
            return stdout, stderr, proc.returncode

        stdout, stderr, returncode = await trio.to_thread.run_sync(run_command)

        output = stdout
        if stderr:
            output += f"\n[stderr]\n{stderr}"

        is_error = returncode != 0
        if is_error and not output.strip():
            output = f"Command exited with code {returncode}"

        # Handle large outputs - write to file in sandbox
        if len(output) > MAX_OUTPUT_SIZE:
            output = await self._handle_large_output(sandbox, output, tool_call.id)

        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=is_error,
            content=output,
            error=f"Exit code {returncode}" if is_error else None,
        )

    async def _handle_large_output(
        self, sandbox: modal.Sandbox, output: str, tool_call_id: str
    ) -> str:
        """Write large output to file in sandbox and return summary.

        Returns a truncated preview with instructions to read the full output.
        Uses chunked writing to avoid ARG_MAX limits.
        """
        import trio

        # Generate output file path
        safe_id = "".join(c for c in tool_call_id if c.isalnum() or c in "-_")[:32]
        output_path = f"/workspace/.tool_outputs/{safe_id}.txt"

        def write_output() -> None:
            # Create directory
            proc = sandbox.exec("bash", "-c", "mkdir -p /workspace/.tool_outputs", timeout=30)
            proc.wait()

            # Write in chunks to avoid ARG_MAX (65KB limit)
            # Each chunk must be small enough after base64 encoding (~48KB raw = ~64KB encoded)
            chunk_size = 40_000  # Safe chunk size

            for i in range(0, len(output), chunk_size):
                chunk = output[i : i + chunk_size]
                chunk_b64 = base64.b64encode(chunk.encode()).decode()

                # Use append mode (>>) for all but first chunk
                redirect = ">" if i == 0 else ">>"

                # Write chunk
                proc = sandbox.exec(
                    "bash",
                    "-c",
                    f"echo '{chunk_b64}' | base64 -d {redirect} {output_path}",
                    timeout=30,
                )
                proc.wait()

        await trio.to_thread.run_sync(write_output)

        # Build summary with head + tail preview
        total_lines = output.count("\n") + 1
        total_kb = len(output) // 1024

        preview_size = MAX_OUTPUT_SIZE // 2
        head = output[:preview_size]
        tail = output[-preview_size:]

        return (
            f"{head}\n\n"
            f"... [{total_kb}KB total, {total_lines} lines - full output saved to file]\n\n"
            f"... (last {preview_size // 1024}KB of output):\n\n"
            f"{tail}\n\n"
            f"Full output saved to: {output_path}\n"
            f"Use `bash command='grep PATTERN {output_path}'` to search, "
            f"or `bash command='head -100 {output_path}'` to see the start."
        )

    async def _exec_read(self, sandbox: modal.Sandbox, tool_call: Any) -> Any:
        """Read file from sandbox."""
        from rollouts.dtypes import ToolResult

        import trio

        file_path = tool_call.args.get("file_path", "")
        offset = tool_call.args.get("offset")
        limit = tool_call.args.get("limit")

        # Build cat command with optional head/tail
        if offset and limit:
            cmd = f"tail -n +{offset} {file_path} | head -n {limit}"
        elif offset:
            cmd = f"tail -n +{offset} {file_path}"
        elif limit:
            cmd = f"head -n {limit} {file_path}"
        else:
            cmd = f"cat {file_path}"

        def run_read() -> tuple[str, str, int]:
            proc = sandbox.exec("bash", "-c", cmd, timeout=30)
            proc.wait()
            return proc.stdout.read(), proc.stderr.read(), proc.returncode

        stdout, stderr, returncode = await trio.to_thread.run_sync(run_read)

        if returncode != 0:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=stderr or f"Failed to read {file_path}",
            )

        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content=stdout,
        )

    async def _exec_write(self, sandbox: modal.Sandbox, tool_call: Any) -> Any:
        """Write file to sandbox."""
        from rollouts.dtypes import ToolResult

        import trio

        file_path = tool_call.args.get("file_path", "")
        content = tool_call.args.get("content", "")

        # Base64 encode to handle special characters
        content_b64 = base64.b64encode(content.encode()).decode()

        # Ensure parent directory exists
        parent_dir = str(Path(file_path).parent)
        cmd = f"mkdir -p {parent_dir} && echo '{content_b64}' | base64 -d > {file_path}"

        def run_write() -> tuple[str, str, int]:
            proc = sandbox.exec("bash", "-c", cmd, timeout=30)
            proc.wait()
            return proc.stdout.read(), proc.stderr.read(), proc.returncode

        stdout, stderr, returncode = await trio.to_thread.run_sync(run_write)

        if returncode != 0:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=stderr or f"Failed to write {file_path}",
            )

        lines = len(content.split("\n"))
        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content=f"Wrote {lines} lines to {file_path}",
        )

    async def _exec_edit(self, sandbox: modal.Sandbox, tool_call: Any) -> Any:
        """Edit file in sandbox by replacing text."""
        from rollouts.dtypes import ToolResult

        import trio

        file_path = tool_call.args.get("file_path", "")
        old_string = tool_call.args.get("old_string", "")
        new_string = tool_call.args.get("new_string", "")

        # Read current content
        def read_file() -> tuple[str, int]:
            proc = sandbox.exec("cat", file_path, timeout=30)
            proc.wait()
            return proc.stdout.read(), proc.returncode

        content, returncode = await trio.to_thread.run_sync(read_file)

        if returncode != 0:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Failed to read {file_path}",
            )

        # Check that old_string exists exactly once
        count = content.count(old_string)
        if count == 0:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"old_string not found in {file_path}",
            )
        if count > 1:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"old_string found {count} times in {file_path} (must be unique)",
            )

        # Replace and write back
        new_content = content.replace(old_string, new_string, 1)
        content_b64 = base64.b64encode(new_content.encode()).decode()

        def write_file() -> tuple[str, int]:
            cmd = f"echo '{content_b64}' | base64 -d > {file_path}"
            proc = sandbox.exec("bash", "-c", cmd, timeout=30)
            proc.wait()
            return proc.stderr.read(), proc.returncode

        stderr, returncode = await trio.to_thread.run_sync(write_file)

        if returncode != 0:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=stderr or f"Failed to write {file_path}",
            )

        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content=f"Edited {file_path}",
        )

    # ── Serialization ─────────────────────────────────────────────────────────

    async def serialize(self) -> dict:
        """Serialize for checkpointing."""
        from dataclasses import asdict

        return {
            "env_kind": "modal_sandbox_coding",
            "version": "1.0",
            "sample_data": self.sample_data,
            "sandbox_config": asdict(self.sandbox_config),
            "sandbox_id": self._sandbox_id,
        }

    @classmethod
    async def deserialize(cls, data: dict) -> ModalSandboxCodingEnvironment:
        """Deserialize from checkpoint."""
        import modal

        sandbox_config = SandboxConfig(**data.get("sandbox_config", {}))

        env = cls(
            sandbox_config=sandbox_config,
            sample_data=data["sample_data"],
        )

        # Reconnect to existing sandbox if available
        sandbox_id = data.get("sandbox_id")
        if sandbox_id:
            try:
                env._sandbox = modal.Sandbox.from_id(sandbox_id)
                env._sandbox_id = sandbox_id
                logger.info(f"Reconnected to sandbox: {sandbox_id}")
            except Exception as e:
                logger.warning(f"Could not reconnect to sandbox {sandbox_id}: {e}")

        return env

    def copy_runtime_from(self, other: ModalSandboxCodingEnvironment) -> None:
        """Copy non-serializable runtime attributes from another instance.

        Called by agents.py after deserialize() to preserve function references
        and other runtime state that can't be serialized.
        """
        # Copy the workspace_setup function (not serializable)
        self.workspace_setup = other.workspace_setup
        # Copy sandbox reference if not reconnected
        if self._sandbox is None and other._sandbox is not None:
            self._sandbox = other._sandbox
            self._sandbox_id = other._sandbox_id
