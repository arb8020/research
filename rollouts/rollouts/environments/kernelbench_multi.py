"""KernelBench multi-turn environment for RL training.

This environment implements the Kevin-style multi-turn RL approach:
- Model generates kernel code in Python code blocks
- Environment compiles, tests, and benchmarks the kernel
- Feedback is returned to the model
- Model can iterate over multiple turns
- Final reward is based on correctness and speedup

Usage:
    from rollouts.environments.kernelbench_multi import KernelBenchMultiTurnEnvironment
    from rollouts.environments.kernelbench_multi import SandboxPoolKernelEvaluator
    from rollouts.gpu_sandbox import SandboxPool

    pool = SandboxPool([])
    env = KernelBenchMultiTurnEnvironment(
        ref_code=ref_code,  # Reference PyTorch code
        backend="cuda",
        max_turns=8,
        evaluator=SandboxPoolKernelEvaluator(pool),
    )
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import trio

from ..agents import AgentState, RunConfig
from ..core import (
    Message,
    StopReason,
    TextContent,
    Tool,
    ToolCall,
    ToolFunction,
    ToolFunctionParameter,
    ToolResult,
)
from ..gpu_sandbox import SandboxPool
from ..gpu_sandbox.config import deserialize_sandbox_config
from ..gpu_sandbox.worker import _build_scoring_script_for_remote, _parse_scoring_output
from .modal_sandbox_resource import ManagedModalSandboxResource
from .resources import KernelEvaluator, SandboxWorkspaceResource

if TYPE_CHECKING:
    from ..gpu_sandbox import SandboxPool

logger = logging.getLogger(__name__)


@dataclass
class SandboxPoolKernelEvaluator:
    pool: SandboxPool

    async def start(self) -> None:
        await self.pool.ensure_capacity()

    async def score_one(
        self,
        kernel_code: str,
        ref_code: str,
        timeout: float,
    ) -> dict[str, Any]:
        await self.start()
        return await self.pool.score_one(kernel_code, ref_code, timeout=timeout)

    async def score_batch(
        self,
        requests: list[dict[str, Any]],
        timeout: float,
    ) -> list[dict[str, Any]]:
        await self.start()
        return await self.pool.score_batch(requests, timeout=timeout)

    def stats(self) -> dict[str, Any]:
        return self.pool.stats()


@dataclass
class SandboxWorkspaceKernelEvaluator:
    workspace: SandboxWorkspaceResource
    working_dir: str = "/workspace"

    async def start(self) -> None:
        await self.workspace.start()

    async def score_one(
        self,
        kernel_code: str,
        ref_code: str,
        timeout: float,
    ) -> dict[str, Any]:
        import base64

        script = _build_scoring_script_for_remote(
            ref_code=ref_code,
            kernel_b64=base64.b64encode(kernel_code.encode()).decode(),
        )
        command = f"""
python3 << 'SCORING_SCRIPT_EOF'
{script}
SCORING_SCRIPT_EOF
"""
        result = await self.workspace.run(
            command,
            cwd=self.working_dir,
            timeout=timeout,
        )
        return _parse_scoring_output(result.stdout, result.stderr, result.returncode)

    def stats(self) -> dict[str, Any]:
        return {
            "kind": "sandbox_workspace_kernel_evaluator",
            "workspace": self.workspace.stats(),
        }


@dataclass
class KernelBenchMultiTurnEnvironment:
    """Multi-turn environment for KernelBench kernel optimization.

    The model generates kernel code in Python code blocks. The environment:
    1. Extracts code from the model's response
    2. Compiles and evaluates the kernel via SandboxPool
    3. Returns feedback (compilation status, correctness, speedup)
    4. Allows multiple refinement turns

    Args:
        ref_code: Reference PyTorch code with Model, get_inputs, get_init_inputs
        backend: "cuda" or "hip"
        max_turns: Maximum number of turns allowed
        current_turn: Current turn number (starts at 0)
        turn_history: History of turns with their results
    """

    ref_code: str
    backend: str = "cuda"
    max_turns: int = 8
    current_turn: int = 0
    turn_history: list[dict[str, Any]] = field(default_factory=list)
    evaluator: KernelEvaluator | None = field(default=None, repr=False)
    kernel_workspace: SandboxWorkspaceResource | None = field(default=None, repr=False)
    evaluator_spec: dict[str, Any] | None = None
    kernel_workspace_state: dict[str, Any] | None = None
    submission_path: str = "/workspace/kernel_submission.py"

    # Evaluation state
    best_speedup: float = 0.0
    best_kernel: str | None = None
    has_correct_kernel: bool = False
    evaluator_provenance: dict[str, Any] | None = None
    sandbox_runtime_provenance: dict[str, Any] | None = None
    sandbox_resource_stats: dict[str, Any] | None = None
    runtime_requirements: Any | None = None

    async def serialize(self) -> dict:
        """Serialize environment state for checkpointing."""
        return {
            "env_kind": "kernelbench_multi",
            "ref_code": self.ref_code,
            "backend": self.backend,
            "max_turns": self.max_turns,
            "current_turn": self.current_turn,
            "turn_history": self.turn_history,
            "evaluator_spec": self.evaluator_spec,
            "kernel_workspace_state": (
                self.kernel_workspace.serialize_state()
                if self.kernel_workspace is not None
                and hasattr(self.kernel_workspace, "serialize_state")
                else self.kernel_workspace_state
            ),
            "best_speedup": self.best_speedup,
            "best_kernel": self.best_kernel,
            "has_correct_kernel": self.has_correct_kernel,
            "evaluator_provenance": self.evaluator_provenance,
            "sandbox_runtime_provenance": self.sandbox_runtime_provenance,
            "sandbox_resource_stats": self.sandbox_resource_stats,
            "submission_path": self.submission_path,
        }

    @staticmethod
    async def deserialize(
        data: dict,
        evaluator: KernelEvaluator | None = None,
    ) -> KernelBenchMultiTurnEnvironment:
        """Deserialize environment from checkpoint."""
        evaluator_spec = data.get("evaluator_spec")
        if evaluator is None and evaluator_spec is not None:
            configs = [
                deserialize_sandbox_config(config_data)
                for config_data in evaluator_spec.get("sandbox_configs", [])
            ]
            evaluator = SandboxPoolKernelEvaluator(SandboxPool(configs))
        kernel_workspace_state = data.get("kernel_workspace_state")
        kernel_workspace = None
        if kernel_workspace_state is not None:
            kind = kernel_workspace_state.get("kind")
            if kind == "managed_modal_sandbox_resource":
                kernel_workspace = ManagedModalSandboxResource.deserialize_state(
                    kernel_workspace_state
                )

        return KernelBenchMultiTurnEnvironment(
            ref_code=data["ref_code"],
            backend=data.get("backend", "cuda"),
            max_turns=data.get("max_turns", 8),
            current_turn=data.get("current_turn", 0),
            turn_history=data.get("turn_history", []),
            evaluator_spec=evaluator_spec,
            kernel_workspace_state=kernel_workspace_state,
            best_speedup=data.get("best_speedup", 0.0),
            best_kernel=data.get("best_kernel"),
            has_correct_kernel=data.get("has_correct_kernel", False),
            evaluator_provenance=data.get("evaluator_provenance"),
            sandbox_runtime_provenance=data.get("sandbox_runtime_provenance"),
            sandbox_resource_stats=data.get("sandbox_resource_stats"),
            submission_path=data.get("submission_path", "/workspace/kernel_submission.py"),
            evaluator=evaluator,
            kernel_workspace=kernel_workspace,
        )

    def get_tools(self) -> list[Tool]:
        """Expose a minimal structured action surface when a sandbox is injected."""
        if self.kernel_workspace is None:
            return []
        return [
            Tool(
                type="function",
                function=ToolFunction(
                    name="write_kernel",
                    description=(
                        "Write a complete `class ModelNew` implementation to the sandbox and "
                        "evaluate it."
                    ),
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "kernel_code": {
                                "type": "string",
                                "description": (
                                    "Full Python source containing `class ModelNew`."
                                ),
                            }
                        },
                    ),
                    required=["kernel_code"],
                ),
            )
        ]

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        """No tools, so no confirmation needed."""
        return False

    def get_tool_formatter(self, tool_name: str) -> Any | None:
        """No custom formatters."""
        return None

    def get_status_info(self) -> dict[str, str] | None:
        """Show current turn and best speedup in TUI."""
        status = {
            "env": "kernelbench-multi",
            "turn": f"{self.current_turn}/{self.max_turns}",
            "best_speedup": f"{self.best_speedup:.2f}x",
            "correct": "✓" if self.has_correct_kernel else "✗",
        }
        if self.kernel_workspace is not None:
            status["mode"] = "write_kernel"
        return status

    def get_system_prompt(self) -> str | None:
        """Explain the tool contract when running with a sandbox resource."""
        if self.kernel_workspace is None:
            return None
        return (
            "Use the `write_kernel` tool to submit a complete `class ModelNew` implementation. "
            "Each tool call writes the code to the sandbox and returns compilation, correctness, "
            "and speed feedback. The selected backend for this task is "
            f"`{self.backend}`; prefer implementations that stay within that backend instead of "
            "switching frameworks mid-episode. The sandbox is preflighted against the selected "
            "backend's runtime contract before the first turn, so treat missing dependency "
            "failures as environment issues rather than a prompt to pivot frameworks."
        )

    async def cleanup(self) -> None:
        if self.kernel_workspace is not None:
            await self.kernel_workspace.close()

    async def initialize(self, session_id: str | None = None) -> None:
        """Eagerly acquire/check required resources before the first model turn."""
        del session_id
        await self._require_evaluator().start()
        await self._ensure_workspace_runtime()

    async def on_session_start(self, session_id: str) -> None:
        """Compatibility shim for older runtime paths."""
        await self.initialize(session_id)

    def _require_evaluator(self) -> KernelEvaluator:
        if self.evaluator is None and self.kernel_workspace is not None:
            self.evaluator = SandboxWorkspaceKernelEvaluator(
                workspace=self.kernel_workspace,
                working_dir=self.kernel_workspace.working_dir,
            )
        if self.evaluator is None:
            raise ValueError(
                "KernelBenchMultiTurnEnvironment requires an injected evaluator. "
                "Construct it with evaluator=... or pass evaluator=... to deserialize()."
            )
        return self.evaluator

    async def _ensure_workspace_runtime(self) -> None:
        if self.kernel_workspace is None:
            return
        if self.sandbox_resource_stats is not None and self.sandbox_runtime_provenance is not None:
            return
        await self.kernel_workspace.start()
        self.sandbox_runtime_provenance = await self.kernel_workspace.describe_runtime()
        self.sandbox_resource_stats = self.kernel_workspace.stats()
        if self.runtime_requirements is not None:
            errors = self.runtime_requirements.validate_worker(
                self.sandbox_runtime_provenance,
                is_local=False,
            )
            if errors:
                raise ValueError(
                    "KernelBench sandbox requirements not met:\n- " + "\n- ".join(errors)
                )

    def get_runtime_metadata(self) -> dict[str, Any]:
        return {
            "best_speedup": self.best_speedup,
            "has_correct_kernel": self.has_correct_kernel,
            "turn_history": self.turn_history,
            "evaluator_provenance": self.evaluator_provenance,
            "sandbox_runtime_provenance": self.sandbox_runtime_provenance,
            "sandbox_resource_stats": self.sandbox_resource_stats,
        }

    def _extract_kernel_code(self, response: str) -> str | None:
        """Extract Python kernel code from model response.

        Tries multiple patterns in order:
        1. ```python ... ``` code blocks
        2. <kernel> ... </kernel> tags
        3. Raw code containing "class ModelNew"

        Returns:
            Extracted code or None if no valid code found
        """
        # Try ```python blocks first
        pattern = r"```python\s*(.*?)\s*```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if "class ModelNew" in code:
                return code

        # Try <kernel> tags
        pattern = r"<kernel>\s*(.*?)\s*</kernel>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if "class ModelNew" in code:
                return code

        # Last resort: look for class ModelNew
        if "class ModelNew" in response:
            start = response.find("class ModelNew")
            code = response[start:]
            # Remove any trailing text that's not part of the class
            lines = code.split("\n")
            result_lines = []
            indent_level = None
            for line in lines:
                if indent_level is None and line.strip().startswith("class ModelNew"):
                    indent_level = len(line) - len(line.lstrip())
                    result_lines.append(line)
                elif indent_level is not None:
                    if (
                        line.strip()
                        and not line.startswith(" " * indent_level)
                        and not line.startswith("\t")
                    ):
                        # New top-level construct
                        break
                    result_lines.append(line)
            return "\n".join(result_lines)

        return None

    async def _evaluate_kernel(self, kernel_code: str) -> dict[str, Any]:
        """Evaluate kernel code using the injected evaluator.

        Compiles, tests correctness, and benchmarks the kernel.

        Returns:
            Dict with keys: compiled, correct, speedup, error, pass_rate
        """
        logger.info(f"[KernelBench] Evaluating kernel ({len(kernel_code)} chars)")
        logger.info(f"[KernelBench] Kernel code preview: {kernel_code[:200]}...")

        try:
            result = await self._require_evaluator().score_one(
                kernel_code,
                self.ref_code,
                timeout=120.0,
            )
            logger.info(f"[KernelBench] Score result: {result}")
            return {
                "compiled": result.get("compiled", 0.0) > 0.5,
                "correct": result.get("correct", 0.0) > 0.5,
                "speedup": result.get("speedup", 0.0),
                "pass_rate": result.get("pass_rate", 0.0),
                "error": result.get("error"),
                "runtime_provenance": result.get("runtime_provenance"),
                "debug_stdout_tail": result.get("debug_stdout_tail"),
                "debug_stderr_tail": result.get("debug_stderr_tail"),
                "returncode": result.get("returncode"),
            }
        except Exception as e:
            logger.exception(f"Kernel evaluation failed: {e}")
            return {
                "compiled": False,
                "correct": False,
                "speedup": 0.0,
                "pass_rate": 0.0,
                "error": str(e),
                "runtime_provenance": None,
                "debug_stdout_tail": None,
                "debug_stderr_tail": None,
                "returncode": None,
            }

    def _format_feedback(
        self,
        turn: int,
        compiled: bool,
        correct: bool,
        speedup: float,
        error: str | None,
    ) -> str:
        """Format evaluation feedback for the model."""
        lines = [f"\n### Turn {turn} Evaluation Results\n"]

        if not compiled:
            lines.append("**Compilation:** ❌ FAILED")
            if error:
                lines.append(f"\n**Error:** {error}")
            lines.append("\nPlease fix the compilation errors and try again.")
        elif not correct:
            lines.append("**Compilation:** ✅ PASSED")
            lines.append("**Correctness:** ❌ FAILED")
            lines.append("\nThe kernel compiles but produces incorrect outputs.")
            lines.append("Please verify your implementation matches the reference.")
        else:
            lines.append("**Compilation:** ✅ PASSED")
            lines.append("**Correctness:** ✅ PASSED")
            lines.append(f"**Speedup:** {speedup:.2f}x over PyTorch baseline")

            if speedup > 1.0:
                lines.append("\nYour kernel is faster than the baseline!")
            else:
                lines.append("\nThe kernel is correct but slower than baseline.")
                lines.append("Consider optimizations like:")
                lines.append("- Tiling/blocking for better memory access")
                lines.append("- Shared memory usage")
                lines.append("- Coalesced memory access patterns")
                lines.append("- Loop unrolling")

        # Show best so far
        if self.best_speedup > 0:
            lines.append(f"\n**Best so far:** {self.best_speedup:.2f}x")

        # Show remaining turns
        remaining = self.max_turns - self.current_turn
        lines.append(f"\n**Remaining turns:** {remaining}")

        if remaining > 0 and not (correct and speedup > 1.5):
            lines.append("\nYou may continue refining your kernel or submit the current best.")
            lines.append("To submit, say 'I am done' or provide your final kernel.")

        return "\n".join(lines)

    def _current_metadata(self, state: AgentState) -> tuple[dict[str, Any], dict[str, Any]]:
        current_metadata = dict(state.actor.trajectory.metadata)
        sample_data = current_metadata.get("sample_data", {})
        if isinstance(sample_data, dict):
            current_metadata.setdefault("problem_id", sample_data.get("problem_id"))
            current_metadata.setdefault("name", sample_data.get("name"))
            current_metadata.setdefault("level", sample_data.get("level"))
            return current_metadata, sample_data
        return current_metadata, {}

    def _record_result(self, kernel_code: str | None, result: dict[str, Any] | None) -> None:
        if result is not None:
            runtime_provenance = result.get("runtime_provenance")
            if runtime_provenance is not None and self.evaluator_provenance is None:
                self.evaluator_provenance = runtime_provenance

            if result["correct"] and result["speedup"] > self.best_speedup:
                self.best_speedup = result["speedup"]
                self.best_kernel = kernel_code
                self.has_correct_kernel = True

            if result["correct"]:
                self.has_correct_kernel = True

            self.turn_history.append({
                "turn": self.current_turn,
                "has_code": True,
                "compiled": result["compiled"],
                "correct": result["correct"],
                "speedup": result["speedup"],
                "error": result.get("error"),
                "runtime_provenance": runtime_provenance,
                "sandbox_runtime_provenance": self.sandbox_runtime_provenance,
                "sandbox_resource_stats": self.sandbox_resource_stats,
                "submission_path": self.submission_path,
                "debug_stdout_tail": result.get("debug_stdout_tail"),
                "debug_stderr_tail": result.get("debug_stderr_tail"),
                "returncode": result.get("returncode"),
            })
            return

        self.turn_history.append({
            "turn": self.current_turn,
            "has_code": False,
            "compiled": False,
            "correct": False,
            "speedup": 0.0,
            "error": "No kernel code found in assistant response.",
            "sandbox_runtime_provenance": self.sandbox_runtime_provenance,
            "sandbox_resource_stats": self.sandbox_resource_stats,
        })

    def _finalize_state(
        self,
        state: AgentState,
        *,
        error: str | None = None,
        debug_stdout_tail: str | None = None,
        debug_stderr_tail: str | None = None,
        returncode: int | None = None,
    ) -> AgentState:
        current_metadata, sample_data = self._current_metadata(state)
        current_metadata.update({
            "problem_id": sample_data.get("problem_id"),
            "name": sample_data.get("name"),
            "level": sample_data.get("level"),
            "best_speedup": self.best_speedup,
            "best_kernel": self.best_kernel,
            "has_correct_kernel": self.has_correct_kernel,
            "turns_used": self.current_turn,
            "turn_history": self.turn_history,
            "evaluator_provenance": self.evaluator_provenance,
            "sandbox_runtime_provenance": self.sandbox_runtime_provenance,
            "sandbox_resource_stats": self.sandbox_resource_stats,
            "submission_path": self.submission_path,
            "error": error,
            "debug_stdout_tail": debug_stdout_tail,
            "debug_stderr_tail": debug_stderr_tail,
            "returncode": returncode,
        })
        return replace(
            state,
            actor=replace(
                state.actor,
                trajectory=replace(state.actor.trajectory, metadata=current_metadata),
            ),
        )

    async def _handle_write_kernel_turn(self, message: Message, state: AgentState) -> AgentState:
        if state.pending_tool_calls:
            return state

        response_text = ""
        if isinstance(message.content, str):
            response_text = message.content
        elif isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, TextContent):
                    response_text += block.text

        stop_phrases = [
            "i am done",
            "i'm done",
            "final answer",
            "submission",
            "i submit",
        ]
        wants_to_stop = any(phrase in response_text.lower() for phrase in stop_phrases)
        if wants_to_stop or self.current_turn >= self.max_turns:
            finalized = self._finalize_state(state)
            stop_reason = StopReason.TASK_COMPLETED if self.has_correct_kernel else StopReason.MAX_TURNS
            return replace(finalized, stop=stop_reason)
        return state

    async def on_assistant_message(self, message: Message, state: AgentState) -> AgentState:
        """Process model response, evaluate kernel, and provide feedback.

        This is the core multi-turn logic:
        1. Extract kernel code from response
        2. Evaluate it (compile, test, benchmark)
        3. Update state with results
        4. Return feedback message to model
        """
        if self.kernel_workspace is not None:
            return await self._handle_write_kernel_turn(message, state)

        # Increment turn counter
        self.current_turn += 1

        # Extract response text
        response_text = ""
        if isinstance(message.content, str):
            response_text = message.content
        elif isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, TextContent):
                    response_text += block.text

        # Check if model wants to stop
        stop_phrases = [
            "i am done",
            "i'm done",
            "final answer",
            "submission",
            "i submit",
        ]
        wants_to_stop = any(phrase in response_text.lower() for phrase in stop_phrases)

        # Extract kernel code
        kernel_code = self._extract_kernel_code(response_text)
        logger.info(
            f"[KernelBench] Extracted kernel_code: {kernel_code is not None}, len={len(kernel_code) if kernel_code else 0}"
        )

        # Get current trajectory metadata
        current_metadata, sample_data = self._current_metadata(state)

        if kernel_code is None:
            # No kernel code found
            feedback = (
                f"\n### Turn {self.current_turn}\n\n"
                "**No kernel code found.**\n\n"
                "Please provide your kernel implementation in a Python code block:\n"
                "```python\n"
                "class ModelNew(nn.Module):\n"
                "    ...\n"
                "```\n"
            )
            self._record_result(None, None)
        else:
            # Evaluate the kernel
            result = await self._evaluate_kernel(kernel_code)
            self._record_result(kernel_code, result)

            # Format feedback
            feedback = self._format_feedback(
                turn=self.current_turn,
                compiled=result["compiled"],
                correct=result["correct"],
                speedup=result["speedup"],
                error=result.get("error"),
            )

            # Check if we should stop
            if wants_to_stop or self.current_turn >= self.max_turns:
                finalized = self._finalize_state(
                    state,
                    error=result.get("error"),
                    debug_stdout_tail=result.get("debug_stdout_tail"),
                    debug_stderr_tail=result.get("debug_stderr_tail"),
                    returncode=result.get("returncode"),
                )
                if self.has_correct_kernel:
                    return replace(finalized, stop=StopReason.TASK_COMPLETED)
                return replace(finalized, stop=StopReason.MAX_TURNS)

        # Add feedback as user message and update metadata
        feedback_msg = Message(role="user", content=feedback)
        current_metadata.update({
            "problem_id": sample_data.get("problem_id"),
            "name": sample_data.get("name"),
            "level": sample_data.get("level"),
            "best_speedup": self.best_speedup,
            "has_correct_kernel": self.has_correct_kernel,
            "turns_used": self.current_turn,
            "turn_history": self.turn_history,
            "evaluator_provenance": self.evaluator_provenance,
            "sandbox_runtime_provenance": self.sandbox_runtime_provenance,
            "sandbox_resource_stats": self.sandbox_resource_stats,
        })

        new_trajectory = replace(
            state.actor.trajectory,
            messages=state.actor.trajectory.messages + [feedback_msg],
            metadata=current_metadata,
        )

        return replace(
            state,
            actor=replace(state.actor, trajectory=new_trajectory),
        )

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: AgentState,
        run_config: RunConfig,
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult:
        del run_config, cancel_scope
        if self.kernel_workspace is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error="This environment does not support tool calls. Please write kernel code directly in your response.",
            )
        if tool_call.name != "write_kernel":
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Unknown tool: {tool_call.name}",
            )

        kernel_code = str(tool_call.args.get("kernel_code", "")).strip()
        if not kernel_code:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error="write_kernel requires non-empty `kernel_code`.",
            )

        await self._ensure_workspace_runtime()
        self.current_turn += 1
        await self.kernel_workspace.write_file(self.submission_path, kernel_code.encode())
        result = await self._evaluate_kernel(kernel_code)
        self._record_result(kernel_code, result)
        feedback = self._format_feedback(
            turn=self.current_turn,
            compiled=result["compiled"],
            correct=result["correct"],
            speedup=result["speedup"],
            error=result.get("error"),
        )
        stop_reason = None
        if self.current_turn >= self.max_turns:
            stop_reason = StopReason.TASK_COMPLETED if self.has_correct_kernel else StopReason.MAX_TURNS

        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content=feedback,
            stop_reason=stop_reason,
            details={
                "submission_path": self.submission_path,
                "compiled": result["compiled"],
                "correct": result["correct"],
                "speedup": result["speedup"],
                "pass_rate": result["pass_rate"],
                "error": result.get("error"),
                "runtime_provenance": result.get("runtime_provenance"),
                "sandbox_runtime_provenance": self.sandbox_runtime_provenance,
                "sandbox_resource_stats": self.sandbox_resource_stats,
                "debug_stdout_tail": result.get("debug_stdout_tail"),
                "debug_stderr_tail": result.get("debug_stderr_tail"),
                "returncode": result.get("returncode"),
            },
        )
