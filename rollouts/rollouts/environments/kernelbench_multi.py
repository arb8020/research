"""KernelBench multi-turn environment for RL training.

This environment implements the Kevin-style multi-turn RL approach:
- Model generates kernel code in Python code blocks
- Environment compiles, tests, and benchmarks the kernel
- Feedback is returned to the model
- Model can iterate over multiple turns
- Final reward is based on correctness and speedup

Usage:
    from rollouts.environments.kernelbench_multi import KernelBenchMultiTurnEnvironment

    env = KernelBenchMultiTurnEnvironment(
        ref_code=ref_code,  # Reference PyTorch code
        backend="cuda",
        max_turns=8,
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
    ToolResult,
)
from .resources import KernelEvaluator

if TYPE_CHECKING:
    from ..gpu_sandbox import SandboxPool

logger = logging.getLogger(__name__)


@dataclass
class SandboxPoolKernelEvaluator:
    pool: SandboxPool | None = None

    def _get_pool(self) -> SandboxPool:
        if self.pool is None:
            from ..gpu_sandbox import SandboxPool

            self.pool = SandboxPool([])
        return self.pool

    async def start(self) -> None:
        pool = self._get_pool()
        if not pool._started:
            await pool.start()

    async def score_one(
        self,
        kernel_code: str,
        ref_code: str,
        timeout: float,
    ) -> dict[str, Any]:
        await self.start()
        return await self._get_pool().score_one(kernel_code, ref_code, timeout=timeout)


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
    evaluator: KernelEvaluator = field(default_factory=SandboxPoolKernelEvaluator, repr=False)

    # Evaluation state
    best_speedup: float = 0.0
    best_kernel: str | None = None
    has_correct_kernel: bool = False
    evaluator_provenance: dict[str, Any] | None = None

    async def serialize(self) -> dict:
        """Serialize environment state for checkpointing."""
        return {
            "env_kind": "kernelbench_multi",
            "ref_code": self.ref_code,
            "backend": self.backend,
            "max_turns": self.max_turns,
            "current_turn": self.current_turn,
            "turn_history": self.turn_history,
            "best_speedup": self.best_speedup,
            "best_kernel": self.best_kernel,
            "has_correct_kernel": self.has_correct_kernel,
            "evaluator_provenance": self.evaluator_provenance,
        }

    @staticmethod
    async def deserialize(data: dict) -> KernelBenchMultiTurnEnvironment:
        """Deserialize environment from checkpoint."""
        return KernelBenchMultiTurnEnvironment(
            ref_code=data["ref_code"],
            backend=data.get("backend", "cuda"),
            max_turns=data.get("max_turns", 8),
            current_turn=data.get("current_turn", 0),
            turn_history=data.get("turn_history", []),
            best_speedup=data.get("best_speedup", 0.0),
            best_kernel=data.get("best_kernel"),
            has_correct_kernel=data.get("has_correct_kernel", False),
            evaluator_provenance=data.get("evaluator_provenance"),
        )

    def get_tools(self) -> list[Tool]:
        """No tools needed - model generates code directly in responses."""
        return []

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        """No tools, so no confirmation needed."""
        return False

    def get_tool_formatter(self, tool_name: str) -> Any | None:
        """No custom formatters."""
        return None

    def get_status_info(self) -> dict[str, str] | None:
        """Show current turn and best speedup in TUI."""
        return {
            "env": "kernelbench-multi",
            "turn": f"{self.current_turn}/{self.max_turns}",
            "best_speedup": f"{self.best_speedup:.2f}x",
            "correct": "✓" if self.has_correct_kernel else "✗",
        }

    def get_system_prompt(self) -> str | None:
        """No additional system prompt - main prompt comes from task."""
        return None

    async def on_session_start(self, session_id: str) -> None:
        """Called when session starts. Ensure evaluation resource is ready."""
        await self.evaluator.start()

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
            result = await self.evaluator.score_one(
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

    async def on_assistant_message(self, message: Message, state: AgentState) -> AgentState:
        """Process model response, evaluate kernel, and provide feedback.

        This is the core multi-turn logic:
        1. Extract kernel code from response
        2. Evaluate it (compile, test, benchmark)
        3. Update state with results
        4. Return feedback message to model
        """
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
        current_metadata = dict(state.actor.trajectory.metadata)

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

            self.turn_history.append({
                "turn": self.current_turn,
                "has_code": False,
                "compiled": False,
                "correct": False,
                "speedup": 0.0,
            })
        else:
            # Evaluate the kernel
            result = await self._evaluate_kernel(kernel_code)
            runtime_provenance = result.get("runtime_provenance")
            if runtime_provenance is not None and self.evaluator_provenance is None:
                self.evaluator_provenance = runtime_provenance

            # Update tracking
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
            })

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
                # Update metadata with final results
                current_metadata.update({
                    "best_speedup": self.best_speedup,
                    "best_kernel": self.best_kernel,
                    "has_correct_kernel": self.has_correct_kernel,
                    "turns_used": self.current_turn,
                    "turn_history": self.turn_history,
                    "evaluator_provenance": self.evaluator_provenance,
                })

                new_trajectory = replace(
                    state.actor.trajectory,
                    metadata=current_metadata,
                )
                new_actor = replace(state.actor, trajectory=new_trajectory)

                if self.has_correct_kernel:
                    return replace(state, actor=new_actor, stop=StopReason.TASK_COMPLETED)
                else:
                    return replace(state, actor=new_actor, stop=StopReason.MAX_TURNS)

        # Add feedback as user message and update metadata
        feedback_msg = Message(role="user", content=feedback)
        current_metadata.update({
            "best_speedup": self.best_speedup,
            "has_correct_kernel": self.has_correct_kernel,
            "turns_used": self.current_turn,
            "turn_history": self.turn_history,
            "evaluator_provenance": self.evaluator_provenance,
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
        """No tools to execute in this environment."""
        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=True,
            content="",
            error="This environment does not support tool calls. Please write kernel code directly in your response.",
        )
