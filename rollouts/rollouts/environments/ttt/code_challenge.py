"""Single-turn code challenge environment for TTT.

Generate code → run it → grade correctness + speed → return reward.
No multi-turn conversation - all iteration happens at the TTT loop level.

Implements the rollouts Environment protocol with no tools - grading happens
in on_assistant_message after the model responds.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import trio

from ...dtypes import (
    AgentState,
    Message,
    Metric,
    RunConfig,
    Score,
    StopReason,
    Tool,
    ToolCall,
    ToolResult,
)
from .._subprocess import run_command

if TYPE_CHECKING:
    pass


# ── Grading Infrastructure ────────────────────────────────────────────────────


@dataclass(frozen=True)
class TestCase:
    """A single test case: input args and expected output."""

    args: tuple[Any, ...]
    expected: Any

    def format_call(self, fn_name: str) -> str:
        """Format as a function call string."""
        args_str = ", ".join(repr(a) for a in self.args)
        return f"{fn_name}({args_str})"


@dataclass
class GradeResult:
    """Result of grading a code submission."""

    passed: int
    total: int
    runtime_ms: float
    error: str | None = None
    timed_out: bool = False

    @property
    def correctness(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


def extract_code(response: str, language: str = "python") -> str | None:
    """Extract code from markdown code block."""
    # Try ```python ... ``` first
    pattern = rf"```{language}\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()  # Last code block

    # Try ``` ... ``` (no language specified)
    pattern = r"```\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # No code block found - maybe raw code?
    if "def " in response or "import " in response:
        return response.strip()

    return None


def build_test_script(code: str, fn_name: str, test_cases: list[TestCase]) -> str:
    """Build a Python script that runs test cases and reports results."""
    test_calls = []
    for i, tc in enumerate(test_cases):
        call = tc.format_call(fn_name)
        test_calls.append(f"""
try:
    result = {call}
    expected = {repr(tc.expected)}
    if result == expected:
        passed += 1
    else:
        failures.append(f"Test {i}: {call} = {{result}}, expected {{expected}}")
except Exception as e:
    failures.append(f"Test {i}: {call} raised {{type(e).__name__}}: {{e}}")
""")

    script = f"""
import time

{code}

passed = 0
failures = []

start = time.perf_counter()
{"".join(test_calls)}
elapsed_ms = (time.perf_counter() - start) * 1000

print(f"PASSED:{{passed}}/{{len(failures) + passed}}")
print(f"RUNTIME_MS:{{elapsed_ms:.3f}}")
for f in failures[:5]:  # Limit failure output
    print(f"FAILURE:{{f}}")
"""
    return script


async def run_code_with_tests(
    code: str,
    fn_name: str,
    test_cases: list[TestCase],
    timeout: float = 10.0,
) -> GradeResult:
    """Run code against test cases in a subprocess."""
    script = build_test_script(code, fn_name, test_cases)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        returncode, stdout, stderr = await run_command(
            f"python {script_path}",
            cwd=str(Path(script_path).parent),
            timeout=timeout,
        )
    except TimeoutError:
        return GradeResult(
            passed=0,
            total=len(test_cases),
            runtime_ms=timeout * 1000,
            timed_out=True,
            error="Execution timed out",
        )
    finally:
        Path(script_path).unlink(missing_ok=True)

    # Parse output
    passed = 0
    total = len(test_cases)
    runtime_ms = 0.0
    error = None

    if returncode != 0:
        error = stderr.strip() or stdout.strip() or f"Exit code {returncode}"
        return GradeResult(passed=0, total=total, runtime_ms=0, error=error)

    for line in stdout.split("\n"):
        if line.startswith("PASSED:"):
            parts = line.split(":")[1].split("/")
            passed = int(parts[0])
            total = int(parts[1])
        elif line.startswith("RUNTIME_MS:"):
            runtime_ms = float(line.split(":")[1])
        elif line.startswith("FAILURE:"):
            if error is None:
                error = line.split(":", 1)[1]

    return GradeResult(passed=passed, total=total, runtime_ms=runtime_ms, error=error)


def _extract_text_content(content: str | list[Any]) -> str:
    """Extract text from message content (str or list of ContentBlocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if hasattr(block, "text"):
                texts.append(block.text)
            elif isinstance(block, dict) and "text" in block:
                texts.append(block["text"])
        return "\n".join(texts)
    return str(content) if content else ""


# ── Environment Implementation ────────────────────────────────────────────────


@dataclass
class CodeChallengeEnvironment:
    """Single-turn code challenge environment (rollouts Environment protocol).

    No tools - model generates code directly in its response.
    Grading happens in on_assistant_message, which extracts code and runs tests.

    Usage:
        env = CodeChallengeEnvironment(
            prompt="Write a function `add(a, b)` that returns a + b",
            fn_name="add",
            test_cases=[TestCase((1, 2), 3), TestCase((0, 0), 0)],
        )
        # Use with run_agent() - grading happens automatically
    """

    prompt: str
    fn_name: str
    test_cases: list[TestCase]
    timeout: float = 10.0
    speed_bonus_weight: float = 0.1

    # Grading result (populated after on_assistant_message)
    _last_score: Score | None = field(default=None, repr=False)

    def get_name(self) -> str:
        return "code_challenge"

    def get_tools(self) -> list[Tool]:
        """No tools - model writes code directly in response."""
        return []

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        return False

    def get_tool_formatter(self, tool_name: str) -> Any:
        return None

    def get_system_prompt(self) -> str | None:
        """Return prompt as system message."""
        return None

    def get_initial_user_message(self) -> str:
        """Return the problem prompt as the initial user message."""
        return self.prompt

    async def on_assistant_message(self, message: Message, state: AgentState) -> AgentState:
        """Grade the assistant's response and stop."""
        content = _extract_text_content(message.content)
        self._last_score = await self._grade(content)

        # Single turn - always stop after grading
        return replace(state, stop=StopReason.TASK_COMPLETED)

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: AgentState,
        run_config: RunConfig,
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult:
        """No tools in this environment."""
        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=True,
            content="",
            error="CodeChallengeEnvironment has no tools",
        )

    async def serialize(self) -> dict:
        """Serialize environment state."""
        return {
            "env_kind": "code_challenge",
            "prompt": self.prompt,
            "fn_name": self.fn_name,
            "test_cases": [(tc.args, tc.expected) for tc in self.test_cases],
            "timeout": self.timeout,
            "speed_bonus_weight": self.speed_bonus_weight,
        }

    @staticmethod
    async def deserialize(data: dict) -> CodeChallengeEnvironment:
        """Deserialize environment state."""
        return CodeChallengeEnvironment(
            prompt=data["prompt"],
            fn_name=data["fn_name"],
            test_cases=[TestCase(tuple(tc[0]), tc[1]) for tc in data["test_cases"]],
            timeout=data.get("timeout", 10.0),
            speed_bonus_weight=data.get("speed_bonus_weight", 0.1),
        )

    # ── Grading ───────────────────────────────────────────────────────────────

    async def _grade(self, response: str) -> Score:
        """Grade a model response. Returns Score with correctness + speed metrics."""
        code = extract_code(response)

        if code is None:
            return Score(
                metrics=(
                    Metric("correctness", 0.0, weight=1.0, metadata={"error": "No code found"}),
                    Metric("runtime_ms", 0.0, weight=0),
                    Metric("speed_bonus", 0.0, weight=self.speed_bonus_weight),
                )
            )

        result = await run_code_with_tests(code, self.fn_name, self.test_cases, self.timeout)

        # Speed bonus: faster = higher (normalized by timeout)
        # 0ms -> 1.0, timeout -> 0.0
        if result.timed_out:
            speed_bonus = 0.0
        else:
            speed_bonus = max(0.0, 1.0 - result.runtime_ms / (self.timeout * 1000))

        return Score(
            metrics=(
                Metric(
                    "correctness",
                    result.correctness,
                    weight=1.0,
                    metadata={
                        "passed": result.passed,
                        "total": result.total,
                        "error": result.error,
                    },
                ),
                Metric("runtime_ms", result.runtime_ms, weight=0),
                Metric("speed_bonus", speed_bonus, weight=self.speed_bonus_weight),
            )
        )

    def get_last_score(self) -> Score | None:
        """Get the score from the last grading (after on_assistant_message)."""
        return self._last_score


# ── Fibonacci Environment ─────────────────────────────────────────────────────

FIBONACCI_TEST_CASES = [
    TestCase((0,), 0),
    TestCase((1,), 1),
    TestCase((2,), 1),
    TestCase((5,), 5),
    TestCase((10,), 55),
    TestCase((20,), 6765),
    TestCase((35,), 9227465),  # Slow for naive O(2^n)
    TestCase((40,), 102334155),  # Will timeout for naive O(2^n)
]

FIBONACCI_PROMPT = """Write a Python function `fib(n)` that returns the Nth Fibonacci number.

The Fibonacci sequence is: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...
- fib(0) = 0
- fib(1) = 1
- fib(n) = fib(n-1) + fib(n-2) for n > 1

Your solution should be efficient enough to compute fib(30) quickly.

```python
def fib(n):
    # Your implementation here
```
"""


def FibonacciEnvironment(
    timeout: float = 5.0,
    speed_bonus_weight: float = 0.1,
) -> CodeChallengeEnvironment:
    """Create a Fibonacci code challenge environment."""
    return CodeChallengeEnvironment(
        prompt=FIBONACCI_PROMPT,
        fn_name="fib",
        test_cases=FIBONACCI_TEST_CASES,
        timeout=timeout,
        speed_bonus_weight=speed_bonus_weight,
    )


# ── Backward Compat ───────────────────────────────────────────────────────────
# Keep old names for existing tests

CodeChallengeEnv = CodeChallengeEnvironment
FibonacciEnv = FibonacciEnvironment
