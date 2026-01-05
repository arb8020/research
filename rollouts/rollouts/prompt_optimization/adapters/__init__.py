"""GEPA adapters for common use cases.

Each adapter provides:
- A frozen dataclass for config
- Pure functions: evaluate_* and make_*_reflective
- Optional wrapper class for backwards compatibility
"""

# System prompt adapter (single prompt optimization)
from .system_prompt import (
    SinglePromptAdapter,  # backwards compat alias
    SinglePromptConfig,  # backwards compat alias
    # Wrapper class (backwards compat)
    SystemPromptAdapter,
    # Config
    SystemPromptConfig,
    # Pure functions
    evaluate_system_prompt,
    make_system_prompt_reflective,
)

# System + user prompt adapter
from .system_user_prompt import SystemUserPromptAdapter

# Terminal-bench adapter
from .terminal_bench import (
    # Wrapper class (backwards compat)
    TerminalBenchAdapter,
    # Config
    TerminalBenchConfig,
    TerminalBenchTask,
    # Pure functions
    evaluate_terminal_bench,
    make_terminal_bench_reflective,
    # Scoring helper
    run_tests_and_score,
)

__all__ = [
    # System prompt
    "SystemPromptConfig",
    "SinglePromptConfig",
    "evaluate_system_prompt",
    "make_system_prompt_reflective",
    "SystemPromptAdapter",
    "SinglePromptAdapter",
    # System + user prompt
    "SystemUserPromptAdapter",
    # Terminal-bench
    "TerminalBenchConfig",
    "TerminalBenchTask",
    "evaluate_terminal_bench",
    "make_terminal_bench_reflective",
    "run_tests_and_score",
    "TerminalBenchAdapter",
]
