"""GEPA adapters for common use cases.

Each adapter provides:
- A frozen dataclass for config
- Pure functions: evaluate_* and make_*_reflective

Usage with run_gepa:
    from functools import partial

    config = SystemPromptConfig(endpoint=endpoint, user_template="{query}", score_fn=score)

    result = await run_gepa(
        seed_candidate={"system": "You are helpful."},
        dataset=my_data,
        evaluate_fn=partial(evaluate_system_prompt, config),
        make_reflective_fn=make_system_prompt_reflective,
        config=GEPAConfig(...),
        reflection_endpoint=endpoint,
    )
"""

# System prompt adapter (single prompt optimization)
from .system_prompt import (
    SystemPromptConfig,
    evaluate_system_prompt,
    make_system_prompt_reflective,
)

# System + user prompt adapter (both prompts optimized)
from .system_user_prompt import (
    SystemUserPromptConfig,
    evaluate_system_user_prompt,
    make_system_user_prompt_reflective,
)

# Terminal-bench adapter
from .terminal_bench import (
    TerminalBenchConfig,
    TerminalBenchTask,
    evaluate_terminal_bench,
    make_terminal_bench_reflective,
    run_tests_and_score,
)

__all__ = [
    # System prompt
    "SystemPromptConfig",
    "evaluate_system_prompt",
    "make_system_prompt_reflective",
    # System + user prompt
    "SystemUserPromptConfig",
    "evaluate_system_user_prompt",
    "make_system_user_prompt_reflective",
    # Terminal-bench
    "TerminalBenchConfig",
    "TerminalBenchTask",
    "evaluate_terminal_bench",
    "make_terminal_bench_reflective",
    "run_tests_and_score",
]
