"""TTT (Test-Time Training) environments.

Single-turn code generation environments for TTT-style iterative improvement.
"""

from .code_challenge import (
    # Backward compat
    CodeChallengeEnv,
    CodeChallengeEnvironment,
    FibonacciEnv,
    FibonacciEnvironment,
    TestCase,
)

__all__ = [
    "CodeChallengeEnvironment",
    "FibonacciEnvironment",
    "TestCase",
    "CodeChallengeEnv",
    "FibonacciEnv",
]
