"""Retry decorator with exponential backoff for network operations.

Tiger Style compliant: Use only at external boundaries (network I/O, file I/O).
Internal code should use assertions and let it crash.
"""

import logging
import time
from typing import Callable, Tuple, Type, TypeVar, cast
import functools

# Type variable for the decorated function
F = TypeVar('F', bound=Callable)


def retry(
    max_attempts: int = 3,
    delay: float = 1,
    backoff: float = 2,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable[[F], F]:
    """Retry decorator with exponential backoff.

    Use this ONLY at external boundaries (network I/O, API calls, file transfers).
    Internal code should use assertions and fail fast.

    Args:
        max_attempts: Maximum number of attempts (default: 3)
        delay: Initial delay in seconds (default: 1)
        backoff: Backoff multiplier (default: 2, gives 1s, 2s, 4s)
        exceptions: Tuple of exceptions to catch (default: all Exception)

    Example:
        @retry(max_attempts=3, delay=1, backoff=2, exceptions=(requests.RequestException,))
        def make_api_call():
            return requests.get("https://api.example.com")
    """
    # Tiger Style: Assert all inputs
    assert max_attempts >= 1, f"max_attempts must be >= 1, got {max_attempts}"
    assert delay > 0, f"delay must be > 0, got {delay}"
    assert backoff >= 1, f"backoff must be >= 1, got {backoff}"
    assert isinstance(exceptions, tuple), f"exceptions must be tuple, got {type(exceptions)}"
    assert len(exceptions) > 0, "exceptions tuple cannot be empty"

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay

            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        # Let the final exception propagate
                        raise

                    # Log the retry for debugging
                    logger = logging.getLogger(func.__module__)
                    logger.warning(
                        f"{func.__name__}() attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {current_delay}s..."
                    )

                    time.sleep(current_delay)
                    current_delay *= backoff

            # Tiger Style: Assert impossible states
            assert False, "Retry loop exited without return or raise"

        return cast(F, wrapper)
    return decorator
