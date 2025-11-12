"""Retry decorator with exponential backoff for network operations.

Tiger Style compliant: Use only at external boundaries (network I/O, file I/O).
Internal code should use assertions and let it crash.
"""

import logging
import time
from typing import Callable, Tuple, Type, TypeVar, cast, Any
import functools
import inspect

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
                    logger = logging.getLogger(getattr(func, '__module__', 'unknown'))
                    logger.warning(
                        f"{getattr(func, '__name__', '<function>')}() attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {current_delay}s..."
                    )

                    time.sleep(current_delay)
                    current_delay *= backoff

            # Tiger Style: Assert impossible states
            assert False, "Retry loop exited without return or raise"

        return cast(F, wrapper)
    return decorator


def async_retry(
    max_attempts: int = 3,
    delay: float = 1,
    backoff: float = 2,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable[[F], F]:
    """Async retry decorator with exponential backoff using trio.

    Use this ONLY at external boundaries (network I/O, API calls).
    Internal code should use assertions and fail fast.

    Args:
        max_attempts: Maximum number of attempts (default: 3)
        delay: Initial delay in seconds (default: 1)
        backoff: Backoff multiplier (default: 2, gives 1s, 2s, 4s)
        exceptions: Tuple of exceptions to catch (default: all Exception)

    Example:
        @async_retry(max_attempts=3, delay=1, backoff=2, exceptions=(openai.RateLimitError,))
        async def make_api_call():
            return await client.chat.completions.create(...)
    """
    # Tiger Style: Assert all inputs
    assert max_attempts >= 1, f"max_attempts must be >= 1, got {max_attempts}"
    assert delay > 0, f"delay must be > 0, got {delay}"
    assert backoff >= 1, f"backoff must be >= 1, got {backoff}"
    assert isinstance(exceptions, tuple), f"exceptions must be tuple, got {type(exceptions)}"
    assert len(exceptions) > 0, "exceptions tuple cannot be empty"

    def decorator(func: F) -> F:
        # Verify function is async
        assert inspect.iscoroutinefunction(func), f"{func.__name__} must be async for async_retry"

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            import trio  # Import here to avoid hard dependency

            attempt = 0
            current_delay = delay

            while attempt < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        # Let the final exception propagate
                        raise

                    # Log the retry for debugging
                    logger = logging.getLogger(getattr(func, '__module__', 'unknown'))
                    logger.warning(
                        f"{getattr(func, '__name__', '<function>')}() attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {current_delay}s..."
                    )

                    await trio.sleep(current_delay)
                    current_delay *= backoff

            # Tiger Style: Assert impossible states
            assert False, "Retry loop exited without return or raise"

        return cast(F, wrapper)
    return decorator
