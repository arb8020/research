"""Retry decorators with exponential backoff for network operations.

Tiger Style compliant: Use only at external boundaries (network I/O, file I/O).
Internal code should use assertions and let it crash.

This is a lightweight alternative to tenacity with the essential features:
- Composable retry/stop/wait strategies
- Retry on exceptions OR results
- Exponential backoff with optional jitter
- Before/after/on_retry callbacks
- Async support via trio
"""

import functools
import inspect
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, cast

# Type variable for the decorated function
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class RetryState:
    """State passed to callbacks during retry attempts."""

    attempt: int
    delay: float
    exception: Exception | None = None
    result: Any = None


# Pure helper functions (no control flow, just computation)
def _calculate_sleep_time(base_delay: float, jitter: bool, max_delay: float | None) -> float:
    """Calculate sleep time with optional jitter and cap. Pure function."""
    assert base_delay > 0, f"base_delay must be > 0, got {base_delay}"
    assert isinstance(jitter, bool), f"jitter must be bool, got {type(jitter)}"

    sleep_time = base_delay
    if jitter:
        # Add up to 100% jitter
        sleep_time = base_delay * (0.5 + random.random())
    if max_delay:
        sleep_time = min(sleep_time, max_delay)

    assert sleep_time > 0, f"sleep_time must be > 0, got {sleep_time}"
    return sleep_time


def _should_retry_on_result(result: Any, retry_predicate: Callable[[Any], bool] | None) -> bool:
    """Check if result should trigger retry. Pure function."""
    if retry_predicate is None:
        return False
    return retry_predicate(result)


def _log_retry_attempt(
    func_name: str,
    module_name: str,
    attempt: int,
    max_attempts: int,
    sleep_time: float,
    exception: Exception | None,
    has_result_predicate: bool,
) -> None:
    """Log retry attempt. Leaf function with no control flow."""
    assert isinstance(func_name, str), f"func_name must be str, got {type(func_name)}"
    assert isinstance(module_name, str), f"module_name must be str, got {type(module_name)}"
    assert attempt > 0, f"attempt must be > 0, got {attempt}"
    assert attempt <= max_attempts, (
        f"attempt must be <= max_attempts, got {attempt} > {max_attempts}"
    )
    assert sleep_time > 0, f"sleep_time must be > 0, got {sleep_time}"

    logger = logging.getLogger(module_name)
    if exception:
        logger.warning(
            f"{func_name}() attempt {attempt}/{max_attempts} failed: {exception}. "
            f"Retrying in {sleep_time:.2f}s..."
        )
    else:
        logger.warning(
            f"{func_name}() attempt {attempt}/{max_attempts} returned retriable result. "
            f"Retrying in {sleep_time:.2f}s..."
        )


def retry(
    max_attempts: int = 3,
    delay: float = 1,
    backoff: float = 2,
    exceptions: tuple[type[Exception], ...] = (Exception,),
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
        def wrapper(*args: Any, **kwargs: Any) -> Any:
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
                    logger = logging.getLogger(getattr(func, "__module__", "unknown"))
                    logger.warning(
                        f"{getattr(func, '__name__', '<function>')}() attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {current_delay}s..."
                    )

                    time.sleep(current_delay)
                    current_delay *= backoff

            # Tiger Style: Assert impossible states
            raise AssertionError("Retry loop exited without return or raise")

        return cast(F, wrapper)

    return decorator


def async_retry(
    max_attempts: int = 3,
    delay: float = 1,
    backoff: float = 2,
    max_delay: float | None = None,
    jitter: bool = False,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    retry_on_result: Callable[[Any], bool] | None = None,
    on_retry: Callable[[RetryState], None] | None = None,
) -> Callable[[F], F]:
    """Async retry decorator with exponential backoff using trio.

    Use this ONLY at external boundaries (network I/O, API calls).
    Internal code should use assertions and fail fast.

    Args:
        max_attempts: Maximum number of attempts (default: 3)
        delay: Initial delay in seconds (default: 1)
        backoff: Backoff multiplier (default: 2, gives 1s, 2s, 4s)
        max_delay: Maximum delay cap (default: None = no cap)
        jitter: Add randomness to delay (default: False)
        exceptions: Tuple of exceptions to catch (default: all Exception)
        retry_on_result: Optional predicate to retry based on result (e.g., lambda x: x is None)
        on_retry: Optional callback called before each retry with RetryState

    Example:
        @async_retry(
            max_attempts=5,
            delay=1,
            backoff=2,
            max_delay=60,
            jitter=True,
            exceptions=(openai.RateLimitError,),
            on_retry=lambda state: logger.info(f"Retry {state.attempt}")
        )
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
                exception_to_retry: Exception | None = None
                result: Any = None

                # Execute function
                try:
                    result = await func(*args, **kwargs)
                except exceptions as e:
                    exception_to_retry = e

                # Tiger Style: Centralize control flow (push ifs up)
                # Check if we should retry
                should_retry = exception_to_retry is not None or _should_retry_on_result(
                    result, retry_on_result
                )

                if not should_retry:
                    # Success - return result
                    return result

                # Prepare for retry
                attempt += 1

                if attempt >= max_attempts:
                    # Final attempt exhausted
                    if exception_to_retry:
                        raise exception_to_retry
                    # Result failed predicate but no more retries
                    return result

                # Calculate delay using pure helper
                sleep_time = _calculate_sleep_time(current_delay, jitter, max_delay)

                # Handle callback or logging
                if on_retry:
                    state = RetryState(
                        attempt=attempt,
                        delay=sleep_time,
                        exception=exception_to_retry,
                        result=result,
                    )
                    on_retry(state)
                else:
                    # Use leaf logging function
                    _log_retry_attempt(
                        func_name=getattr(func, "__name__", "<function>"),
                        module_name=getattr(func, "__module__", "unknown"),
                        attempt=attempt,
                        max_attempts=max_attempts,
                        sleep_time=sleep_time,
                        exception=exception_to_retry,
                        has_result_predicate=retry_on_result is not None,
                    )

                await trio.sleep(sleep_time)
                current_delay *= backoff

            # Tiger Style: Assert impossible states
            raise AssertionError("Retry loop exited without return or raise")

        return cast(F, wrapper)

    return decorator
