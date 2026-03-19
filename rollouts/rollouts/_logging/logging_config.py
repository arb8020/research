"""Logging configuration for rollouts.

Tiger Style: Explicit configuration, bounded resources, fail-fast.

Two main functions:
- setup_logging(): General app-level logging (console + optional file)
- setup_eval_logging(): Eval-specific logging (events.jsonl)
"""

import atexit
import logging
import logging.config
import logging.handlers
import os
import queue as queue_mod
from pathlib import Path
from typing import Any

from .json_formatter import JSONFormatter


def setup_logging(
    level: str | None = None,
    use_json: bool | None = None,
    use_rich: bool | None = None,
    use_color: bool | None = None,
    logger_levels: dict[str, str] | None = None,
    log_file: str | None = None,
    rich_tracebacks: bool = False,
    use_queue_handler: bool = True,
    max_log_bytes: int = 100_000_000,
    backup_count: int = 5,
) -> None:
    """Setup standardized logging configuration using dict config.

    Tiger Style: Bounded log files, explicit parameters, assertions.

    Args:
        level: Default log level for root logger (default: INFO or LOG_LEVEL env var)
        use_json: Whether to use JSON formatter for console (default: False for human-readable)
        use_rich: Whether to use RichHandler for console output (default: False).
                 If True, produces clean CLI output with colors and formatting.
                 Overridden to False if use_json=True or use_color=True.
        use_color: Whether to use ANSI color formatter for console (default: False).
                  If True, produces colorized output with minimal formatting.
                  Format: [HH:MM:SS] message (color indicates level).
                  Overrides use_rich if both are True.
        logger_levels: Dict mapping logger names to specific log levels
                      e.g. {"httpx": "WARNING", "paramiko": "ERROR"}
        log_file: Optional log file path. If provided, logs in JSONL format to file
                 with automatic rotation when file reaches max_log_bytes
        rich_tracebacks: Whether to enable rich tracebacks (only applies when use_rich=True)
        use_queue_handler: Whether to use QueueHandler for async-safe logging (default: True).
                          Recommended for async code (trio/asyncio) to prevent blocking.
        max_log_bytes: Maximum bytes per log file before rotation (default: 100MB).
                       Tiger Style: All files must be bounded!
        backup_count: Number of rotated log files to keep (default: 5)

    Returns:
        None. Configures Python's global logging state.

    Example:
        >>> from .._logging import setup_logging
        >>> setup_logging(level="DEBUG", log_file="logs/app.jsonl")
        >>> import logging
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Application started")
    """
    # Tiger Style: Assert preconditions
    assert max_log_bytes > 0, f"max_log_bytes must be > 0, got {max_log_bytes}"
    assert backup_count >= 0, f"backup_count must be >= 0, got {backup_count}"
    level = level or os.getenv("LOG_LEVEL", "INFO")
    use_json = use_json if use_json is not None else os.getenv("LOG_JSON", "").lower() == "true"
    use_rich = use_rich if use_rich is not None else False
    use_color = use_color if use_color is not None else False
    logger_levels = logger_levels or {}
    package_name = __package__ or "rollouts._logging"

    # JSON mode and color mode override rich mode
    if use_json or use_color:
        use_rich = False

    formatters: dict[str, Any] = {
        "standard": {
            "format": "[%(asctime)s] %(levelname)s: %(message)s",
            "datefmt": "%H:%M:%S",
        },
        "minimal": {"format": "%(message)s"},
        "color": {
            "()": f"{package_name.rsplit('.', 1)[0]}._logging.color_formatter.ColorFormatter",
            "show_timestamp": True,
        },
        "json": {
            "()": f"{package_name.rsplit('.', 1)[0]}._logging.json_formatter.JSONFormatter",
            "fmt_keys": {
                "level": "levelname",
                "logger": "name",
                "module": "module",
                "function": "funcName",
                "line": "lineno",
            },
        },
    }

    # Choose handler and formatter based on mode
    handlers: dict[str, Any]
    if use_rich:
        handlers = {
            "console": {
                "class": "rich.logging.RichHandler",
                "level": "DEBUG",  # Let loggers control their own levels
                "formatter": "minimal",
                "rich_tracebacks": rich_tracebacks,
                "show_time": False,
                "show_path": False,
            }
        }
    else:
        # Determine console formatter
        console_formatter = "standard"  # Default
        if use_json:
            console_formatter = "json"
        elif use_color:
            console_formatter = "color"

        handlers = {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",  # Let loggers control their own levels
                "formatter": console_formatter,
                "stream": "ext://sys.stderr",
            }
        }

    # Add file handler for JSONL logging if log_file specified
    # Tiger Style: Bounded! Use RotatingFileHandler to prevent unbounded growth
    handler_list = ["console"]
    if log_file:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "json",  # Always use JSON for file output
            "filename": log_file,
            "mode": "a",
            "maxBytes": max_log_bytes,  # Tiger: Bounded!
            "backupCount": backup_count,  # Keep N rotated files
        }
        handler_list.append("file")

    # mCoding pattern: Use QueueHandler for async-safe logging
    # Python 3.12+ QueueHandler in dictConfig automatically creates QueueListener!
    # The listener runs in a background thread, prevents blocking in async code
    if use_queue_handler:
        handlers["queue_handler"] = {
            "class": "logging.handlers.QueueHandler",
            "handlers": handler_list.copy(),  # Wrap our actual handlers
            "respect_handler_level": True,  # Each handler keeps its own level
        }
        handler_list = ["queue_handler"]  # Route all logs through queue

    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "loggers": {},
        "root": {"level": level, "handlers": handler_list},
    }

    # Add specific logger configurations
    loggers_config = config["loggers"]
    assert isinstance(loggers_config, dict), "loggers must be dict"

    for logger_name, logger_level in logger_levels.items():
        loggers_config[logger_name] = {
            "level": logger_level,
            "handlers": handler_list,
            "propagate": False,  # Don't propagate to root to avoid duplicate logs
        }

    logging.config.dictConfig(config)

    # mCoding pattern: Start QueueListener and register cleanup
    # Python 3.12+ creates the listener automatically, we just need to start it
    if use_queue_handler:
        get_handler_by_name = getattr(logging, "getHandlerByName", None)
        queue_handler = get_handler_by_name("queue_handler") if callable(get_handler_by_name) else None
        listener = getattr(queue_handler, "listener", None)
        if listener is not None:
            listener.start()
            # Register cleanup on exit (mCoding pattern)
            atexit.register(listener.stop)


# ── Eval-specific logging ──────────────────────────────────────────────────────


class EvalLoggingContext:
    """Manages eval-specific logging handlers for the duration of an eval run.

    Returned by setup_eval_logging(). Call teardown() when the eval is done
    to remove handlers and close file handles.

    Usage:
        ctx = setup_eval_logging(output_dir)
        try:
            # Run evaluation...
            _event_logger.info("sample_start", extra={"sample_id": "001"})
        finally:
            ctx.teardown()
    """

    def __init__(
        self,
        logger: logging.Logger,
        queue_handler: logging.handlers.QueueHandler,
        listener: logging.handlers.QueueListener,
    ) -> None:
        self.logger = logger
        self.queue_handler = queue_handler
        self.listener = listener

    def teardown(self) -> None:
        """Stop listener, remove handler, close files."""
        self.listener.stop()
        self.logger.removeHandler(self.queue_handler)


def setup_eval_logging(
    output_dir: Path,
    logger_name: str = "rollouts.eval.events",
    max_log_bytes: int = 100_000_000,
    backup_count: int = 5,
) -> EvalLoggingContext:
    """Configure eval-specific logging: rotating events.jsonl only.

    Adds handlers to the named logger (not root) for the duration of an
    eval run. Returns an EvalLoggingContext — call .teardown() when done.

    The events stream is the only canonical operational log. Per-sample JSONL
    logs were removed to avoid duplicating the same attempt across multiple
    overlapping artifact formats.

    This implements the "wide events" pattern from logging_sucks.md:
    - One comprehensive event per significant action
    - All context in extra={} fields
    - Queryable via SQL, not grep

    Args:
        output_dir: Directory for output files
        logger_name: Logger name (default: rollouts.eval.events)
        max_log_bytes: Max bytes before rotation (default: 100MB)
        backup_count: Number of rotated files to keep

    Returns:
        EvalLoggingContext for cleanup

    Example:
        ctx = setup_eval_logging(Path("results/my_eval"))
        logger = logging.getLogger("rollouts.eval.events")

        logger.info("sample_start", extra={"sample_id": "001", "name": "test"})
        logger.info("llm_call", extra={
            "sample_id": "001",
            "duration_ms": 1234,
            "tokens_in": 100,
            "tokens_out": 500,
        })
        logger.info("sample_end", extra={"sample_id": "001", "reward": 0.85})

        ctx.teardown()
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    formatter = JSONFormatter()

    # events.jsonl is the canonical operational event stream.
    events_handler = logging.handlers.RotatingFileHandler(
        str(output_dir / "events.jsonl"),
        mode="a",
        maxBytes=max_log_bytes,
        backupCount=backup_count,
    )
    events_handler.setFormatter(formatter)
    events_handler.setLevel(logging.INFO)

    # Wrap the file writer in QueueHandler for non-blocking writes.
    # QueueHandler puts records onto a queue; QueueListener consumes
    # from that queue and dispatches to the real handler in a background thread.
    q: queue_mod.Queue[logging.LogRecord] = queue_mod.Queue()
    queue_handler = logging.handlers.QueueHandler(q)
    listener = logging.handlers.QueueListener(q, events_handler, respect_handler_level=True)
    listener.start()

    logger = logging.getLogger(logger_name)
    logger.addHandler(queue_handler)
    logger.setLevel(logging.DEBUG)  # Let handlers decide what to filter
    logger.propagate = False  # Events go to files only, not root

    return EvalLoggingContext(
        logger=logger,
        queue_handler=queue_handler,
        listener=listener,
    )
