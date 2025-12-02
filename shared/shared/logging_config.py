"""Shared logging configuration for research monorepo.

Tiger Style: Explicit configuration, bounded resources, fail-fast.
Sean Goedecke: Boring, well-tested patterns (Python's logging module).
"""
import logging.config
import logging.handlers
import os
import sys
from queue import Queue
from typing import Any, Dict, Optional


def setup_logging(
    level: Optional[str] = None,
    use_json: Optional[bool] = None,
    use_rich: Optional[bool] = None,
    use_color: Optional[bool] = None,
    logger_levels: Optional[Dict[str, str]] = None,
    log_file: Optional[str] = None,
    rich_tracebacks: bool = False,
    use_queue_handler: bool = True,
    max_log_bytes: int = 100_000_000,
    backup_count: int = 5,
) -> None:
    """Setup standardized logging configuration using dict config.

    Tiger Style: Bounded log files, explicit parameters, assertions.
    Sean Goedecke: Uses Python's standard logging.config (boring, reliable).

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
                      e.g. {"bifrost": "DEBUG", "broker": "WARNING", "paramiko": "ERROR"}
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
        >>> from shared.logging_config import setup_logging
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

    # JSON mode and color mode override rich mode
    if use_json or use_color:
        use_rich = False

    formatters = {
        "standard": {
            "format": "[%(asctime)s] %(levelname)s: %(message)s",
            "datefmt": "%H:%M:%S"
        },
        "minimal": {
            "format": "%(message)s"
        },
        "color": {
            "()": "shared.color_formatter.ColorFormatter",
            "show_timestamp": True
        },
        "json": {
            "()": "shared.json_formatter.JSONFormatter",
            "fmt_keys": {
                "level": "levelname",
                "logger": "name",
                "module": "module",
                "function": "funcName",
                "line": "lineno"
            }
        }
    }

    # Choose handler and formatter based on mode
    if use_rich:
        handlers = {
            "console": {
                "class": "rich.logging.RichHandler",
                "level": "DEBUG",  # Let loggers control their own levels
                "formatter": "minimal",
                "rich_tracebacks": rich_tracebacks,
                "show_time": False,
                "show_path": False
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
                "stream": "ext://sys.stdout"
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
    # Python 3.11 requires manual Queue and QueueListener setup
    # The listener runs in a background thread, prevents blocking in async code
    if use_queue_handler:
        # Python 3.12+ supports 'handlers' parameter in dictConfig
        # Python 3.11 requires manual setup
        if sys.version_info >= (3, 12):
            handlers["queue_handler"] = {
                "class": "logging.handlers.QueueHandler",
                "handlers": handler_list.copy(),  # Wrap our actual handlers
                "respect_handler_level": True,  # Each handler keeps its own level
            }
        else:
            # Python 3.11: Create queue manually
            # We'll configure QueueHandler with just the queue, then create listener later
            handlers["queue_handler"] = {
                "class": "logging.handlers.QueueHandler",
                "queue": Queue(-1),  # Unbounded queue (will be created by dictConfig)
            }
        handler_list = ["queue_handler"]  # Route all logs through queue

    config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "loggers": {},
        "root": {
            "level": level,
            "handlers": handler_list
        }
    }

    # Add specific logger configurations
    loggers_config = config["loggers"]
    assert isinstance(loggers_config, dict), "loggers must be dict"

    for logger_name, logger_level in logger_levels.items():
        loggers_config[logger_name] = {
            "level": logger_level,
            "handlers": handler_list,
            "propagate": False  # Don't propagate to root to avoid duplicate logs
        }

    logging.config.dictConfig(config)

    # mCoding pattern: Start QueueListener and register cleanup
    if use_queue_handler:
        # Get the queue_handler - getHandlerByName is Python 3.12+
        if sys.version_info >= (3, 12):
            queue_handler = logging.getHandlerByName("queue_handler")
        else:
            # Python 3.11: Get handler from root logger
            queue_handler = None
            for handler in logging.root.handlers:
                if isinstance(handler, logging.handlers.QueueHandler):
                    queue_handler = handler
                    break

        if queue_handler is not None:
            if sys.version_info >= (3, 12):
                # Python 3.12+: QueueListener is created automatically, just start it
                if hasattr(queue_handler, 'listener'):
                    queue_handler.listener.start()
                    # Register cleanup on exit (mCoding pattern)
                    import atexit
                    atexit.register(queue_handler.listener.stop)
            else:
                # Python 3.11: Manually create and start QueueListener
                # Recreate the actual handlers that logs should be routed to
                actual_handlers = []
                handlers_dict = config.get("handlers", {})

                # Get the original handler list (before queue_handler was added)
                # We need to recreate these handlers from the config
                for handler_name in ["console", "file"]:
                    if handler_name in handlers_dict and handler_name != "queue_handler":
                        handler_config = handlers_dict[handler_name]
                        handler_class_path = handler_config.get("class", "")

                        # Create handler instance based on config
                        if "RichHandler" in handler_class_path:
                            from rich.logging import RichHandler
                            handler = RichHandler(
                                level=handler_config.get("level", "DEBUG"),
                                rich_tracebacks=handler_config.get("rich_tracebacks", False),
                                show_time=handler_config.get("show_time", False),
                                show_path=handler_config.get("show_path", False),
                            )
                            # Set formatter
                            formatter_name = handler_config.get("formatter", "minimal")
                            if formatter_name == "minimal":
                                handler.setFormatter(logging.Formatter("%(message)s"))
                        elif "StreamHandler" in handler_class_path:
                            handler = logging.StreamHandler(sys.stdout)
                            handler.setLevel(handler_config.get("level", "DEBUG"))
                            # Set formatter
                            formatter_name = handler_config.get("formatter", "standard")
                            formatter_config = config["formatters"].get(formatter_name, {})
                            if "color" in formatter_name:
                                from shared.color_formatter import ColorFormatter
                                handler.setFormatter(ColorFormatter(
                                    show_timestamp=formatter_config.get("show_timestamp", True)
                                ))
                            elif "json" in formatter_name:
                                from shared.json_formatter import JSONFormatter
                                handler.setFormatter(JSONFormatter(
                                    fmt_keys=formatter_config.get("fmt_keys", {})
                                ))
                            else:
                                handler.setFormatter(logging.Formatter(
                                    fmt=formatter_config.get("format", "[%(asctime)s] %(levelname)s: %(message)s"),
                                    datefmt=formatter_config.get("datefmt", "%H:%M:%S")
                                ))
                        elif "RotatingFileHandler" in handler_class_path:
                            handler = logging.handlers.RotatingFileHandler(
                                filename=handler_config.get("filename"),
                                mode=handler_config.get("mode", "a"),
                                maxBytes=handler_config.get("maxBytes", max_log_bytes),
                                backupCount=handler_config.get("backupCount", backup_count),
                            )
                            handler.setLevel(handler_config.get("level", "DEBUG"))
                            # File handler always uses JSON formatter
                            from shared.json_formatter import JSONFormatter
                            formatter_config = config["formatters"].get("json", {})
                            handler.setFormatter(JSONFormatter(
                                fmt_keys=formatter_config.get("fmt_keys", {})
                            ))
                        else:
                            continue

                        actual_handlers.append(handler)

                # Create and start QueueListener with the actual handlers
                if actual_handlers and hasattr(queue_handler, 'queue'):
                    listener = logging.handlers.QueueListener(
                        queue_handler.queue,
                        *actual_handlers,
                        respect_handler_level=True
                    )
                    listener.start()

                    # Store listener reference for cleanup
                    queue_handler.listener = listener

                    # Register cleanup on exit
                    import atexit
                    atexit.register(listener.stop)