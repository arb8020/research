"""Logging utilities for rollouts framework.

Tiger Style: Simple, bounded logging with timestamped results directories.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from rollouts._logging import setup_logging


def init_rollout_logging(
    experiment_name: str,
    results_base_dir: Path = Path("results"),
    log_level: str = "INFO",
    logger_levels: dict | None = None,
) -> Path:
    """Initialize logging for a rollout experiment.

    Tiger Style: Creates timestamped results directory and sets up dual logging:
    - Console: Clean colorized output with ANSI codes (stdout)
    - File: Detailed JSONL logs for debugging (error_log.jsonl)

    Args:
        experiment_name: Name of the experiment (e.g., "screenspot_eval")
        results_base_dir: Base directory for results (default: "results/")
        log_level: Default log level (default: "INFO")
        logger_levels: Optional dict of logger-specific levels
                      e.g. {"rollouts": "DEBUG", "httpx": "WARNING"}
                      User-provided levels override defaults

    Returns:
        Path to the timestamped results directory

    Example:
        >>> result_dir = init_rollout_logging("my_eval")
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Starting evaluation")  # Goes to stdout + file
        >>> # Save other outputs to result_dir
        >>> (result_dir / "results.json").write_text(json.dumps(results))
    """
    # Default logger levels for third-party libraries (suppress verbose HTTP client logging)
    # These are noisy at INFO level and should only appear at WARNING+
    default_logger_levels = {
        "httpx": "WARNING",
        "httpcore": "WARNING",
        "httpcore.http11": "WARNING",
        "httpcore.http2": "WARNING",
        "openai": "WARNING",
        "anthropic": "WARNING",
        "paramiko": "WARNING",  # Suppress SSH connection details ("Connected (version 2.0...)")
        "paramiko.transport": "WARNING",
    }

    # Merge user-provided levels with defaults (user overrides take precedence)
    merged_levels = {**default_logger_levels, **(logger_levels or {})}

    # Create timestamped result directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    result_dir_name = f"{experiment_name}_{timestamp}"
    result_dir = results_base_dir / result_dir_name
    result_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging with file handler
    log_file = result_dir / "error_log.jsonl"
    setup_logging(
        level=log_level,
        use_json=False,  # Human-readable console output
        use_color=True,  # Colorized output with ANSI codes
        logger_levels=merged_levels,
        log_file=str(log_file),
    )

    # Log initialization
    logger = logging.getLogger(__name__)
    logger.info(f"results directory: {result_dir}")
    logger.info(f"error log: {log_file}")

    return result_dir


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.

    Convenience wrapper around logging.getLogger.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
