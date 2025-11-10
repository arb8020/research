"""Logging utilities for rollouts framework.

Tiger Style: Simple, bounded logging with timestamped results directories.

Note: This assumes 'shared' package is available (from /Users/chiraagbalu/research/shared/).
For standalone usage without shared, use standard logging.getLogger(__name__) directly.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from shared.logging_config import setup_logging
except ImportError:
    # Fallback if shared is not available
    def setup_logging(
        level: str = "INFO",
        use_json: bool = False,
        use_rich: bool = False,
        logger_levels: Optional[dict] = None,
        log_file: Optional[str] = None,
    ) -> None:
        """Simple logging setup without shared dependency."""
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=getattr(logging, level.upper()), format=format_str)
        if logger_levels:
            for logger_name, logger_level in logger_levels.items():
                logging.getLogger(logger_name).setLevel(getattr(logging, logger_level.upper()))
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(format_str))
            logging.getLogger().addHandler(file_handler)


def init_rollout_logging(
    experiment_name: str,
    results_base_dir: Path = Path("results"),
    log_level: str = "INFO",
    logger_levels: Optional[dict] = None,
) -> Path:
    """Initialize logging for a rollout experiment.

    Tiger Style: Creates timestamped results directory and sets up dual logging:
    - Console: Clean human-readable output (stdout)
    - File: Detailed JSONL logs for debugging (error_log.jsonl)

    Args:
        experiment_name: Name of the experiment (e.g., "screenspot_eval")
        results_base_dir: Base directory for results (default: "results/")
        log_level: Default log level (default: "INFO")
        logger_levels: Optional dict of logger-specific levels
                      e.g. {"rollouts": "DEBUG", "httpx": "WARNING"}

    Returns:
        Path to the timestamped results directory

    Example:
        >>> result_dir = init_rollout_logging("my_eval")
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Starting evaluation")  # Goes to stdout + file
        >>> # Save other outputs to result_dir
        >>> (result_dir / "results.json").write_text(json.dumps(results))
    """
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
        use_rich=False,  # Keep it simple for now
        logger_levels=logger_levels or {},
        log_file=str(log_file),
    )

    # Log initialization
    logger = logging.getLogger(__name__)
    logger.info(f"📂 Results directory: {result_dir}")
    logger.info(f"📝 Error log: {log_file}")

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
