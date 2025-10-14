import logging.config
import os
from typing import Any, Dict, Optional


def setup_logging(level: Optional[str] = None, use_json: Optional[bool] = None, use_rich: Optional[bool] = None,
                  logger_levels: Optional[Dict[str, str]] = None, log_file: Optional[str] = None,
                  rich_tracebacks: bool = False):
    """Setup standardized logging configuration using dict config.

    Args:
        level: Default log level for root logger
        use_json: Whether to use JSON formatter for console (default: False for human-readable)
        use_rich: Whether to use RichHandler for console output (default: False).
                 If True, produces clean CLI output with colors and formatting.
                 Overridden to False if use_json=True.
        logger_levels: Dict mapping logger names to specific log levels
                      e.g. {"bifrost": "DEBUG", "broker": "WARNING", "paramiko": "ERROR"}
        log_file: Optional log file path. If provided, logs in JSONL format to file
                 while keeping human-readable console output
        rich_tracebacks: Whether to enable rich tracebacks (only applies when use_rich=True)
    """
    level = level or os.getenv("LOG_LEVEL", "INFO")
    use_json = use_json if use_json is not None else os.getenv("LOG_JSON", "").lower() == "true"
    use_rich = use_rich if use_rich is not None else False
    logger_levels = logger_levels or {}

    # JSON mode overrides rich mode
    if use_json:
        use_rich = False

    formatters = {
        "standard": {
            "format": "[%(asctime)s] %(levelname)s: %(message)s",
            "datefmt": "%H:%M:%S"
        },
        "minimal": {
            "format": "%(message)s"
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
        handlers = {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",  # Let loggers control their own levels
                "formatter": "json" if use_json else "standard",
                "stream": "ext://sys.stdout"
            }
        }

    # Add file handler for JSONL logging if log_file specified
    handler_list = ["console"]
    if log_file:
        handlers["file"] = {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "json",  # Always use JSON for file output
            "filename": log_file,
            "mode": "a"
        }
        handler_list.append("file")

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


def parse_logger_levels(log_level_specs: Optional[list] = None) -> Dict[str, str]:
    """Parse logger level specifications from command line format.
    
    Args:
        log_level_specs: List of strings in format "logger_name:level"
                        e.g. ["bifrost:DEBUG", "broker:WARNING", "paramiko:ERROR"]
    
    Returns:
        Dict mapping logger names to log levels
    """
    if not log_level_specs:
        return {}
    
    result = {}
    for spec in log_level_specs:
        if ":" not in spec:
            raise ValueError(f"Invalid log level spec: {spec}. Expected format: logger_name:level")
        
        logger_name, level = spec.split(":", 1)
        level = level.upper()
        
        # Validate log level
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in valid_levels:
            raise ValueError(f"Invalid log level: {level}. Must be one of {valid_levels}")
        
        result[logger_name] = level
    
    return result