"""Internal logging utilities for rollouts.

Provides standardized logging configuration with:
- Color formatting for console output
- JSON formatting for structured logs
- Async-safe queue handlers
- File rotation with bounded sizes
- Eval-specific logging with one canonical events stream

Two main patterns:
1. setup_logging() - General app-level logging (console + optional file)
2. setup_eval_logging() - Eval-specific logging (events.jsonl)

The eval logging follows the "wide events" pattern:
- One comprehensive event per significant action (not many small logs)
- All context in extra={} fields (sample_id, duration_ms, tokens, etc.)
- Queryable via SQL, not grep
"""

from .color_formatter import ColorFormatter, Colors
from .json_formatter import JSONFormatter
from .logging_config import EvalLoggingContext, setup_eval_logging, setup_logging

__all__ = [
    "setup_logging",
    "setup_eval_logging",
    "EvalLoggingContext",
    "ColorFormatter",
    "Colors",
    "JSONFormatter",
]
