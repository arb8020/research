"""JSON formatter for structured logging.

Tiger Style: Explicit field mapping, no magic.
"""

import datetime as dt
import json
import logging
from typing import Any

# Built-in log record attributes that shouldn't be included in extras
# Tiger Style: Use frozenset (immutable) for module-level constants
LOG_RECORD_BUILTIN_ATTRS = frozenset({
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "getMessage",
})


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(self, *, fmt_keys: dict[str, str] | None = None) -> None:
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    def format(self, record: logging.LogRecord) -> str:
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord) -> dict[str, Any]:
        always_fields: dict[str, Any] = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(record.created, tz=dt.timezone.utc).isoformat(),
            "logger": record.name,
            "level": record.levelname,
        }

        if record.exc_info is not None:
            always_fields["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info is not None:
            always_fields["stack_info"] = self.formatStack(record.stack_info)

        # Map custom keys from config
        message: dict[str, Any] = {}
        for key, val in self.fmt_keys.items():
            # Try to get from always_fields first, then from record
            msg_val = always_fields.pop(val, None)
            if msg_val is not None:
                message[key] = msg_val
            else:
                message[key] = getattr(record, val)

        # Add remaining always_fields that weren't mapped
        message.update(always_fields)

        # Add any extra attributes not in built-in attrs
        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN_ATTRS:
                message[key] = val

        return message
