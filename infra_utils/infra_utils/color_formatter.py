"""ANSI color formatter for clean, colorized console output.

Tiger Style: Simple, explicit ANSI codes. No external dependencies.
"""

import logging


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"

    # Level colors
    DEBUG = "\033[36m"  # Cyan
    INFO = "\033[37m"  # White (default)
    WARNING = "\033[33m"  # Yellow
    ERROR = "\033[31m"  # Red
    CRITICAL = "\033[31;1m"  # Bold Red

    # Component colors
    TIMESTAMP = "\033[90m"  # Dark gray


class ColorFormatter(logging.Formatter):
    """Formatter with ANSI color codes based on log level.

    Format: [HH:MM:SS] message
    - Timestamp in dark gray
    - Message colored by level (INFO=white, WARNING=yellow, ERROR=red)
    - No explicit level prefix (color indicates level)

    Tiger Style: Explicit color mapping, bounded output.
    """

    LEVEL_COLORS = {
        logging.DEBUG: Colors.DEBUG,
        logging.INFO: Colors.INFO,
        logging.WARNING: Colors.WARNING,
        logging.ERROR: Colors.ERROR,
        logging.CRITICAL: Colors.CRITICAL,
    }

    def __init__(self, show_timestamp: bool = True) -> None:
        """Initialize formatter.

        Args:
            show_timestamp: Whether to show timestamp prefix (default: True)
        """
        super().__init__()
        self.show_timestamp = show_timestamp

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with ANSI colors.

        Args:
            record: Log record to format

        Returns:
            Formatted string with ANSI color codes
        """
        # Get color for this level
        level_color = self.LEVEL_COLORS.get(record.levelno, Colors.INFO)

        # Format timestamp
        timestamp = self.formatTime(record, datefmt="%H:%M:%S")

        # Format message
        message = record.getMessage()

        # Add exception info if present
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)

        # Build output with colors
        if self.show_timestamp:
            return (
                f"{Colors.TIMESTAMP}[{timestamp}]{Colors.RESET} "
                f"{level_color}{message}{Colors.RESET}"
            )
        else:
            return f"{level_color}{message}{Colors.RESET}"
