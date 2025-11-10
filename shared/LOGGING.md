# Shared Logging Module

**Purpose**: Error/debug logging for research monorepo projects (rollouts, bifrost, broker, slime).

**NOT for training metrics** - see `rollouts/training/metrics.py` for that.

## Design Philosophy

- **Tiger Style**: Bounded resources, explicit configuration, fail-fast
- **Sean Goedecke**: Boring, well-tested patterns (Python's logging module)
- **mCoding**: Modern Python logging with QueueHandler for async safety

## Quick Start

```python
from shared.logging_config import setup_logging
import logging

# Setup logging once at app start
setup_logging(
    level="INFO",
    log_file="logs/app.jsonl",
    use_queue_handler=True,  # Recommended for async code
)

# Use logging everywhere
logger = logging.getLogger(__name__)
logger.info("Application started")
logger.error("Something went wrong", extra={"user_id": 123})
```

## Features

### 1. Console + File Output
- **Console**: Human-readable by default, JSON optional
- **File**: Always JSONL (machine-readable)
- **Both**: Simultaneous output to console and file

### 2. Log Rotation (Tiger Style: Bounded!)
- **Default**: 100MB max file size, 5 backup files
- **Prevents**: Unbounded disk usage
- **Format**: `app.jsonl`, `app.jsonl.1`, `app.jsonl.2`, ...

### 3. QueueHandler for Async Safety (mCoding Pattern)
- **Purpose**: Prevents blocking in async code (trio/asyncio)
- **How**: Logs go to queue → background thread writes to files
- **Auto**: Python 3.12+ creates QueueListener automatically
- **Cleanup**: Registered with `atexit` for graceful shutdown

### 4. Per-Logger Levels
```python
setup_logging(
    level="INFO",
    logger_levels={
        "bifrost": "DEBUG",      # Verbose for my code
        "httpx": "WARNING",      # Quiet for HTTP client
        "paramiko": "ERROR",     # Very quiet for SSH
    }
)
```

## Configuration

### Basic
```python
setup_logging(level="INFO")
# Logs to console only, human-readable
```

### With File Output
```python
setup_logging(
    level="DEBUG",
    log_file="logs/app.jsonl",
)
# Logs to console (INFO+) and file (DEBUG+) in JSONL
```

### For Async Code (Recommended)
```python
setup_logging(
    level="INFO",
    log_file="logs/app.jsonl",
    use_queue_handler=True,  # Non-blocking logging!
)
```

### With Rich Console Output
```python
setup_logging(
    level="INFO",
    use_rich=True,  # Pretty colors and formatting
    rich_tracebacks=True,  # Beautiful exception formatting
)
```

### Custom Log Rotation
```python
setup_logging(
    log_file="logs/app.jsonl",
    max_log_bytes=10_000_000,  # 10MB max
    backup_count=3,  # Keep 3 old files
)
```

## Structured Logging

Add extra fields to log messages:

```python
logger.info(
    "Training step completed",
    extra={
        "step": 100,
        "loss": 0.5,
        "reward": 0.3,
    }
)

# JSONL output:
# {"message": "Training step completed", "step": 100, "loss": 0.5, "reward": 0.3, ...}
```

## Integration with Rollouts

```python
# rollouts/logging_utils.py wraps this nicely

from rollouts.logging_utils import init_rollout_logging

# Creates timestamped results directory with logging
result_dir = init_rollout_logging("my_experiment")

# Logs to:
# - Console (human-readable)
# - results/my_experiment_20250109-142030/error_log.jsonl
```

## Why Not Use This for Training Metrics?

| Aspect | Error Logs (this module) | Training Metrics (separate) |
|--------|-------------------------|----------------------------|
| **Purpose** | Debug issues | Track model performance |
| **Frequency** | Sporadic | Every N steps |
| **Volume** | Low | High (1000s of points) |
| **Structure** | Messages + context | Timeseries numeric data |
| **Tools** | grep, tail | pandas, matplotlib, W&B |
| **Failure mode** | Can crash app | MUST NOT crash training |

Training metrics should use `rollouts/training/metrics.py` (JSONL/CSV/W&B).

## Files

- `shared/logging_config.py`: Main configuration function
- `shared/json_formatter.py`: JSON formatter for structured logs

## References

- [mCoding Modern Python Logging](https://github.com/mCodingLLC/VideosSampleCode/tree/master/videos/135_modern_logging)
- [Python Logging Cookbook](https://docs.python.org/3/howto/logging-cookbook.html)
- [Tiger Style Safety](https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md)
