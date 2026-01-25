# Python Logging (mCoding)

`logging` is the standard even though it's flawed.

## Why logging matters
- Multi-destination output (stdout + file, maybe emails for errors)
- `dictConfig` makes configuration explicit

## Core components
- **Filters**: drop by level or transform records
- **Formatters**: record object → string
- **Handlers**: where logs go (stdout, file, email, service)
- **Loggers**: what you use in code

## Log propagation
Logs propagate up the logger tree: `logger.a.x` → `logger.a` → root logger.
- Dropped by handler: still propagates
- Dropped by logger: stops completely

## Best practices

### Keep it simple
- Put all handlers/filters on root logger
- Leave propagation on (default)
- Third-party libs get logged/formatted same way as your code

### Don't use root logger in your code
```python
# Bad
logging.info("message")

# Good
logger = logging.getLogger("my_app")
logger.info("message")
```

### Logger granularity
- One logger per major subcomponent
- Don't need one per file (they're globals, lives for program lifetime)

## dictConfig

```python
from logging.config import dictConfig
```

Required fields:
- `version: 1` (only valid value, for future compatibility)
- `disable_existing_loggers: false` (so third-party libs still log)

### External references
Use `ext://` prefix for external variables:
```json
"stream": "ext://sys.stdout"
```

### Custom class args
Use `()` instead of `class` when you need custom constructor args:
```json
{
  "()": "my_module.MyFormatter",
  "fmt_keys": {"level": "levelname"}
}
```
With `class`, keys are hardcoded to built-in's interface.

## Config storage
- Store in JSON or YAML, load at startup
- JSON preferred (built-in parser, less error-prone)
- YAML needs `pyyaml` dependency

## File logging
- `RotatingFileHandler`: size limit + backup count
- Pick a few megabytes for real usage

## JSON logs
- No built-in JSON formatter - write your own
- Subclass `logging.Formatter`, override `format(record) -> str`
- File extension: `.jsonl` (JSON Lines - each line valid JSON)
- Include timestamp in ISO-8601 UTC format
- Use parent methods to extract exception info

### Extras
```python
logger.info("message", extra={"user_id": 123})
```
Formatter pulls `extra` attributes onto JSON output.

## Custom filters
- Subclass `logging.Filter`, override `filter(record) -> bool`
- Can alter/censor records, not just drop them

## Non-blocking logging

Logging is sync/blocking by default. Use `QueueHandler` to log off main thread.

```python
# In config
{
  "class": "logging.handlers.QueueHandler",
  "handlers": ["stderr", "file"],
  "respect_handler_level": true  # Default is false!
}
```

### Thread management
```python
import atexit

queue_handler = logging.getHandlerByName("queue")
if queue_handler:
    queue_handler.listener.start()
    atexit.register(queue_handler.listener.stop)
```

Alternative: subclass `QueueHandler` to start thread in `__init__`.

## Library code
- Don't configure logging - let applications handle it
- Can still create loggers and log messages
- Just no `dictConfig` or adding handlers/formatters
- Default behavior: warnings+ to stderr
