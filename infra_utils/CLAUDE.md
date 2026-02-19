# infra_utils

Shared utilities for broker, bifrost, and other workspace packages. Credential discovery, validation, retry, SSH, logging.

## Key modules

### config.py — credential discovery

```python
from infra_utils.config import get_runpod_key, get_prime_key, get_lambda_key, get_vast_key

key = get_runpod_key()  # reads RUNPOD_API_KEY env var or .env file
```

Precedence: env var → `.env` file → None

### validation.py — Tiger Style validators

```python
from infra_utils.validation import validate_ssh_key_path, validate_timeout, validate_port

path = validate_ssh_key_path("~/.ssh/id_ed25519")  # raises AssertionError if invalid
timeout = validate_timeout(30, min_value=1, max_value=300)
port = validate_port(9100)
```

### retry.py — exponential backoff decorator

```python
from infra_utils.retry import async_retry

@async_retry(max_attempts=3, delay=1, backoff=2, exceptions=(httpx.HTTPError,))
async def fetch():
    ...
```

### logging_config.py — structured logging setup

```python
from infra_utils.logging_config import setup_logging

setup_logging(level="INFO", log_file="logs/app.jsonl")
logger = logging.getLogger(__name__)
logger.info("started", extra={"run_id": "abc"})
```

## Key gotchas

- `get_*_key()` returns `None` if not found — always check before using
- Validators use assertions (crash on invalid input — intended for boundary checking)
- Not for training metrics — use rollouts' own metrics system for that
