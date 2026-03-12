# Error Handling Notes

## Core Principle

Failure handling should make the program more honest, not less.

Prefer explicit failure channels. Use assertions for bugs, exceptions for boundary rejection and unrecoverable preconditions, and tuple returns for expected operational failures where the caller has a meaningful choice.

The goal is not "never crash". The goal is to preserve correctness and make control flow obvious.

## The Three Failure Kinds

### 1. Programmer Errors

This means our code is wrong. An invariant was broken. A state that should be impossible happened anyway.

Use `assert`.

```python
def process_batch(items: list[Item]) -> list[Result]:
    assert items is not None
    assert len(items) > 0

    return [process_item(item) for item in items]
```

If this fails, we should crash loudly and fix the bug. Continuing would just spread corrupted assumptions deeper into the program.

### 2. Boundary Rejection

This means the entry point cannot proceed with the given input or environment.

Examples:
- Invalid CLI arguments
- Missing required config
- Malformed JSON at parse time
- Invalid request payload

Use exceptions at the boundary.

```python
def load_config(config_path: Path) -> Config:
    if not config_path.exists():
        raise ConfigNotFoundError(str(config_path))

    text = config_path.read_text()
    data = json.loads(text)
    return parse_config(data)
```

This is not "exceptional" in the emotional sense. It just means this entry point rejects the input and refuses to continue.

### 3. Operational Failures

This means the operation may legitimately fail, and the caller has a meaningful decision to make.

Examples:
- Network timeout
- Remote host unavailable
- GPU busy
- Cache miss
- Trying one provider, then another

Use explicit tuple returns: `(result, error)`.

```python
async def connect_to_gpu(config: Config) -> tuple[Connection | None, str | None]:
    if not config.host:
        return None, "host required"

    client, err = await open_connection(config.host)
    if err:
        return None, f"connection failed: {err}"

    return client, None
```

Tuple returns are a lightweight Python version of `Result[T, E]`: success and failure are part of the function's contract, not an ambient side channel.

## Why We Avoid Defensive try/except Everywhere

People are often too afraid of programs erroring. This leads to code that catches too much, falls back too eagerly, and keeps running on bad assumptions.

That style is dangerous because:
- It hides bugs instead of surfacing them
- It makes control flow harder to read
- It weakens invariants
- It creates false confidence: "didn't crash" gets mistaken for "worked"

Bad:

```python
def compute_total(order: Order) -> int:
    try:
        return sum(item.price_cents for item in order.items)
    except Exception:
        return 0
```

This is not recovery. It is lying.

Better:

```python
def compute_total(order: Order) -> int:
    assert order.items is not None
    return sum(item.price_cents for item in order.items)
```

## Fall Back Only When It Preserves Correctness

Do not add fallback code just because crashing feels scary.

Add fallback code only when the fallback is a real alternative plan that preserves correctness.

Good fallback:

```python
def load_config() -> Config:
    if LOCAL_CONFIG.exists():
        return parse_config_file(LOCAL_CONFIG)

    if DEFAULT_CONFIG.exists():
        return parse_config_file(DEFAULT_CONFIG)

    raise ConfigError("No config file found")
```

This is acceptable because:
- The fallback is intentional
- The ordering is explicit
- Failure is still surfaced if no valid plan works

Bad fallback:

```python
def load_user_settings(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}
```

This silently changes program meaning and makes bugs harder to find.

## Where try/except Is Okay

Use `try/except` at boundaries, where Python or external libraries force it, and then convert into your own error model.

Good uses:
- Filesystem and network I/O
- Wrapping library exceptions
- Optional imports
- Transaction rollback / cleanup
- Boundary parsing code

```python
def read_json(path: Path) -> tuple[dict | None, str | None]:
    try:
        text = path.read_text()
    except OSError as e:
        return None, f"read failed: {e}"

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"invalid json: {e}"

    return data, None
```

The important part is that the `try/except` stays near the boundary. Do not smear exception handling throughout core business logic just to keep the program limping along.

## What Core Logic Should Look Like

After parsing and validation, core logic should run on trusted data.

That means:
- Assertions for invariants
- Straight-line control flow
- Explicit tuple returns for expected operational failure
- No broad exception swallowing
- No fake default values

```python
def build_deployment_plan(config: Config) -> tuple[Plan | None, str | None]:
    assert config.model is not None
    assert config.region is not None

    capacity, err = check_capacity(config.region)
    if err:
        return None, err

    plan = make_plan(config, capacity)
    return plan, None
```

## Tuple Return Conventions

For single recoverable errors:
- Success: `(value, None)`
- Failure: `(None, error_message)`

For validation that should accumulate multiple issues:
- Return `list[str]`
- Empty list means success

```python
def validate_config(config: dict) -> list[str]:
    errors = []
    if "model" not in config:
        errors.append("Missing 'model'")
    if "learning_rate" not in config:
        errors.append("Missing 'learning_rate'")
    return errors
```

## Decision Guide

Ask:

1. Is this a bug in our code?
2. Is this invalid input or an unmet precondition at the boundary?
3. Is this an operational failure where the caller can retry, fallback, aggregate, or report?

Use:
- `assert` for `1`
- `raise` for `2`
- `(result, error)` for `3`

## Rules of Thumb

- Prefer honest failure to dishonest continuation.
- Catch errors to translate or recover, not to hide them.
- Do not use exceptions for routine domain control flow.
- Do not return fake defaults unless the default is part of the domain model.
- Do not add fallback code without being able to explain why the fallback is correct.
- If a failure means your assumptions were wrong, crash and fix the assumptions.
