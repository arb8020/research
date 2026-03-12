# Error Handling

> **Core Principle:** Prefer honest failure to dishonest continuation.

Choose the error channel based on what kind of failure happened, not based on fear of crashing and not based on habit.

Use:
- `assert` for programmer errors and broken invariants
- `raise` for invalid input and unrecoverable preconditions at boundaries
- `(result, error)` tuple returns for expected operational failures where the caller has a meaningful choice

The goal is to preserve correctness and make control flow obvious.

---

## Quick Decision Guide

Ask:

1. Is this a bug in our code?
2. Is this invalid input or an unmet precondition at the boundary?
3. Is this an operational failure where the caller can retry, fallback, aggregate, or report?

Use:
- `assert` for `1`
- `raise` for `2`
- `(result, error)` for `3`

```text
Is this a programmer error?
  YES -> assert
  NO  ->

Is this boundary rejection or an unrecoverable precondition?
  YES -> raise
  NO  ->

Is this an expected operational failure with a meaningful caller choice?
  YES -> return (result, error)
  NO  -> raise
```

---

## The Three Failure Kinds

### 1. Programmer Errors -> `assert`

This means our code is wrong. An invariant broke. A state that should be impossible happened anyway.

```python
def process_batch(items: list[Item]) -> list[Result]:
    assert items is not None
    assert len(items) > 0

    return [process_item(item) for item in items]
```

Use `assert` when:
- A caller violated an internal contract
- Parsed data should already be trusted
- A control-flow assumption should never fail if the code is correct

Do not catch assertion failures just to keep going. If an invariant broke, the process is already in a state you did not design for.

### 2. Boundary Rejection -> `raise`

This means an entry point cannot proceed with the given input or environment.

Examples:
- Invalid CLI args
- Missing required config
- Malformed JSON at parse time
- Invalid HTTP payload
- Missing dependency at startup

```python
def load_config(config_path: Path) -> Config:
    if not config_path.exists():
        raise ConfigNotFoundError(str(config_path))

    text = config_path.read_text()
    data = json.loads(text)
    return parse_config(data)
```

Use exceptions here because:
- The current layer is rejecting the input
- There is no meaningful local recovery path
- The caller should see a clear failure and fix the input or environment

This is not “exceptions everywhere”. This is exceptions at the edges.

### 3. Operational Failures -> `(result, error)`

This means the operation may legitimately fail, and the caller has a real decision to make.

Examples:
- Network timeout
- Connection refused
- Remote host unavailable
- GPU busy
- Cache miss
- Trying one provider, then another

```python
async def connect_to_gpu(config: Config) -> tuple[Connection | None, str | None]:
    if not config.host:
        return None, "host required"

    client, err = await open_connection(config.host)
    if err:
        return None, f"connection failed: {err}"

    return client, None
```

Tuple returns are a lightweight Python version of `Result[T, E]`: success and failure are part of the function contract instead of ambient control flow.

Use tuple returns when:
- Retry makes sense
- Fallback makes sense
- Partial failure is acceptable
- The caller should decide how to surface the error

---

## Why We Avoid Defensive Exception Handling

People are often too afraid of programs erroring. That fear produces code full of broad `try/except`, fake defaults, and fallback branches that exist only to avoid a crash.

That style is dangerous because:
- It hides bugs instead of surfacing them
- It makes control flow harder to follow
- It weakens invariants
- It turns “the program kept running” into fake evidence that it worked

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

The style rule is:

> Do not add fallback code just because crashing feels scary. Add fallback code only when the fallback is a real alternative plan that preserves correctness.

---

## Fallbacks: When They Are Good and When They Are Bad

### Good fallback

Fallbacks are good when they are part of the intended design.

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
- The alternatives are explicit
- The behavior remains correct
- Failure is still surfaced if no valid plan works

### Bad fallback

```python
def load_user_settings(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}
```

This is bad because:
- `{}` may not be a valid substitute
- Program meaning changes silently
- Real bugs get hidden
- Downstream code now runs on untrusted assumptions

The distinction:
- Fallback = choose another valid plan
- Suppression = ignore an error and pretend things are fine

Allow the first. Be suspicious of the second.

---

## Where `try/except` Is Appropriate

Use `try/except` near boundaries where Python or external libraries force it, and then translate into your own error model.

Good places:
- Filesystem I/O
- Network I/O
- Wrapping third-party library exceptions
- Optional imports
- Cleanup and rollback
- Parsing external data

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

The important part is locality:
- Catch close to the source
- Translate into a smaller set of meaningful errors
- Do not smear `try/except` through core business logic

Avoid:
- `except Exception:` in core logic
- silent catches
- catch-and-log-then-continue without a real recovery plan

If you must catch broadly at a boundary, convert immediately and explain why.

---

## What Core Logic Should Look Like

After parsing and validation, core logic should run on trusted data.

That means:
- Assertions for invariants
- Straight-line control flow
- Tuple returns for expected operational failure
- No broad exception swallowing
- No fake defaults

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

The core of the program should not constantly defend itself from states that the boundaries already promised to reject.

---

## Return Conventions

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
    elif config["learning_rate"] <= 0:
        errors.append("learning_rate must be positive")
    return errors
```

Use richer error types if the caller needs structured recovery, but keep the control flow explicit.

---

## Relationship to `Result[T, E]`

A tuple return is the simplest Python version of a result type.

Conceptually:

```python
Result[T, E] = Ok(T) | Err(E)
```

A real `Result` type becomes attractive when:
- You are chaining many failable operations
- You want helpers like `.map()` or `.and_then()`
- You need more structured composition than raw tuples provide

For most code in this codebase, tuples are enough. The important thing is not the exact syntax. The important thing is that failure remains explicit and honest.

---

## Rules of Thumb

- Prefer honest failure to dishonest continuation.
- Catch errors to translate or recover, not to hide them.
- Do not use exceptions for routine domain control flow.
- Do not return fake defaults unless the default is part of the domain model.
- Do not add fallback code without being able to explain why the fallback is correct.
- If a failure means your assumptions were wrong, crash and fix the assumptions.
- Keep exception handling at the edges and keep the core logic clean.
