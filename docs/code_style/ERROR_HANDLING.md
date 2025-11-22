# Error Handling: Composable Results Over Defensive Exceptions

> **Core Principle:** Keep core logic clean. Don't litter business code with defensive try/except. Use composable Result types for railway-oriented programming and concurrent error collection.

---

## The Problem with Defensive Exception Handling

### Defensive Code is Noise

```python
# BAD - exception handling obscures control flow
def process_pipeline(data: dict) -> ProcessedData | None:
    try:
        validated = validate_input(data)
    except ValidationError as e:
        log_error(f"Validation failed: {e}")
        return None

    try:
        transformed = transform_data(validated)
    except TransformError as e:
        log_error(f"Transform failed: {e}")
        return None

    try:
        enriched = enrich_data(transformed)
    except EnrichError as e:
        log_error(f"Enrich failed: {e}")
        return None

    try:
        saved = save_data(enriched)
    except SaveError as e:
        log_error(f"Save failed: {e}")
        return None

    return saved
```

**Problems:**
- Control flow buried in exception handling
- Repetitive error logging
- Hard to see the happy path
- Each step needs its own try/except block

### Concurrent Error Collection is Painful

```python
# BAD - exceptions don't compose for concurrent work
async def process_batch(items: list[Item]) -> list[Result]:
    results = []
    errors = []

    for item in items:
        try:
            result = await process_item(item)
            results.append(result)
        except Exception as e:
            errors.append((item.id, str(e)))

    # Lost concurrency - processing serially
    # Can't use asyncio.gather because exceptions fail-fast

    if errors:
        # What now? Raise? Return partial? Log and continue?
        pass

    return results
```

**Problems:**
- Can't run concurrently (exceptions fail-fast)
- Lost track of which item failed
- Unclear how to handle partial failures

---

## The Solution: Composable Result Types

### Basic Result Type

```python
from dataclasses import dataclass
from typing import Generic, TypeVar, Callable

T = TypeVar('T')
E = TypeVar('E')

@dataclass(frozen=True)
class Result(Generic[T, E]):
    """Result type for composable error handling."""
    _value: T | None = None
    _error: E | None = None

    @staticmethod
    def ok(value: T) -> "Result[T, E]":
        """Create successful result."""
        return Result(_value=value)

    @staticmethod
    def err(error: E) -> "Result[T, E]":
        """Create failed result."""
        return Result(_error=error)

    @property
    def is_ok(self) -> bool:
        return self._error is None

    @property
    def value(self) -> T:
        """Get value (raises if error)."""
        if self._error is not None:
            raise ValueError(f"Called .value on Err: {self._error}")
        return self._value

    @property
    def error(self) -> E:
        """Get error (raises if ok)."""
        if self._error is None:
            raise ValueError("Called .error on Ok")
        return self._error

    def and_then(self, f: Callable[[T], "Result[T, E]"]) -> "Result[T, E]":
        """Chain operations (railway-oriented programming).

        If this is Err, skip f and propagate error.
        If this is Ok, apply f to the value.
        """
        if self._error is not None:
            return Result.err(self._error)
        return f(self._value)

    def map(self, f: Callable[[T], T]) -> "Result[T, E]":
        """Transform success value, pass through error."""
        if self._error is not None:
            return Result.err(self._error)
        return Result.ok(f(self._value))

    def unwrap_or(self, default: T) -> T:
        """Get value or default."""
        return self._value if self._error is None else default
```

### Clean Pipeline Code

```python
# GOOD - clean control flow with Result
def process_pipeline(data: dict) -> Result[ProcessedData, str]:
    """Railway-oriented: short-circuits on first error."""
    return (
        validate_input(data)
        .and_then(transform_data)
        .and_then(enrich_data)
        .and_then(save_data)
    )

# Individual steps return Result
def validate_input(data: dict) -> Result[ValidatedData, str]:
    if "model" not in data:
        return Result.err("Missing required field: model")

    if data.get("batch_size", 0) <= 0:
        return Result.err("batch_size must be positive")

    return Result.ok(ValidatedData(**data))

def transform_data(validated: ValidatedData) -> Result[TransformedData, str]:
    # Just return Result - no try/except needed
    if not validated.data_path.exists():
        return Result.err(f"Data not found: {validated.data_path}")

    transformed = apply_transform(validated)
    return Result.ok(transformed)
```

**Benefits:**
- Happy path is clear (the chain of operations)
- Error handling is explicit (return Result.err)
- No try/except spam
- Automatic short-circuiting

---

## Concurrent Error Collection

### The Pattern

```python
async def process_batch_concurrent(
    items: list[Item]
) -> tuple[list[ProcessedItem], list[tuple[int, str]]]:
    """Process all items concurrently, collect all errors."""

    # Run everything concurrently (don't fail-fast)
    results: list[Result[ProcessedItem, str]] = await asyncio.gather(
        *[process_item_safe(item) for item in items]
    )

    # Partition successes and failures
    successes = [r.value for r in results if r.is_ok]
    failures = [
        (i, r.error)
        for i, r in enumerate(results)
        if not r.is_ok
    ]

    return successes, failures


async def process_item_safe(item: Item) -> Result[ProcessedItem, str]:
    """Wrap processing in Result (no exceptions escape)."""
    try:
        result = await process_item(item)
        return Result.ok(result)
    except Exception as e:
        return Result.err(str(e))
```

**Usage:**
```python
successes, failures = await process_batch_concurrent(items)

# Handle batch errors together
if failures:
    log_batch_failures(failures)

# Continue with successes
if successes:
    save_batch(successes)
```

**Benefits:**
- Full concurrency (all items run in parallel)
- Track which items failed
- Collect all errors (not just first)
- Decide how to handle partial success

---

## Use the `returns` Library

Don't reinvent this. Use **[dry-python/returns](https://github.com/dry-python/returns)**:

```bash
pip install returns
```

### Railway-Oriented Programming

```python
from returns.result import Result, Success, Failure
from returns.pipeline import flow

# GOOD - clean pipeline with >> operator
def process_pipeline(data: dict) -> Result[ProcessedData, str]:
    return (
        validate_input(data)
        >> transform_data
        >> enrich_data
        >> save_data
    )  # Auto short-circuit on first Failure

# Or with flow()
def process_pipeline(data: dict) -> Result[ProcessedData, str]:
    return flow(
        data,
        validate_input,
        lambda r: r.bind(transform_data),
        lambda r: r.bind(enrich_data),
        lambda r: r.bind(save_data),
    )
```

### Wrapping Risky Operations

```python
from returns.result import safe

# Automatically wrap exceptions into Result
@safe
def load_config(path: Path) -> Config:
    """Raises become Failure automatically."""
    data = json.loads(path.read_text())  # JSONDecodeError → Failure
    return Config(**data)  # ValidationError → Failure

# Usage
result = load_config(Path("config.json"))
match result:
    case Success(config):
        print(f"Loaded: {config}")
    case Failure(error):
        print(f"Failed: {error}")
```

### IO Monad for Side Effects

```python
from returns.io import IO, impure_safe

# Mark impure operations explicitly
@impure_safe
def save_to_database(data: ProcessedData) -> None:
    db.insert(data)  # Side effect

# Compose pure and impure operations
def process_and_save(data: dict) -> IO[Result[None, str]]:
    return (
        validate_input(data)
        >> transform_data
        >> save_to_database  # IO[Result[None, Exception]]
    )
```

---

## Assertions vs Production Invariants

### Critical: The `-O` Flag

**Python's `-O` flag strips all `assert` statements!**

While rarely used in practice, the failure mode is catastrophic: production code silently skips invariant checks.

### The Distinction (Tiger Style)

> "Assertions detect **programmer errors**. Unlike **operating errors**, which are expected and which must be handled, assertion failures are unexpected."

**Programmer errors** (use `assert` - development only):
- Off-by-one bugs
- Type mismatches
- Violated preconditions
- "This should never happen if code is correct"

**Operating errors** (use `if` + Result/Exception - always enforced):
- Invalid user input
- Missing files
- Network failures
- Configuration validation

### The Pattern

```python
# WRONG - assertion for production invariant
def compute_attention(embeddings, num_heads):
    assert embeddings.shape[-1] % num_heads == 0  # STRIPPED WITH -O!
    ...

# RIGHT - distinguish debug vs production
def compute_attention(embeddings, num_heads):
    # Production invariant (always enforced)
    if embeddings.shape[-1] % num_heads != 0:
        raise ValueError(
            f"Embedding dim {embeddings.shape[-1]} must be divisible "
            f"by num_heads {num_heads}"
        )

    # Debug assertions (development only)
    assert embeddings.ndim == 2, "Expected 2D embeddings"
    assert num_heads > 0, "num_heads must be positive"

    # Safe to proceed - invariant is guaranteed
    head_dim = embeddings.shape[-1] // num_heads
    ...
```

### With Result Types

```python
def compute_attention(
    embeddings: Tensor,
    num_heads: int
) -> Result[Tensor, str]:
    """Returns Err for invalid inputs."""

    # Production validation (always enforced)
    if embeddings.shape[-1] % num_heads != 0:
        return Result.err(
            f"Embedding dim {embeddings.shape[-1]} must be divisible "
            f"by num_heads {num_heads}"
        )

    # Debug assertions (development checks)
    assert embeddings.ndim == 2
    assert num_heads > 0

    attention = compute_attention_scores(embeddings, num_heads)
    return Result.ok(attention)
```

---

## When to Use Exceptions

### When Exceptions Are Actually Better

**Use exceptions instead of Result if:**

1. **Quick scripts/prototypes** - Overhead of Result types not worth it for throwaway code
2. **Team strongly prefers idiomatic Python** - Social factors matter (will they fight the pattern?)
3. **Heavy exception-based library integration** - Wrapping every third-party call is tedious
4. **Simple fail-fast behavior** - If you always want to crash on error, exceptions are fine

**The tradeoff is real.** Result types add:
- More boilerplate (wrapping/unwrapping)
- Learning curve for team members
- Friction with Python ecosystem
- Verbose compared to try/except

**Choose consciously based on:**
- **Project longevity** - Throwaway script vs maintained codebase?
- **Team preferences** - Will they embrace or resist the pattern?
- **Error handling complexity** - Simple fail-fast vs complex recovery/composition?
- **Concurrency needs** - Do you need to collect multiple failures?

**Both are valid.** This document shows the Result approach for when you want:
- Composable error handling
- Concurrent error collection
- Explicit control flow in business logic

But don't use Results dogmatically if exceptions fit your context better.

### At System Boundaries

Exceptions are appropriate at **boundaries** where you convert internal Results to external representations:

```python
# Internal: clean Result-based code
def process_pipeline(data: dict) -> Result[ProcessedData, str]:
    return (
        validate_input(data)
        >> transform_data
        >> save_data
    )

# Boundary: CLI entry point
def main():
    result = process_pipeline(load_config())

    match result:
        case Success(data):
            print(f"Success: processed {len(data.items)} items")
            sys.exit(0)
        case Failure(error):
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)

# Boundary: API handler
@app.post("/api/process")
def api_process(data: dict):
    result = process_pipeline(data)

    match result:
        case Success(processed):
            return {"status": "success", "data": processed}
        case Failure(error):
            return {"status": "error", "message": error}, 400

# Boundary: wrapping third-party libs
from returns.result import safe

@safe
def call_external_api(endpoint: str) -> dict:
    """Exceptions from requests → Failure."""
    response = requests.get(endpoint)  # Can raise
    response.raise_for_status()  # Can raise
    return response.json()  # Can raise
```

### For Third-Party Integration

When working with libraries that raise exceptions:

```python
# Wrap at the boundary
def load_model_safe(path: Path) -> Result[Model, str]:
    """Wrap torch.load exceptions."""
    try:
        model = torch.load(path)  # Can raise various exceptions
        return Result.ok(model)
    except FileNotFoundError:
        return Result.err(f"Model not found: {path}")
    except RuntimeError as e:
        return Result.err(f"Failed to load model: {e}")
    except Exception as e:
        return Result.err(f"Unexpected error: {e}")

# Now use in pipeline
def setup_training(config: Config) -> Result[TrainingSetup, str]:
    return (
        load_model_safe(config.model_path)
        >> initialize_optimizer
        >> load_dataset
    )
```

---

## Matklad's Separation: Handling vs Reporting

From [Error Codes for Control Flow](https://matklad.github.io/2025/11/06/error-codes-for-control-flow.html):

> "Displaying an error message to the user is a different aspect of error handling than branching based on a specific error condition."

### Two Concerns

1. **Error Handling** (branching/recovery) - needs error **type/kind**
2. **Error Reporting** (diagnostics) - needs error **details/context**

### The Diagnostic Sink Pattern (from Zig)

```python
@dataclass
class Diagnostics:
    """Collect detailed error information for reporting."""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, msg: str):
        self.errors.append(msg)

    def warning(self, msg: str):
        self.warnings.append(msg)

    def has_errors(self) -> bool:
        return len(self.errors) > 0


class ErrorKind(Enum):
    """Simple error codes for branching."""
    PARSE_ERROR = "parse_error"
    VALIDATION_ERROR = "validation_error"
    IO_ERROR = "io_error"


def parse_config(
    source: str,
    diag: Diagnostics | None = None
) -> Result[Config, ErrorKind]:
    """
    If caller wants to handle: pass diag=None, switch on ErrorKind
    If caller wants to report: pass Diagnostics, extract messages
    """

    try:
        data = json.loads(source)
    except json.JSONDecodeError as e:
        if diag:
            diag.error(f"JSON parse error at line {e.lineno}: {e.msg}")
        return Result.err(ErrorKind.PARSE_ERROR)

    if "model" not in data:
        if diag:
            diag.error("Missing required field 'model'")
            diag.warning("See docs/config.md for schema")
        return Result.err(ErrorKind.VALIDATION_ERROR)

    return Result.ok(Config(**data))


# Usage 1: Handle the error (branch on kind)
result = parse_config(source, diag=None)
match result:
    case Success(config):
        use_config(config)
    case Failure(ErrorKind.PARSE_ERROR):
        try_alternative_format()
    case Failure(ErrorKind.VALIDATION_ERROR):
        use_default_config()

# Usage 2: Report the error (show diagnostics)
diag = Diagnostics()
result = parse_config(source, diag)
if not result.is_ok:
    for error in diag.errors:
        print(f"Error: {error}", file=sys.stderr)
    for warning in diag.warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    sys.exit(1)
```

---

## Summary: The Recommendations

### Core Principles

1. **Keep business logic clean** - No defensive try/except spam
2. **Use Result types internally** - Composable, explicit control flow
3. **Convert at boundaries** - Results → Exceptions/HTTP/CLI responses
4. **Distinguish assertions from invariants** - Never use `assert` for production checks
5. **Separate handling from reporting** - Error kind (for branching) vs diagnostics (for user)

### The Pattern

```python
# Internal: Result-based pipeline
def process(data: dict) -> Result[Output, str]:
    return (
        validate(data)
        >> transform
        >> save
    )

# Boundary: Convert to appropriate representation
def main():
    match process(data):
        case Success(output): handle_success(output)
        case Failure(error): handle_error(error)
```

### Production Invariants

```python
# NEVER use assert for production checks
if critical_invariant_violated:
    raise ValueError("Invariant violated")  # Or return Result.err()

# Use assert only for development
assert precondition_from_caller, "Caller should guarantee this"
```

### Concurrent Error Collection

```python
# Run all, collect errors
results = await asyncio.gather(*[process(item) for item in items])
successes = [r.value for r in results if r.is_ok]
failures = [(i, r.error) for i, r in enumerate(results) if not r.is_ok]
```

### Use Libraries

- **[dry-python/returns](https://github.com/dry-python/returns)** - Result types, railway-oriented programming
- **[result](https://github.com/rustedpy/result)** - Rust-like Result for Python
- Don't reinvent monads

---

## Further Reading

- [Matklad: Error Codes for Control Flow](https://matklad.github.io/2025/11/06/error-codes-for-control-flow.html)
- [Matklad: Error ABI](https://matklad.github.io/2025/11/09/error-ABI.html)
- [Tiger Style](https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md) - Assertions for programmer errors
- [Joe Duffy: Error Model](https://joeduffyblog.com/2015/12/19/safe-native-code/#error-model)
- [Railway Oriented Programming](https://fsharpforfunandprofit.com/rop/) (F# but principles apply)
