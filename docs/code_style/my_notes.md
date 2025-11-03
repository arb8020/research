# My Code Style Notes

## Error Handling and Control Flow

**Core principle:** Control flow should be explicit and linear. Prefer tuple returns for errors over try/except.

**Why:** try/except obscures control flow. When you see a try block, it's unclear:
- Is the exception case rare (1 in 1000) or common (1 in 2)?
- What's the actual execution path?
- Are we using exceptions for normal logic or truly exceptional cases?

Explicit control flow with if/else makes the code's behavior obvious.

### Internal code always uses tuple returns

```python
# ✅ Good: Explicit error handling with tuple returns
def validate_config(config: dict) -> list[str]:
    """Returns list of errors, empty if valid."""
    errors = []
    if "model" not in config:
        errors.append("Missing 'model'")
    if "learning_rate" not in config:
        errors.append("Missing 'learning_rate'")
    elif config["learning_rate"] <= 0:
        errors.append("learning_rate must be positive")
    return errors

def process(data) -> tuple[Result | None, str | None]:
    """Returns (result, error). Error is None on success."""
    if not is_valid(data):
        return None, "invalid data"

    intermediate, err = transform(data)
    if err:
        return None, f"transform failed: {err}"

    return intermediate, None

# Caller has explicit control flow
result, err = process(data)
if err:
    print(f"Error: {err}")
    return 1
```

### Even deep call stacks use tuple returns

Explicit > concise. Yes, it's more verbose, but control flow is crystal clear:

```python
# ✅ Good: Verbose but explicit
def parse_config(text: str) -> tuple[Config | None, str | None]:
    tokens, err = tokenize(text)
    if err:
        return None, f"Tokenize failed: {err}"

    sections, err = parse_sections(tokens)
    if err:
        return None, f"Parse failed: {err}"

    config, err = build_config(sections)
    if err:
        return None, f"Build failed: {err}"

    return config, None

# ❌ Bad: Concise but implicit (exceptions hide control flow)
def parse_config(text: str) -> Config:
    tokens = tokenize(text)  # might raise - when? always? sometimes?
    sections = parse_sections(tokens)  # might raise - same error type?
    return Config(sections)
```

### Only use try/except when unavoidable

**When you MUST use try/except:**

1. **Wrapping external library calls** - they raise, you can't change that
2. **Transaction rollback patterns** - need exception to trigger cleanup
3. **Optional imports** - idiomatic Python
4. **External I/O** - filesystem, network operations

But even then, **return tuples** to your caller:

```python
# ✅ Good: Use try/except at boundary, but return tuple
def load_file(path: Path) -> tuple[str | None, str | None]:
    try:
        content = path.read_text()  # External: must use try/except
        return content, None
    except OSError as e:
        return None, f"Failed to read {path}: {e}"

def load_model(model_id: str) -> tuple[Model | None, str | None]:
    if not model_id:
        return None, "model_id required"

    try:
        model = AutoModel.from_pretrained(model_id)  # External lib
        return model, None
    except Exception as e:
        return None, f"Failed to load {model_id}: {e}"

# Optional imports
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
```

### Assertions for programmer errors

From Tiger Style: Use assertions for bugs in our code (preconditions, invariants):

```python
def process(data):
    assert data is not None, "caller must provide data"
    assert len(data) > 0, "caller must validate non-empty"

    # ... rest of logic
```

Assertions are for things that should **never** happen if our code is correct. They crash the program if violated.

### Return type conventions

**For single errors:**
- Success: `(value, None)`
- Failure: `(None, error_message)`

**For multiple errors (validation):**
- Return `list[str]` (empty list = success)

```python
# Single error
def get_user(user_id: str) -> tuple[User | None, str | None]:
    if not user_id:
        return None, "user_id required"
    user = db.query_one_or_none(...)
    if not user:
        return None, "user not found"
    return user, None

# Multiple errors
def validate_config(config: dict) -> list[str]:
    errors = []
    # ... collect all errors ...
    return errors  # empty = valid
```

### When exceptions ARE ok for control flow

There are rare cases where exceptions are genuinely the right tool:

1. **Transaction rollback** - exception triggers cleanup
   ```python
   def transfer_money(from_account, to_account, amount):
       try:
           db.begin_transaction()
           debit(from_account, amount)
           credit(to_account, amount)
           db.commit()
       except Exception:
           db.rollback()
           raise  # or return None, "Transaction failed"
   ```

2. **Deep parsing/validation of external data** where the whole call stack is "at the boundary"
   - Note: We still prefer tuple returns even here for explicitness
   - But if you're processing user input through 10+ functions, exceptions are pragmatic
   - In Rust you'd use `Result<T, E>` with `?`, Python doesn't have that

**Key test:** Is the exception propagating from an external boundary? If yes, it might be justified. If no, use tuple returns.

### Design APIs to avoid forcing try/except on callers

Provide explicit variants so callers can choose their control flow:

```python
# ❌ Bad: Forces callers into try/except for normal "not found" case
def get_user(user_id: str) -> User:
    """Raises UserNotFound if not exists."""
    user = db.query_one(...)
    if not user:
        raise UserNotFound(user_id)
    return user

# Caller must use try/except for normal operation:
try:
    user = get_user(user_id)
except UserNotFound:
    user = create_user(user_id)

# ✅ Good: Provide explicit options
def get_user(user_id: str) -> User:
    """Get user by ID. Raises UserNotFound if not exists."""
    ...

def get_user_or_none(user_id: str) -> User | None:
    """Get user by ID, or None if not exists."""
    ...

def get_user_or_create(user_id: str) -> User:
    """Get user by ID, creating if necessary."""
    ...

# Now callers choose clean control flow:
user = get_user_or_create(user_id)  # No try/except needed
```

This is Casey Muratori's "redundancy" principle - give users options so they're not forced into awkward patterns.
