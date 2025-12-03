# Avoiding LLM Code Patterns

> **Core Principle:** LLMs tend toward verbosity, defensiveness, and over-abstraction. Good code is concise, trusts its boundaries, and solves the problem at hand.

---

## Common Patterns to Avoid

### 1. Unnecessary Comments

LLMs love to explain obvious code:

```python
# BAD - comment restates the code
def get_user(user_id: int) -> User:
    # Get the user from the database
    user = db.query(User).filter(User.id == user_id).first()
    # Return the user
    return user

# GOOD - code speaks for itself
def get_user(user_id: int) -> User:
    return db.query(User).filter(User.id == user_id).first()
```

**Watch for:**
- Comments that restate what the code does (`# increment counter` before `counter += 1`)
- Section headers for 3-line sections (`# --- Validation ---`)
- Docstrings on obvious private helpers
- Comments explaining standard library functions

**Keep comments for:**
- *Why* something non-obvious is done
- Business logic that isn't clear from code
- Links to issues/docs for workarounds
- Invariants and assumptions

### 2. Defensive Try/Catch Everywhere

LLMs wrap everything in try/except "just in case":

```python
# BAD - defensive exception handling in trusted code path
def process_validated_data(data: ValidatedData) -> Result:
    try:
        transformed = transform(data)
    except Exception as e:
        logger.error(f"Transform failed: {e}")
        return None

    try:
        enriched = enrich(transformed)
    except Exception as e:
        logger.error(f"Enrich failed: {e}")
        return None

    return enriched

# GOOD - trust your boundaries, let errors propagate
def process_validated_data(data: ValidatedData) -> Result:
    transformed = transform(data)
    enriched = enrich(transformed)
    return enriched
```

**The rule:** Validate at boundaries, trust internally. See [ERROR_HANDLING.md](ERROR_HANDLING.md).

**Watch for:**
- Try/except around code that "shouldn't fail"
- Catching `Exception` and logging without re-raising
- Defensive null checks after functions that never return null
- `if x is not None` when x is guaranteed by the type system

### 3. Type Escape Hatches

LLMs use `Any` and `# type: ignore` to silence type errors:

```python
# BAD - casting to escape type system
def process(data: dict) -> Result:
    config: Any = data.get("config")  # Gave up on types
    return run(config)  # type: ignore[arg-type]

# GOOD - fix the actual type issue
def process(data: ProcessInput) -> Result:
    return run(data.config)
```

**Watch for:**
- `Any` annotations (grep for `: Any`)
- `# type: ignore` comments without explanation
- Casts that "fix" type errors (`cast(Foo, something_else)`)
- Union types that include `None` unnecessarily

### 4. Over-Abstraction

LLMs create abstractions for single-use code:

```python
# BAD - abstraction for one use case
class DataProcessorFactory:
    def create_processor(self, config: Config) -> DataProcessor:
        return DataProcessor(config)

class DataProcessor:
    def __init__(self, config: Config):
        self.config = config

    def process(self, data: Data) -> Result:
        return transform(data, self.config)

# Usage (the only usage)
processor = DataProcessorFactory().create_processor(config)
result = processor.process(data)

# GOOD - just write the code
result = transform(data, config)
```

**Watch for:**
- Factory classes with one factory method
- Wrapper classes that just delegate
- "Manager", "Handler", "Processor" classes with one method
- Abstractions created before the second use case exists

### 5. Verbose Variable Names

LLMs favor overly descriptive names:

```python
# BAD - names longer than the logic
user_authentication_response_data = authenticate(user)
processed_user_authentication_result = process(user_authentication_response_data)

# GOOD - concise and clear
auth = authenticate(user)
result = process(auth)
```

**The balance:**
- Short names for short scopes (`i`, `x`, `err`)
- Descriptive names for long-lived/wide-scope variables
- Domain terms over generic descriptions (`user` not `user_object_instance`)

### 6. Redundant Validation

LLMs validate the same thing multiple times:

```python
# BAD - re-validating already validated data
def outer(user_input: str) -> Result:
    if not user_input:
        raise ValueError("Input required")
    validated = validate(user_input)
    return inner(validated)

def inner(data: ValidatedData) -> Result:
    # Already validated! This is paranoid.
    if not data:
        raise ValueError("Data required")
    if not data.field:
        raise ValueError("Field required")
    return process(data)

# GOOD - validate once at boundary, trust downstream
def outer(user_input: str) -> Result:
    validated = validate(user_input)  # All validation here
    return inner(validated)

def inner(data: ValidatedData) -> Result:
    assert data.field  # Invariant, not validation
    return process(data)
```

### 7. Inconsistent Style

LLMs don't maintain file-local conventions:

```python
# File uses single quotes everywhere...
config = {'key': 'value'}
name = 'hello'

# ...then LLM adds code with double quotes
new_config = {"other_key": "other_value"}  # Inconsistent!
message = "world"  # Inconsistent!
```

**Watch for:**
- Quote style changes
- Naming convention changes (`snake_case` vs `camelCase`)
- Import style changes (absolute vs relative)
- Different patterns for the same operation

---

## Prevention Strategies

### 1. Read Before Writing

Before modifying a file, understand its conventions:
- How are errors handled?
- What's the comment density?
- What naming patterns are used?
- What abstractions exist?

### 2. Delete Aggressively

After LLM generates code, ask:
- Can I delete this comment?
- Can I delete this try/except?
- Can I inline this helper?
- Can I remove this null check?

### 3. Trust the Type System

If the type says it's not null, don't check for null. If the type is wrong, fix the type.

### 4. Match the Neighborhood

New code should be indistinguishable from existing code. If you can tell which parts were AI-generated, something's wrong.

---

## Quick Checklist

Before committing LLM-generated code:

- [ ] Removed comments that restate the code
- [ ] Removed try/except that catches "just in case"
- [ ] No new `Any` types or `# type: ignore`
- [ ] No new abstractions for single-use code
- [ ] Variable names match file conventions
- [ ] Validation happens at boundaries only
- [ ] Style matches the rest of the file
