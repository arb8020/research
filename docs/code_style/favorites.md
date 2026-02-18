# Favorites - The Essential Patterns

> **The Meta-Pattern**: Make the invisible visible. Make implicit assumptions explicit. Make control flow obvious. If someone reading your code has to simulate it mentally to understand what it does, you failed. The code should read like instructions to a human, not incantations for a compiler.

---

## The Workflow

**Starting something new:**

1. **Write usage code first** → What do I *want* this to look like?
2. **Write code that makes it work** → Get it working, don't abstract yet
3. **Apply explicit control flow + semantic compression** → Refactor for clarity and reuse

---

## Core Favorites

### 1. Write Usage Code First *(Casey Worst API)*

> "Always write the usage code first... this is the only way to get a nice, clean perspective on how the API would work if it had no constraints."

**Why it matters:** When you design APIs "in your head," you miss awkward interfaces. Write 5 lines of dream usage code first. If it feels clunky, the API is clunky.

**The ETW example:** Microsoft needed elevated privileges + a dedicated thread + manual memory layout just to copy kernel events to userspace. If they'd written usage code first, they'd have seen the absurdity immediately.

```python
# Write THIS first
config = Config()
config.training.learning_rate = 1e-3
train(config)

# Not this
config = ConfigBuilder() \
    .with_training_params(TrainingParams.builder()
        .learning_rate(1e-3)
        .build())
    .build()
```

---

### 2. Don't Reuse Until 2+ Examples *(Casey Semantic Compression)*

> "Like a good compressor, I don't reuse anything until I have at least two instances of it occurring. My mantra is, 'make your code usable before you try to make it reusable'."

**Why it matters:** Prevents premature abstraction. When you see the same thing twice, you have real examples to compress. Not speculative "maybe I'll need this."

**The discipline:**
- First time: Write it specific to the problem
- Second time: Write it specific again (copy-paste is fine!)
- Third time: Now extract the common parts

**Anti-pattern:**
```python
# NO - premature abstraction
class DataProcessor:
    def process(self, data, config):
        # Generic processing logic that tries to handle all cases
        pass
```

**Better:**
```python
# First time - just solve the problem
def process_user_data(users):
    cleaned = [u for u in users if u.email]
    return cleaned

# Second time - solve it again
def process_event_data(events):
    cleaned = [e for e in events if e.timestamp]
    return cleaned

# Third time - NOW compress
def filter_by_field(items, field_name):
    return [item for item in items if getattr(item, field_name)]
```

---

### 3. Continuous Granularity *(Casey Granularity)*

> "It is always important to avoid granularity discontinuities... never supply a higher-level function that can't be trivially replaced by a few lower-level functions."

**Why it matters:** Users can start simple, drop down when needed. No "hit a wall" moments where you're forced to rewrite everything.

**The pattern:**
```python
# Low level - full control
draw_button(x, y, width, height, "Click Me", colors...)

# Mid level - common case
if layout.push_button("Click Me"):
    do_thing()

# High level - even simpler
layout.bool_button("Enabled", &enabled)
```

Each level *uses* the lower level. Don't delete the lower functions when you add higher ones.

**Anti-pattern:** Only providing the high-level function, forcing users to either use it exactly as-is or reimplement from scratch.

---

### 4. Abstraction = Coupling *(CodeAesthetic)*

> "I consider coupling to be an equal and opposite reaction of abstraction. For every bit of abstraction you add, you've added more coupling."

**Why it matters:** Abstraction isn't free. Every extraction couples things together. Only abstract when the benefit outweighs the coupling cost.

**The test:** Before extracting a common base class, ask:
- What am I coupling together?
- Is the benefit worth forcing these things to share a contract?
- What flexibility am I losing?

**Example:**
```python
# BAD - couples both to file-based input
class FileSaver:
    def __init__(self, filename):
        self.filename = filename

class XMLWriter(FileSaver): pass
class JSONWriter(FileSaver): pass

# Now you can't have DatabaseWriter or CloudWriter
# You've coupled "saving" to "files"

# BETTER - keep them separate
class XMLWriter:
    def __init__(self, filename): ...

class JSONWriter:
    def __init__(self, filename): ...

# No coupling. Can add DatabaseWriter without breaking abstraction.
```

---

### 5. Assertions Everywhere, Split Them *(Tiger Style)*

> "The assertion density of the code must average a minimum of two assertions per function. Split compound assertions: prefer assert(a); assert(b); over assert(a and b)."

**Why it matters:** When an assertion fails, you know *exactly* which invariant broke. Compound assertions hide this information.

**The pattern:**
```python
def process(data):
    assert data is not None  # If this fails, you know: data was None
    assert len(data) > 0      # If this fails, you know: data was empty
    assert all(d.valid for d in data)  # If this fails, you know: invalid item

    # NOT: assert data is not None and len(data) > 0
    # Because then you don't know which condition failed
```

**Also use assertions for documentation:**
```python
# Document invariant relationships
assert batch_size * num_gpus == total_batch_size
assert embedding_dim % num_heads == 0  # Must be evenly divisible
```

---

### 6. Push Ifs Up, Fors Down *(Tiger Style)*

> "Centralize control flow... try to keep all switch/if statements in the 'parent' function, and move non-branchy logic fragments to helper functions. Centralize state manipulation. Let the parent function keep all relevant state in local variables, and use helpers to compute what needs to change."

**Why it matters:** When debugging, you follow the ifs. When changing logic, you edit the helpers. Clean separation of concerns.

**The pattern:**
```python
# PARENT: Has all the control flow
def process_request(request):
    if request.needs_auth:
        user = authenticate(request)
        if not user:
            return error_response("Unauthorized")
    else:
        user = None

    if request.type == "query":
        data = fetch_data(request.params)
        result = transform_query_result(data)
    else:
        result = transform_command_result(request.params)

    return success_response(result)

# HELPERS: Pure computation, no branching
def authenticate(request):
    token = extract_token(request)
    return lookup_user(token)

def transform_query_result(data):
    return {"results": data, "count": len(data)}
```

Parent has the ifs. Helpers do work. Easy to trace, easy to modify.

---

### 7. State Invariants Positively *(Tiger Style)*

> "Negations are not easy! State invariants positively."

**Why it matters:** Reduces cognitive load. The positive form reads naturally with how you think about bounds.

**The pattern:**
```python
# GOOD - reads naturally
if index < length:
    # Valid: we can access index
    process(items[index])
else:
    # Invalid: index out of bounds
    handle_error()

# HARDER TO READ
if index >= length:
    # It's NOT true that the invariant holds
    handle_error()
```

The first form matches how you think: "index is less than length" is the happy path.

---

### 8. Minimize Stateful Components *(Sean System Design)*

> "You should try and minimize the amount of stateful components in any system... A stateful service can't be automatically repaired."

**Why it matters:** Stateless services can crash and restart cleanly. Stateful services get into bad states that require manual intervention.

**The pattern:**
```python
# STATELESS - can restart anytime
def render_pdf(file_data):
    return convert_to_html(file_data)

# STATEFUL - if it crashes, state is lost
class PDFProcessor:
    def __init__(self):
        self.cache = {}
        self.queue = []

    def process(self, file_data):
        # What happens if this crashes mid-processing?
        pass
```

**System design rule:** One service owns each piece of state. Other services are stateless and call it.

---

### 9. Single Assignment *(Carmack SSA)*

> "You should strive to never reassign or update a variable outside of true iterative calculations in loops."

**Why it matters:** Every intermediate value stays visible in the debugger. Self-documenting transformations. No "which version of this variable is this?"

**The pattern:**
```python
# GOOD - each transformation has a name
raw_data = fetch_from_db()
filtered_data = [d for d in raw_data if d.valid]
sorted_data = sorted(filtered_data, key=lambda d: d.timestamp)
result = transform_to_response(sorted_data)

# BAD - reusing same name
data = fetch_from_db()
data = [d for d in data if d.valid]  # Which 'data' is this?
data = sorted(data, key=lambda d: d.timestamp)
result = transform_to_response(data)
```

When you hit a breakpoint, you can inspect `raw_data`, `filtered_data`, `sorted_data` separately.

---

### 10. No Magic Constants *(Casey Worst API)*

> "Microsoft didn't ever give them symbolic names. So you're just supposed to read the documentation and remember that 1 means the timestamps come from QueryPerformanceCounter."

**Why it matters:**
- Makes code readable (USE_QUERY_PERFORMANCE_COUNTER vs 1)
- Makes code searchable (can grep for constant name)
- Makes code robust to API changes (new SDK can deprecate the constant)

**The pattern:**
```python
# BAD
context.timestamp_type = 1  # What does 1 mean?

# GOOD
TIMESTAMP_QUERY_PERFORMANCE_COUNTER = 1
TIMESTAMP_SYSTEM_TIME = 2
TIMESTAMP_CPU_CYCLE = 3

context.timestamp_type = TIMESTAMP_QUERY_PERFORMANCE_COUNTER
```

Only exception: 0, 1, -1 in obvious contexts (like `count = 0`).

---

## Error Handling Decision Tree

```
Is this a programmer error (bug in my code)?
  YES → assert (dev only, stripped with -O)
  NO ↓

Is this input validation at a system boundary?
  YES → raise exception (parse, validate, fail fast)
  NO ↓

Can the caller meaningfully recover or retry?
  YES → tuple return (result, error) or Result type
  NO → raise exception
```

| Pattern | When | Example |
|---------|------|---------|
| `assert` | Internal invariants, programmer bugs | `assert len(items) > 0` |
| `raise` | Boundary violations, invalid input, can't continue | `raise ConfigNotFoundError(path)` |
| `(result, err)` | Operational failures, caller decides recovery | `return None, "SSH failed"` |

**Same error, different contexts:**
```python
# Config missing at startup → Exception (fix config and retry)
if not config_path.exists():
    raise ConfigNotFoundError(config_path)

# SSH key missing during deployment → Tuple (try alternatives)
if not key_path.exists():
    return None, f"SSH key not found: {key_path}"
```

**Critical:** Never use `assert` for production checks - Python's `-O` flag strips them!

---

## Function Length & Decomposition

### 70 Line Max, Meaningful Splits *(Tiger Style)*

> "Restrict the length of function bodies to reduce the probability of poorly structured code. We enforce a hard limit of 70 lines per function."

**Why it matters:** Forces you to think about proper decomposition. If you can't fit it in 70 lines, you haven't found the right helper functions yet.

**Guidelines for splitting:**
- Parent has control flow (ifs/switches)
- Helpers are pure computation
- Each function has one clear purpose
- "Hourglass shape": few params, simple return, meaty logic inside

**Good split:**
```python
def handle_request(request):  # ~40 lines, all control flow
    if not request.valid:
        return error_response("Invalid")

    if request.needs_auth:
        user, err = authenticate(request)
        if err:
            return error_response(err)
    else:
        user = None

    data, err = fetch_data(request.params)
    if err:
        return error_response(err)

    result = transform_result(data, request.format)
    return success_response(result, user)

def transform_result(data, format):  # ~20 lines, pure computation
    if format == "json":
        return jsonify(data)
    elif format == "xml":
        return xmlify(data)
    else:
        return data
```

---

## Classes vs Functions

**The test:**
1. Does it own a resource (socket, process, file, GPU memory pool)? → **Class**
2. Does it need cleanup (`shutdown()`, `close()`)? → **Class**
3. Is it just data? → **Frozen dataclass**
4. Is it mutable state without resource ownership? → **State dict + pure functions**
5. Is it computation/orchestration? → **Pure function**

| Use Case | Pattern |
|----------|---------|
| Config, metrics, data | `@dataclass(frozen=True)` |
| Resource ownership, lifecycle | Regular class |
| Mutable state (no resources) | State dict + functions |
| Math, transforms, batch prep | Pure function |
| Training loops, orchestration | Pure function calling objects |

**The nmoe pattern: State dict + pure functions** *(from Noumena-Network/nmoe)*

When you have mutable state but don't own resources, use a caller-provided dict instead of `self`:

```python
# BAD - class hides state
class Optimizer:
    def __init__(self):
        self.state = {}  # Hidden in object
    def step(self, params):
        # How do you checkpoint self.state? Fragile.
        pass

# GOOD - state dict passed explicitly (nmoe pattern)
def step(params, *, state: dict):
    """State lives in caller-provided dict."""
    if "exp_avg" not in state:
        state["exp_avg"] = torch.zeros_like(params)
    state["step"] = state.get("step", 0) + 1
    # Caller owns state, can checkpoint/inspect/compose

# Usage
optimizer_state = {}  # Caller owns this
step(params, state=optimizer_state)
step(params, state=optimizer_state)
save_checkpoint({"optimizer": optimizer_state})  # Easy!
```

**Why this works:**
- **Transparent**: All state visible in caller's scope
- **Checkpointable**: State dicts serialize trivially
- **Testable**: Pass different dicts for testing
- **Composable**: Easy to combine multiple state dicts
- **No hidden mutations**: Functions don't hide state changes

**The pattern: Functions orchestrate objects**
```python
# Objects for resource ownership only
backend = PyTorchBackend(model, optimizer)  # owns GPU tensors
buffer = DataBuffer(dataset)                 # owns file handles

# State dicts for mutable state
optimizer_state = {}
cache_state = {}

# Functions orchestrate (explicit inputs/outputs)
result, err = run_training(config, backend, buffer, optimizer_state)
loss = compute_loss(logprobs, advantages, beta)
```

**When in doubt, start with a function + state dict. Upgrade to class only when you own resources.**

---

## The One Rule to Rule Them All

**If someone reading your code has to simulate it mentally to understand what it does, you failed.**

Code should read like instructions to a human:
- Explicit control flow (no hidden magic)
- Named intermediate values (SSA style)
- Clear boundaries (stateless helpers)
- Obvious invariants (assertions)
- Natural progression (positive conditions)

**The ultimate test:**
1. Can I explain this to someone in 30 seconds?
2. If I debug this at 3am, will I understand it?
3. If requirements change, what breaks?
4. Did I write the usage code first and like how it looked?
5. Are there assertions checking my assumptions?
6. Could someone delete half of this without the other half breaking?

If you answer "no" to any of these, reconsider the design.
