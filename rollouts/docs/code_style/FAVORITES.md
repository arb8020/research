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

## Error Handling: Composable Results

> "Don't litter core logic with defensive try/except. Use composable Result types for clean control flow and concurrent error collection."

**The problem:**
```python
# BAD - defensive exception spam obscures logic
try:
    result1 = step1(data)
except Step1Error:
    return None

try:
    result2 = step2(result1)
except Step2Error:
    return None
```

**The solution:**
```python
# GOOD - railway-oriented programming
def process_pipeline(data) -> Result[Output, str]:
    return (
        step1(data)
        .and_then(step2)
        .and_then(step3)
    )  # Auto short-circuit, no try/except
```

**Critical: Assertions vs Production Invariants**

**Never use `assert` for production checks** - Python's `-O` flag strips them!

```python
# WRONG - stripped with python -O
assert amount > 0  # DISAPPEARS in production!

# RIGHT - always enforced
if amount <= 0:
    raise ValueError(f"Amount must be positive, got {amount}")
```

**Use `assert` only for:** Development checks, documenting assumptions, catching programmer errors during testing.

**See [ERROR_HANDLING.md](ERROR_HANDLING.md) for:** Railway-oriented programming, concurrent error collection, the diagnostic sink pattern, and when to use exceptions vs Results.

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

## Related Deep Dives

For more on these topics, see:
- **[ERROR_HANDLING.md](ERROR_HANDLING.md)** - Composable Results, railway-oriented programming, concurrent error collection, assertions vs invariants
- **[IMMUTABILITY_AND_FP.md](IMMUTABILITY_AND_FP.md)** - Frozen dataclasses, pure functions, and explicit state boundaries
- **[CLASSES_VS_FUNCTIONAL.md](CLASSES_VS_FUNCTIONAL.md)** - When to use classes vs functions

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
