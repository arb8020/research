# Code Philosophy

We have the blessing and curse of software being more malleable than hardware. Since code is easy to write, the important thing an engineer brings to the table is identifying the point in the space of correct programs that is the easiest program to keep working with. We can think of writing new code in two phases: first, getting the software to work the way we want it to, and secondly, getting it to be easy to write new software to work with it and on top of it. Both of these are a programmer's responsibility.

## Understanding and Usage Code

First we must understand what the codebase is already capable of. Where in the existing API can we hook into to accomplish what needs to be done? Where do we have gaps where we might need to expose a more granular part of a process? Where might we need to consume the output of a previous codepath and add a transformation on top of it? 

We first write the ideal code from the perspective of either the user or future codebase maintainers. What will I want to be able to write in the future? How can I express a requirement as a composition of primitives? As we explore a codebase, we might find that we have to bend the "ideal" usage code into something more pragmatic. Oftentimes if we have to do this, we might realize that the underlying code needs to be changed. This should happen as little as possible if we model our problems correctly, but real life systems and business requirements are fluid, so deeper refactors are necessary from time to time.

## Implementation and Testing

As we sharpen the ideal usage code from a vague idea into what the user wants to write, and would be able to write given the current or expected codebase state, we can then move to actually getting that code working. Get it working, manually verify it works. And then we leave tests in our wake, crystallizing the state of the working codebase in a compressed manner. This makes it easier to be confident about future changes to the code—no unintended regressions.

## Cleanup: State is the Enemy

And now that we've got the working state crystallized, we can clean up the guts of the code. The enemy is state. Humans only have about 6-7 working memory chunks, and they come to the codebase with a few of them already used, thinking about the problem they need to solve. So our code should minimize owners of state and make it explicit wherever possible. 

Name things well—the reader should immediately know what something is and does from the name. Use types to constrain the space of what any variable might be at a given time, to both the reader and maintainer's benefit. Move all code that deals with externals to the boundary of our program. Complicated parsing and validation can live away from the clean hotpath—guards at the gates validate external data. And assertions inside document what should always be true.

---

## References

**On good design being invisible:**
> "Good design is self-effacing: bad design is often more impressive than good. I'm always suspicious of impressive-looking systems. If a system has distributed-consensus mechanisms, many different forms of event-driven communication, CQRS, and other clever tricks, I wonder if there's some fundamental bad decision that's being compensated for."
> — Sean Goedecke, *Everything I know about good system design*

**On writing usage code first:**
> "Always write the usage code first... this is the only way to get a nice, clean perspective on how the API would work if it had no constraints."
> — Casey Muratori, *Designing and Evaluating Reusable Components*

**On granularity and avoiding discontinuities:**
> "It is always important to avoid granularity discontinuities... never supply a higher-level function that can't be trivially replaced by a few lower-level functions."
> — Casey Muratori, *Complexity and Granularity*

**On when to abstract:**
> "Like a good compressor, I don't reuse anything until I have at least two instances of it occurring. Many programmers don't understand how important this is, and try to write 'reusable' code right off the bat, but that is probably one of the biggest mistakes you can make. My mantra is, 'make your code usable before you try to make it reusable'."
> — Casey Muratori, *Semantic Compression*

**On integration tests:**
> "Integration tests sweet spot according to grug: high level enough test correctness of system, low level enough, with good debugger, easy to see what break."
> — Grugbrain, *The Grug Brained Developer*

**On minimizing state:**
> "You should try and minimize the amount of stateful components in any system... A stateful service can't be automatically repaired. If your database gets a bad entry in it, you have to manually go in and fix it up."
> — Sean Goedecke, *Everything I know about good system design*

**On cognitive load:**
> "If someone reading your code has to simulate it mentally to understand what it does, you failed. The code should read like instructions to a human, not incantations for a compiler."
> — FAVORITES synthesis of Tiger Style, Carmack, et al.

**On abstraction as coupling:**
> "I consider coupling to be an equal and opposite reaction of abstraction. For every bit of abstraction you add, you've added more coupling."
> — CodeAesthetic, *Abstraction Can Make Your Code Worse*

**On single assignment (SSA style):**
> "You should strive to never reassign or update a variable outside of true iterative calculations in loops. Having all the intermediate calculations still available is helpful in the debugger, and it avoids problems where you move a block of code and it silently uses a version of the variable that wasn't what it originally had."
> — John Carmack

**On function structure and control flow:**
> "Good function shape is often the inverse of an hourglass: a few parameters, a simple return type, and a lot of meaty logic between the braces. Centralize control flow. When splitting a large function, try to keep all switch/if statements in the 'parent' function, and move non-branchy logic fragments to helper functions... In other words, 'push ifs up and fors down'."
> — Tiger Style, TigerBeetle

```python
# Parent function: has all the control flow (ifs)
def process_request(request):
    if request.needs_auth:
        user = authenticate(request)
        if not user:
            return error_response("Unauthorized")
    else:
        user = None

    if request.type == "query":
        result = handle_query(request.params)
    else:
        result = handle_command(request.params)

    return success_response(result)

# Helper functions: pure computation, no branching
def handle_query(params):
    data = fetch_data(params)
    return {"results": data, "count": len(data)}

def handle_command(params):
    return {"status": "ok", "applied": params}
```

**On assertions:**
> "The assertion density of the code must average a minimum of two assertions per function. Split compound assertions: prefer `assert(a); assert(b);` over `assert(a and b);`. The former is simpler to read, and provides more precise information if the condition fails."
> — Tiger Style, TigerBeetle

**On proving code works:**
> "Your job is to deliver code you have proven to work. Almost anyone can prompt an LLM to generate a thousand-line patch... What's valuable is contributing code that is proven to work."
> — Simon Willison

**On code review in the LLM era:**
> "The review comment isn't really for the LLM—it's to align the human developer's mental model... Humans must collectively maintain a shared vision of what the system should do."
> — Edward Z. Yang, *Code Review as Human Alignment in the Era of LLMs*

**On LLMs as enthusiastic juniors:**
> "Working with AI agents is like working with enthusiastic juniors who never develop the judgement over time that a real human would... About once an hour I notice that the agent is doing something suspicious, and when I dig deeper I'm able to set it on the right track and save hours of wasted effort."
> — Sean Goedecke, *If you are good at code review, you will be good at using AI agents*

**On structural vs nitpicky review:**
> "The best code review is structural. It brings in context from parts of the codebase that the diff didn't mention. Ideally, that context makes the diff shorter and more elegant... If you're a nitpicky code reviewer, I think you will struggle to use AI tooling effectively. You'll be forever tweaking individual lines of code, asking for a .reduce instead of a .map.filter, bikeshedding function names, and so on. At the same time, you'll miss the opportunity to guide the AI away from architectural dead ends."
> — Sean Goedecke, *If you are good at code review, you will be good at using AI agents*

---

## Additional References (to incorporate)

### On Writing Usage Code First (with example)

> "Before we take a look at the actual Event Tracing for Windows API, I want to walk the walk here and do exactly what I said to do: write the usage code first. Whenever you evaluate an API, or create a new one, you must always, always, ALWAYS start by writing some code as if you were a user trying to do the thing that the API is supposed to do. This is the only way to get a nice, clean perspective on how the API would work if it had no constraints on it whatsoever."
> — Casey Muratori, *The Worst API Ever Made*

```python
# Write THIS first — what do you WANT to write?
config = Config()
config.training.learning_rate = 1e-3
train(config)

# Not this — don't start with the implementation
config = ConfigBuilder() \
    .with_training_params(TrainingParams.builder()
        .learning_rate(1e-3)
        .build())
    .build()
```

### On Continuous Granularity (with example)

> "Adding bool_button(), instead of modifying push_button() was crucial in that it created a third level of granularity. If you happen to want a toggle boolean, you can call bool_button(). If you want a highlightable button, but don't want a toggle boolean, you can call push_button(). And if you want something else altogether, you can still call the UI system directly."
> — Casey Muratori, *Complexity and Granularity*

The pattern: each level *uses* the lower level. Don't delete lower functions when you add higher ones.

```python
# Low level - full control
draw_button(x, y, width, height, "Click Me", button_color, text_color)

# Mid level - common case
if layout.push_button("Click Me"):
    do_thing()

# High level - even simpler (wraps push_button)
layout.toggle_button("Enabled", enabled_flag)
```

### On Passing Data vs Managing Shadow State

> "The idea behind immediate mode is I've gone through the code, I've figured out exactly what I want to do, I just want to call the API with it right there. I don't want to have to worry about having set up a retained mode structure previously... What typically happens is I've got something like 'is the user pressing the X button' and what I have to do is write the diff for every part of my game where I diff their retained mode version with what I actually know to be going on."
> — Casey Muratori, *Designing and Evaluating Reusable Components*

The problem: some APIs force you to maintain a "shadow copy" of your state inside their system. Every time your state changes, you have to figure out the diff and update their copy. This is error-prone and verbose.

```python
# BAD - you have to sync your state with theirs
# "If X button pressed, make sure their system knows about the joint"
# "If X button released, make sure their system forgets the joint"
if pressing_x_button:
    if not hook_line_exists:
        create_hook_line(rocket, pole)
else:
    if hook_line_exists:
        delete_hook_line()
simulate()

# GOOD - just pass what you want, every frame
# No shadow state to manage, no diffs to compute
if pressing_x_button:
    do_joint(rocket, pole)
simulate()
```

The principle: prefer APIs where you pass data in and get data out, rather than APIs where you have to keep their internal state in sync with yours.

### On Classes vs Functions

> "If you can make it a pure function, make it a pure function. Only use a class when you have legitimate persistent state."

**The test:**
- Does it own a resource (socket, process, file handle)? → Class
- Does it need cleanup or lifecycle management? → Class
- Is it just data that doesn't change? → Frozen dataclass
- Is it computation or transformation? → Pure function
- Is it orchestrating other things? → Pure function that calls methods on objects

**Configuration should be frozen dataclasses, not mutable classes:**

```python
# BEFORE: Mutable class with setters
class Config:
    def __init__(self, lr, batch_size):
        self.lr = lr
        self.batch_size = batch_size

    def set_lr(self, lr):
        self.lr = lr  # Hidden mutation — who changed this? when?

# AFTER: Frozen dataclass — immutable, serializable, hashable
@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    batch_size: int
    num_epochs: int

    # To "change" config, create a new one:
    # new_config = replace(old_config, learning_rate=0.001)
```

**Iteration state can be a frozen dataclass + pure functions, not a mutable class:**

```python
# BEFORE: Mutable class with hidden state
@dataclass
class DataBuffer:
    prompts: list[str]
    epoch_id: int = 0
    sample_offset: int = 0
    seed: int = 42

    def get_prompts(self, n: int) -> list[str]:
        # Mutates self.epoch_id, self.sample_offset internally
        # Caller can't see what changed
        ...

# Usage: state changes are hidden
buffer = DataBuffer(prompts=prompts)
batch = buffer.get_prompts(32)  # What changed inside buffer?

# AFTER: Frozen state + pure function
@dataclass(frozen=True)
class BufferState:
    epoch_id: int = 0
    sample_offset: int = 0
    seed: int = 42

def get_samples(samples, state, n) -> tuple[list[Sample], BufferState]:
    # Pure function: returns new state, doesn't mutate
    ...
    return batch, new_state

# Usage: state changes are explicit
state = BufferState(seed=42)
batch, state = get_samples(samples, state, n=32)  # New state returned
```

**Classes are for resources that need lifecycle management:**

```python
# Worker owns a socket and child process — needs cleanup
@dataclass
class Worker:
    pid: int
    _sock: socket.socket

    def send(self, msg: Any) -> None:
        # Writes to socket (stateful I/O)
        ...

    def recv(self, max_size: int) -> Any:
        # Reads from socket (stateful I/O)
        ...

    def wait(self) -> None:
        # Cleanup: wait for child process to exit
        os.waitpid(self.pid, 0)
```

**Functions orchestrate stateful objects:**

```python
# BEFORE: God object that does everything
class Trainer:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.dataset = None
        self.metrics = []
        self.config = {}

    def do_everything(self):
        # 500 lines mixing state management with computation
        ...

# AFTER: Separate concerns — objects for state, functions for orchestration
backend = PyTorchTrainingBackend(model, optimizer, device)  # Owns model state
samples = load_samples_from_jsonl("data.jsonl")             # Just data
state = BufferState(seed=42)                                # Iteration state

def run_sft_training(config, backend, samples, state) -> tuple[TrainingResult, BufferState]:
    """Pure function that orchestrates stateful objects."""
    metrics_history = []
    for step in range(config.num_steps):
        batch, state = get_samples(samples, state, n=config.batch_size)
        metrics, err = sft_training_step(backend, batch, config)
        if err:
            return None, state
        metrics_history.append(metrics)
    return TrainingResult(metrics=metrics_history), state
```

### On Stating Invariants Positively

> "Negations are not easy! State invariants positively."
> — Tiger Style, TigerBeetle

```python
# GOOD - reads naturally, positive case first
if index < length:
    # Valid: we can access index
    process(items[index])
else:
    # Invalid: index out of bounds
    handle_error()

# HARDER TO READ - negation
if index >= length:
    # It's NOT true that the invariant holds
    handle_error()
```

### On Error Handling Philosophy

**Three categories:**
1. **Assertions** - for programmer errors / invariants that should never be violated
2. **Explicit error returns** - for expected failures (user input, network, external systems)
3. **Crash loud** - for infrastructure issues / corruption

```python
# Assertions for invariants (dev-time checks)
assert batch_size > 0, "batch_size must be positive"
assert embedding_dim % num_heads == 0, "must be evenly divisible"

# Explicit error returns for expected failures
def process_payment(card) -> tuple[Payment | None, str | None]:
    if card.expired:
        return None, "Card expired"
    return charge(card), None

# Crash loud for corruption
if checksum != expected:
    raise CorruptionError("Data integrity violation - cannot continue")
```

### On Parse at the Boundary, Assert Internally

> "Move all code that deals with externals to the boundary of our program. Complicated parsing and validation can live away from the clean hotpath—guards at the gates validate external data. And assertions inside document what should always be true."

At the boundary, you validate and reject bad data. Internally, you use assertions to document what *should* already be true — they catch programmer errors (someone bypassed the boundary), not user errors.

```python
# At the boundary: validate everything, reject bad input
def handle_request(raw_request: dict) -> Response:
    user_id = parse_user_id(raw_request.get("user_id"))  # raises if invalid
    amount = parse_positive_int(raw_request.get("amount"))  # raises if invalid
    return process_payment(user_id, amount)

# Internal function: assertions document the contract
def process_payment(user_id: UserId, amount: PositiveInt) -> Payment:
    assert user_id is not None  # If this fails, caller has a bug
    assert amount > 0           # Types should guarantee this, assert confirms it
    # Just do the work
    ...
```

The assertions aren't doing the validation work (the boundary already did that). They document the contract and give you a clear crash if someone violates it, instead of weird downstream behavior.

### On Testing Philosophy (expanded)

> "grug prefer write most tests after prototype phase, when code has begun firm up... also, test shaman often talk unit test very much, but grug not find so useful. grug experience that ideal tests are not unit test or either end-to-end test, but in-between test."
> — Grugbrain, *The Grug Brained Developer*

**The testing pyramid we actually want:**
1. **Unit tests** - useful at start, but break as implementation changes. Don't get attached.
2. **Integration tests** - sweet spot. High enough to test correctness, low enough to debug.
3. **End-to-end tests** - small, curated suite for critical paths. "Kept working religiously on pain of clubbing."

**One exception:** "When bug found, grug always try first reproduce bug with regression test then fix bug."

### On Abstraction Tradeoffs (when it IS worth it)

> "There are two cases where it would make me decide it was worth it. One would be if we added more save options. If we had three or more we might want to extract... The other case would be if we needed our program to defer or repeat saving at a different point in the program... In both cases, it becomes worth it when we want to separate the decision of which saver we want from the time we actually want to save."
> — CodeAesthetic, *Abstraction Can Make Your Code Worse*

**Abstract when:**
- You have 3+ instances (not 2)
- You need to separate "what" from "when"
- The coupling cost is less than the duplication cost

**Don't abstract when:**
- It's just "assigning a variable" - not worth the coupling
- You'd be coupling unrelated things to the same input
- The "benefit" is removing one duplicate line

### On Single Assignment (with example)

```python
# GOOD - each transformation has a name, all visible in debugger
raw_data = fetch_from_db()
filtered_data = [d for d in raw_data if d.valid]
sorted_data = sorted(filtered_data, key=lambda d: d.timestamp)
result = transform_to_response(sorted_data)

# BAD - reusing same name, can't inspect intermediate states
data = fetch_from_db()
data = [d for d in data if d.valid]  # Which 'data' is this?
data = sorted(data, key=lambda d: d.timestamp)
result = transform_to_response(data)
```

### On Code Review (structural, not nitpicky)

> "The best code review is structural. It brings in context from parts of the codebase that the diff didn't mention. Ideally, that context makes the diff shorter and more elegant: instead of building out a new system for operation X, we can reuse a system that already exists."
> — Sean Goedecke, *If you are good at code review, you will be good at using AI agents*

**Good review asks:** "Is this even the right place for this code?"
**Bad review asks:** "Should this be a .reduce instead of .map.filter?"
