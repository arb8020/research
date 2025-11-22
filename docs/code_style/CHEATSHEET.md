# Code Style Cheatsheet

## Core Principles (Always Apply)

- **Explicit > Clever**: Make control flow obvious. If someone has to think hard to understand it, simplify it. *(Tiger Style)* — "Use only very simple, explicit control flow for clarity."
- **Write usage code first**: Before implementing, write how you WANT to call it. This reveals the right API. *(Casey Worst API)* — "Always write the usage code first... this is the only way to get a nice, clean perspective on how the API would work if it had no constraints."
- **Single assignment**: Don't reassign variables. Create new names for transformed values (helps debugging). *(Carmack SSA)* — "You should strive to never reassign or update a variable outside of true iterative calculations in loops."
- **Assertions everywhere**: Assert preconditions, postconditions, and invariants. 2+ assertions per function. *(Tiger Style)* — "The assertion density of the code must average a minimum of two assertions per function."
- **Split compound assertions**: Use `assert(a); assert(b);` not `assert(a and b)` for clearer failure messages. *(Tiger Style)* — "Split compound assertions: prefer assert(a); assert(b); over assert(a and b)."
- **State invariants positively**: Write `if (index < length)` not `if (index >= length)` (easier to reason about). *(Tiger Style)* — "Negations are not easy! State invariants positively."
- **Abstraction = coupling**: Every abstraction couples things together. Only abstract when benefit > coupling cost. *(CodeAesthetic)* — "I consider coupling to be an equal and opposite reaction of abstraction."
- **A little duplication < over-coupling**: Better to repeat a few lines than force unrelated things together. *(CodeAesthetic)* — "I think that a little code repetition brings less pain when changing code than over coupling."
- **Put limits on everything**: All loops, queues, buffers must have upper bounds. No unbounded growth. *(Tiger Style)* — "Put a limit on everything because, in reality, this is what we expect—everything has a limit."
- **Always say why**: Every decision needs a comment explaining the rationale, not just what. *(Tiger Style)* — "Always motivate, always say why... it increases the hearer's understanding, and makes them more likely to adhere or comply."

## Design Method (How to Build)

- **Don't reuse until you have 2+ examples**: Write it specific first. Compress only after seeing the pattern twice. *(Casey Semantic Compression)* — "Like a good compressor, I don't reuse anything until I have at least two instances of it occurring."
- **Make code usable before reusable**: Get one version working. Then make it general when you need it elsewhere. *(Casey Semantic Compression)* — "My mantra is, 'make your code usable before you try to make it reusable'."
- **Write specific → compress**: Start with copy-paste. When you see repetition, extract it. Don't prematurely abstract. *(Casey Semantic Compression)* — "I always begin by just typing out exactly what I want to happen in each specific case... Then, when I find myself doing the same thing a second time somewhere else, that is when I pull out the reusable portion."
- **Keep all granularity levels**: When you create a high-level function, keep the low-level ones accessible too. *(Casey Granularity)* — "It is always important to avoid granularity discontinuities... never supply a higher-level function that can't be trivially replaced by a few lower-level functions."
- **Each level should work standalone**: Don't force users to jump from fine-grained to coarse-grained with nothing in between. *(Casey Granularity)* — "If you wanted a highlightable button that doesn't work with a boolean at all, you either have to go all the way down to the lowest level of granularity... or introduce a temporary boolean everywhere in a very inconvenient way."
- **Never delete lower-level functions**: When you bundle things, keep the unbundled versions available. *(Casey Granularity)* — "So as long as you just don't delete the smaller pieces as you build bigger ones, you automatically end up avoiding granularity discontinuities."
- **Objects emerge from compression**: Don't design classes upfront. Let them arise naturally from grouping related functions. *(Casey Semantic Compression)* — "This is the correct way to give birth to 'objects'... if you just forget all that, and write simple code, you can always create your objects after the fact."
- **Restrict function length to 70 lines**: Forces you to think about proper decomposition. *(Tiger Style)* — "Restrict the length of function bodies to reduce the probability of poorly structured code. We enforce a hard limit of 70 lines per function."
- **Push ifs up, fors down**: Keep control flow in parent functions. Push non-branching logic to helpers. *(Tiger Style)* — "Centralize control flow... try to keep all switch/if statements in the 'parent' function, and move non-branchy logic fragments to helper functions."
- **Keep leaf functions pure**: Helpers should compute, not mutate. Let parent manage state changes. *(Tiger Style)* — "Centralize state manipulation. Let the parent function keep all relevant state in local variables, and use helpers to compute what needs to change."

## API Design (When Building for Others)

- **Provide redundancy**: Multiple ways to do the same thing (e.g., pass Matrix OR Quaternion). *(Casey Code Reuse)* — "Redundancy is just you know in its most basic form is something like this I wanted to pass a 3X3 Matrix before and now I want to pass a quan so the API gives me two calls."
- **Avoid coupling**: Function A shouldn't force you to call function B. Keep things independent. *(Casey Code Reuse)* — "Coupling is when you have one thing and if you do that thing in the API you're required to do some other thing."
- **No retained state**: Don't make users mirror your internal state. Use immediate mode when possible. *(Casey Code Reuse)* — "Retention is just hey if I have stuff that is you know data that I kind of own... but the API forces me to announce that data to it and it keeps a copy."
- **Caller controls flow**: Don't force callbacks unless absolutely necessary. Let caller orchestrate. *(Casey Code Reuse)* — "Flow control is just you know a measure of flow control anyways it's like who is calling who... if you can get away with just always having the game in control and it calls the app and it returns to you well that's always simpler."
- **Transparent data > opaque handles**: Let users see and modify structs directly when safe. *(Casey Code Reuse)* — "Any data which doesn't have a reason for being opaque should be transparent... you always have the choice as a developer of not touching the data structures of the API."
- **No magic constants**: Always `#define` or `enum` your constants. Never bare numbers (except 0, 1). *(Casey Worst API)* — "Microsoft didn't ever give them symbolic names. So you're just supposed to read the documentation and remember that 1 means the timestamps come from QueryPerformanceCounter."
- **Explicit parameters at call site**: Pass `{.cache = data, .rw = read}` not `{}` to avoid depending on defaults. *(Tiger Style)* — "Explicitly pass options to library functions at the call site, instead of relying on the defaults... avoids latent, potentially catastrophic bugs in case the library ever changes its defaults."
- **Granularity, redundancy, no coupling**: The three key properties of good reusable APIs. *(Casey Code Reuse)* — "The five things are in order uh granularity... there's redundancy... there's coupling... there's retention... and finally there's flow control."

## Error Handling (Python)

- **Return tuples for errors**: `(result | None, error | None)` for single errors. `list[str]` for multiple. *(My Notes)* — "Internal code always uses tuple returns... Returns (result, error). Error is None on success."
- **Explicit over try/except**: Use `result, err = do_thing(); if err: return None, err` pattern. *(My Notes)* — "try/except obscures control flow. When you see a try block, it's unclear: Is the exception case rare (1 in 1000) or common (1 in 2)?"
- **Try/except only at boundaries**: Use when calling external libs (file I/O, network, third-party code). *(My Notes)* — "Only use try/except when unavoidable: Wrapping external library calls - they raise, you can't change that."
- **Wrap external exceptions into tuples**: Catch exceptions at boundary, return tuple to caller. *(My Notes)* — "Use try/except at boundary, but return tuple: try: content = path.read_text(); return content, None except OSError as e: return None, f'Failed to read {path}: {e}'"
- **Assertions for programmer errors**: Use `assert` for things that should never happen if your code is correct. *(My Notes + Tiger Style)* — "Assertions detect programmer errors. Unlike operating errors, which are expected and which must be handled, assertion failures are unexpected."
- **Provide explicit variants**: Offer `get_user()`, `get_user_or_none()`, `get_user_or_create()` so callers choose. *(My Notes + Casey Code Reuse)* — "Provide explicit options... Now callers choose clean control flow: user = get_user_or_create(user_id) // No try/except needed."
- **Don't force callers into try/except**: If "not found" is common, don't make it an exception. *(My Notes)* — "Forces callers into try/except for normal 'not found' case... Caller must use try/except for normal operation."

## Python/ML Specific

- **Use shape suffixes for tensors**: `attention_scores_BNHTS` where B=batch, N=heads, etc. Document dimensions once. *(Shazeer Shape Suffixes)* — "When known, the name of a tensor should end in a dimension-suffix composed of those letters, e.g. input_token_id_BL for a two-dimensional tensor."
- **Explicitly-sized integers**: Use `int32`, `int64`, not `int`. Be explicit about sizes. *(Tiger Style)* — "Use explicitly-sized types like u32 for everything, avoid architecture-specific usize."
- **Dataclass configs with hierarchy**: Nested dataclasses for related settings. Serialize to JSON for reproducibility. *(Experiment Config)* — "Pythonic + Hierarchical + Serializable... Nested dataclasses organize related settings."
- **Save exact config with outputs**: Every experiment run should save its config alongside results. *(Experiment Config)* — "Save exact config used: config.save(output_dir / 'config.json')"
- **Name experiments with lineage**: `02_high_lr_03.py` (from exp 2, trying high LR, becomes exp 3). *(Experiment Config)* — "Following @OnurBerkTore: <#>_<active_change>_<#+1>.py ... This tracks experiment lineage in the filename itself."
- **Fork + sockets over multiprocessing**: Use `os.fork()` and `socket.socketpair()` for simpler IPC semantics. *(Heinrich Multiprocessing)* — "Python multiprocessing is overused garbage. You spend all your time serializing and deserializing pickled stuff."
- **Use memfd for shared memory**: `os.memfd_create()` with `mmap()` for shared data between processes. *(Heinrich Multiprocessing)* — "fd = os.memfd_create(name); os.ftruncate(fd, size) ... The kernel refcounts the fd like a file."
- **Debug configs are mandatory**: Every training codebase needs a fast debug config for quick iteration. *(Experiment Config)* — "when testing a new training codebase if I don't see a debug config that allow me to do a short run on limited hardware then I start to question the ability of the dev to maintain the repo."

## System Design

- **Minimize stateful components**: Stateless services are easier to reason about, restart, and scale. *(Sean System Design)* — "You should try and minimize the amount of stateful components in any system... A stateful service can't be automatically repaired."
- **One service owns each table**: Don't have 5 services all writing to the same database table. *(Sean System Design)* — "Avoid having five different services all write to the same table. Instead, have four of them send API requests (or emit events) to the first service."
- **Query the database**: Use JOINs and let DB do the work. Don't fetch and stitch in application. *(Sean System Design)* — "When querying the database, query the database. It's almost always more efficient to get the database to do the work than to do it yourself."
- **Read from replicas**: Send read queries to replicas, writes to primary. Tolerate replication lag. *(Sean System Design)* — "Send as many read queries as you can to database replicas... The more you can avoid reading from the write node, the better."
- **Fast operations vs slow operations**: Split user-facing (fast) from background work (slow). Use job queues. *(Sean System Design)* — "A service has to do some things fast... But a service has to do other things that are slow... The general pattern is splitting out the minimum amount of work needed to do something useful for the user."
- **Cache sparingly**: Only cache after you've exhausted optimization. Caches add state and complexity. *(Sean System Design)* — "The typical pattern is that junior engineers learn about caching and want to cache everything, while senior engineers want to cache as little as possible... A cache is a source of state."
- **Events for high-volume, low-urgency**: Use event bus (Kafka) when many consumers need data, not time-sensitive. *(Sean System Design)* — "Events are good for when the code sending the event doesn't necessarily care what the consumers do with the event, or when the events are high-volume and not particularly time-sensitive."
- **Push vs pull**: Pull (polling) is simpler. Push (events) is more efficient. Choose based on scale. *(Sean System Design)* — "The simplest is to pull... The alternative is to push... For data that doesn't change much, it's much easier to make a hundred HTTP requests whenever the data changes."
- **Focus on hot paths**: Identify critical code (most traffic, most important). Design those carefully. *(Sean System Design)* — "When you're designing a system, there are lots of different ways users can interact with it... The trick is to mainly focus on the 'hot paths'."
- **Log aggressively on unhappy paths**: Every error condition should log WHY it happened. *(Sean System Design)* — "One thing I've learned from my most paranoid colleagues is to log aggressively during unhappy paths... you should log out the condition that was hit."
- **Watch p95/p99, not averages**: Your slowest requests are from your biggest users. *(Sean System Design)* — "For user-facing metrics like time per-request, you also need to watch the p95 and p99... Even one or two very slow requests are scary, because they're disproportionately from your largest and most important users."
- **Fail open vs fail closed**: Rate limiting fails open. Auth fails closed. Choose per feature. *(Sean System Design)* — "In my view, a rate limiting system should almost always fail open... However, auth should (obviously) always fail closed."
- **Circuit breakers for retries**: Don't retry forever. Back off if service is struggling. *(Sean System Design)* — "If you can, put high-volume API calls inside a 'circuit breaker': if you get too many 5xx responses in a row, stop sending requests for a while to let the service recover."
- **Boring technology**: Use well-tested components. Complex systems should start simple and earn complexity. *(Sean System Design)* — "Good system design is not about clever tricks, it's about knowing how to use boring, well-tested components in the right place."

## Frontend (React)

- **UIs are thin wrappers over data**: Avoid local state unless it's truly independent of business logic. *(Aiden React Patterns)* — "UIs are a thin wrapper over data. You should avoid using local state (like useState) unless you have to, and it's independent of the business logic."
- **Flatten UI state into calculations**: If it can be computed, compute it. Don't store it. *(Aiden React Patterns)* — "Even then, consider if you can flatten the UI state into a basic calculation. useState is only necessary if it's truly reactive."
- **useState only for truly reactive state**: If it's not triggering re-renders based on events, don't use state. *(Aiden React Patterns)* — "useState is only necessary if it's truly reactive."
- **Create components for nested conditionals**: Extract complex if/else trees into separate components. *(Aiden React Patterns)* — "Choose to create a new component abstraction when you're nesting conditional logic, or top level if/else statements."
- **Avoid dependent logic in useEffect**: Make logic explicit rather than relying on reactive behavior. *(Aiden React Patterns)* — "Avoid putting dependent logic in useEffects - it causes misdirection of what the logic is doing. Choose to explicitly define logic rather than depend on implicit reactive behavior."
- **Comment setTimeout usage**: They're usually hacks. Explain why it's necessary. *(Aiden React Patterns)* — "setTimeouts are flaky and usually a hack. Provide a comment on why."
- **Design in progressive layers**: Start simple (core feature), reveal advanced features as users need them. *(Ryolu Design Layers)* — "The best software doesn't overwhelm you on day one or abandon you when you get sophisticated. it grows with you through carefully designed 'layers'."
- **Smooth transitions between layers**: Don't make users "hit a wall" when they need more power. *(Ryolu Design Layers)* — "The real challenge isn't designing each layer, it's designing the transitions between them. that moment when someone outgrows the defaults shouldn't feel like hitting a wall."
- **Choose distinctive typography**: Avoid Inter, Roboto, Arial. Pick fonts with character. *(Frontend Skill)* — "Avoid generic fonts like Arial and Inter; opt instead for distinctive choices that elevate the frontend's aesthetics."
- **Commit to aesthetic direction**: Pick a strong theme (minimal, brutal, retro, etc.) and execute fully. *(Frontend Skill)* — "Choose a clear conceptual direction and execute it with precision. Bold maximalism and refined minimalism both work – the key is intentionality."
- **Motion for high-impact moments**: One great page-load animation > scattered micro-interactions. *(Frontend Skill)* — "Focus on high-impact moments: one well-orchestrated page load with staggered reveals creates more delight than scattered micro-interactions."
- **Backgrounds create atmosphere**: Don't default to solid colors. Add texture, gradients, depth. *(Frontend Skill)* — "Create atmosphere and depth rather than defaulting to solid colors. Add contextual effects and textures that match the overall aesthetic."

## When Working in Teams

- **Introduce patterns incrementally**: Don't rewrite everything. Apply to new code first. *(General Practice)*
- **Compress in your own code**: Lead by example. Let others see the benefits. *(Casey Semantic Compression)*
- **Discuss tradeoffs explicitly**: "This adds coupling but saves us X" is better than "trust me." *(CodeAesthetic + Tiger Style)*
- **Document the why**: Leave comments explaining the reasoning, especially when bucking conventions. *(Tiger Style)* — "Always motivate, always say why."
- **Adapt to team conventions**: If team uses exceptions, use tuple returns in your modules, exceptions at boundaries. *(My Notes + Pragmatism)*

## Context-Dependent Guidelines

**Safety-critical code** (databases, embedded systems, financial): *(Tiger Style)*
- Follow Tiger Style strictly (static allocation, 70-line max, assertion density) — "All memory must be statically allocated at startup."
- No dynamic behavior after initialization
- All memory allocated at startup
- Fixed upper bounds on everything

**Research/prototype code** (ML experiments, one-off scripts): *(Adapted Tiger Style)*
- Tiger Style principles but relaxed constraints
- Dynamic allocation is fine (batches, models vary)
- Focus on explicitness and debuggability
- Keep: assertions for programmer errors, centralized control flow, 70-line limit
- Relax: static allocation (impossible for dynamic batches), recursion limits (sometimes needed)

**Library/framework code** (for others to use): *(Casey Code Reuse)*
- Casey's full API design rules — "The five things are... granularity... redundancy... coupling... retention... flow control."
- Multiple granularity levels
- Extensive redundancy
- Zero coupling between independent features

**Application code** (internal services, tools): *(Sean System Design)*
- Sean's boring technology — "Good system design is not about clever tricks."
- Minimal abstraction
- Optimize for maintainability over flexibility

### Tiger Style Context

Tiger Style is from TigerBeetle (financial database). Some rules are safety-critical specific:

**Apply to all code:**
- ✅ Assertions for programmer errors (not production invariants)
- ✅ All errors must be handled
- ✅ Centralize control flow (push ifs up, fors down)
- ✅ 70-line function limit
- ✅ Explicit control flow, no magic
- ✅ State invariants positively

**Only for safety-critical systems:**
- ⚠️ Static memory allocation at startup (impossible for ML with dynamic batches)
- ⚠️ No recursion (sometimes needed for tree structures, parsers)
- ⚠️ Fixed upper bounds on everything (trades flexibility for safety)

**For ML/research code:** Apply the principles (explicit, testable, bounded) but relax the hard constraints that assume embedded/safety-critical context.

## Red Flags

- **Too many ifs in one place**: Extract to separate functions or data-driven approach. *(Tiger Style)* — "Centralize control flow."
- **Callbacks for simple data transfer**: Should be a direct call that returns data. *(Casey Worst API)* — "ProcessTrace is defined to never return... you have to create an entirely new thread to do nothing but block on ProcessTrace!"
- **Opaque handles for simple data**: Let users see the struct. *(Casey Code Reuse)* — "Any data which doesn't have a reason for being opaque should be transparent."
- **Forced coupling**: "To use A, you must also use B." *(Casey Code Reuse)* — "Coupling is when you have one thing and if you do that thing in the API you're required to do some other thing."
- **Granularity gaps**: Can do X or Z, but not Y in between. *(Casey Granularity)* — "It is always important to avoid granularity discontinuities."
- **Premature abstraction**: "I might need this someday" is not a reason. *(Casey Semantic Compression)* — "I don't reuse anything until I have at least two instances of it occurring."
- **Inheritance for code reuse**: Composition almost always better. *(Casey Semantic Compression)* — "The fallacy of 'object-oriented programming' is exactly that: that code is at all 'object-oriented'. It isn't."
- **State machines for simple toggles**: Overkill for boolean flags. *(Aiden React Patterns - cautionary)*
- **Clever tricks over clarity**: If you're proud of how clever it is, it's probably wrong. *(Tiger Style + General)*

## The Ultimate Test

**Before shipping, ask:** *(Synthesis of all principles)*
1. **Can I explain this to someone in 30 seconds?** *(Tiger Style: explicit control flow)*
2. **If I debug this at 3am, will I understand it?** *(Carmack SSA + Tiger Style assertions)*
3. **If requirements change, what breaks?** *(Casey Granularity + CodeAesthetic coupling)*
4. **Did I write the usage code first and like how it looked?** *(Casey Worst API)*
5. **Are there assertions checking my assumptions?** *(Tiger Style)*
6. **Could someone delete half of this without the other half breaking?** *(Casey Granularity: avoid coupling)*

**If you answer "no" to any of these, reconsider the design.**
