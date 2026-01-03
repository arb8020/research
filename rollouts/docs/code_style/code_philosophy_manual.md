# Code Philosophy Manual

Here's a reasonably clear workflow that we can use to make our LLM coding faster, given that the nature of writing code is different than before. Software has only gotten more malleable, pushing engineer responsibility higher up the abstraction layer. It's more important than ever that we understand the existing code, write the ideal usage code, get it working, do manual testing and crystallize it into tests, then refactor the guts into code that will be easy to reason about for the next person—whenever we make new features, refactor code, and fix bugs.

## What Good Code Looks Like

Understanding code and code review underpins basically all of what follows. We want this to be as easy as possible. So what does code that is easy to review and extend look like? It looks like code that makes the reader's job as easy as possible. We should minimize the amount of working memory that the reader has to use to process a piece of code. Any time the user has to simulate what a function does or remember what something does or means because it's unintuitive, we failed.[[1]](#ref-1)

In practice, this looks like:
- **Obvious control flow** — push ifs up and fors down[[2]](#ref-2)
- **Visible, minimal state** — one thing owns state, many things query it[[3]](#ref-3)
- **Names as compressed documentation** — single assignment, each value has one name[[4]](#ref-4)
- **Types as program state constraint** — illegal states are unrepresentable[[5]](#ref-5)
- **Continuous granularity** — can drop down a level without major rewrites[[6]](#ref-6)
- **Explicit orchestration** — clear owners of state, pure functions for computation[[3]](#ref-3)
- **Parsing at the edges** — external validation at boundaries, happy hotpaths inside[[5]](#ref-5)
- **Assertions document invariants** — split compound assertions, two per function minimum[[7]](#ref-7)
- **Errors crash or compose** — crash loudly for impossible internal states, retry with backoff at external boundaries for transient failures (rate limits, connections), crash for non-retryable errors (bad model ID, invalid config)[[14]](#ref-14)
- **Avoiding overly granular tests** — integration tests over unit tests[[8]](#ref-8)

## The Workflow

Now how do we write this code? Let's think about the space of coding tasks: features, fixes, refactors. These basically all first require understanding the current codebase state. What's the new behavior we want to add/modify, and where should it go? What's the behavior we need to fix? What's the behavior we need to keep the same, but make more clear?

A useful framing here is thinking about "usage code."[[9]](#ref-9) For a bug fix, the "usage code" is a regression test—what's the list of commands/buggy state we can set up to observe the unintended behavior? We start by writing this. For features/behavior changes and refactors, we start by writing the commands the user might enter into their terminal, the behavior they would exhibit in interacting with a GUI, and the expected output or result. This might also mean the literal code that they might write internally to get something done, for internal packages, or for services that consume other services' outputs.

So we start with understanding our problem in the "real life" domain, and then figure out how to translate it into code. This typically involves a lot of grepping/reading code, revising our usage code accordingly, until we have a solid understanding of what the current system is capable of and what we need to add. Oftentimes a feature or bug fix might suggest a refactor before the actual feature/bug fix gets added, but it's frequently more useful to first get the code working, and refactor the guts after. This is because of the next important piece: testing.

## Testing

Testing is very much like eating vegetables or going to the gym. It starts out as boring until you feel the effects of it and then you look forward to it.[[8]](#ref-8) Since LLMs barf out code so quickly and easily, without testing, you may as well have not written any code at all. Anyone can lazily type a feature request into Claude. Your responsibility as the engineer is to make sure it works before marking a PR as ready for review.[[10]](#ref-10)

The easiest way for you to do that is to turn the usage code you wrote into a test. This also explains why we prefer integration tests as much as possible—they test the API surface that actually matters, not implementation details that will change.[[8]](#ref-8) We start with manual testing, to validate that we see the commands or frontend literally produce what we want to see on the computer screen. Note that this should literally be done manually—don't ask your agent to do it for you, it may lie about success. Once you've done this, we crystallize it into the appropriate form of test. For bugs, this looks like a regression test that fails when you undo your patch. For frontends/backend code, this looks like scripts that mimic potential user workflows. The goal is to have the absolute minimal test suite that we really trust.

## Refactoring

Now that the code is confirmed working and tested, our job is to leave the codebase better than we found it. It's now time to look at whatever horrible monstrosity the LLM wrote (ideally this shouldn't happen—if we follow codebase guidelines the LLM tends to pick up on the way we like to write code, or even just getting code style docs into context), and clean it up.

Think of yourself as a compressor.[[11]](#ref-11) Think about how we can go from working code to the most efficient code—efficient here meaning anywhere from performance of the literal processors to future code writing and understanding efficiency. Don't abstract until you've done something twice; otherwise you're guessing at the wrong seams.[[11]](#ref-11) This is where our opinions on code style from above really come into play to make code review and future PRs as easy as possible. And the tests make it easy to make sure that your refactors preserve functionality—ideally your test code shouldn't change (because you already wrote the ideal API/usage code, right?), but the guts will become more clear. Linting rules, type checking, LLM passes with specific context about code style all come into play here.

**Optional local tooling:** If you want automated linting and type checking on commit, copy the [pre-commit config](pre-commit-config.yaml) to your repo root as `.pre-commit-config.yaml` and run `pre-commit install`. This sets up:
- **ruff** — fast Python linter with auto-fix
- **ruff-format** — Python formatter
- **ty** — strict type checker

## Code Review

Finally, the responsibility of the code reviewer. Your job is structural.[[12]](#ref-12) Your job is to surface potential misalignments about the problem statement, ideal usage code, or approach taken. The guts of the code don't matter if we have to rewrite the whole PR. Are we reverse engineering frontend requests when we could have hit an API? Is the usage code unintuitive?

LLMs are enthusiastic juniors who never develop judgment over time.[[12]](#ref-12) About once an hour you'll notice they're doing something suspicious—building a full job queue when you just need a non-blocking request. The reviewer catches these architectural dead ends before they ship.

Ideally the majority of alignment happens in design docs, and we spend most of our time there. The turnaround from "maybe this seems like the right approach" to "prototype it" should be very fast since code is so cheap. And the code reviewer's job should be very easy given the above principles.

It's important also that a human at least skims the code, or asks the important questions to the LLM, and reads the code citations the LLM gives them. It's easy for bad code—or worse, bad ideas—to slip through the cracks without this, though the above guardrails will minimize this risk. The review comment isn't really for the LLM. It's to align the human developer's mental model.[[13]](#ref-13)

---

## References

<a id="ref-1"></a>
**[1] On cognitive load:**
> "If someone reading your code has to simulate it mentally to understand what it does, you failed. The code should read like instructions to a human, not incantations for a compiler."
>
> — [FAVORITES.md](FAVORITES.md)

<a id="ref-2"></a>
**[2] On control flow:**
> "Good function shape is often the inverse of an hourglass: a few parameters, a simple return type, and a lot of meaty logic between the braces. Centralize control flow. When splitting a large function, try to keep all switch/if statements in the 'parent' function, and move non-branchy logic fragments to helper functions... In other words, 'push ifs up and fors down'."
>
> — [tiger_style_safety.md](tiger_style_safety.md)

<a id="ref-3"></a>
**[3] On minimizing state:**
> "You should try and minimize the amount of stateful components in any system... A stateful service can't be automatically repaired. If your database gets a bad entry in it, you have to manually go in and fix it up."
>
> — [sys_design_sean_goedecke.md](sys_design_sean_goedecke.md)

<a id="ref-4"></a>
**[4] On single assignment:**
> "You should strive to never reassign or update a variable outside of true iterative calculations in loops. Having all the intermediate calculations still available is helpful in the debugger, and it avoids problems where you move a block of code and it silently uses a version of the variable that wasn't what it originally had."
>
> — [carmack_python_ssa.md](carmack_python_ssa.md)

<a id="ref-5"></a>
**[5] On types and boundaries:**
> "All memory must be statically allocated at startup... Declare variables at the smallest possible scope, and minimize the number of variables in scope, to reduce the probability that variables are misused."
>
> — [tiger_style_safety.md](tiger_style_safety.md)

<a id="ref-6"></a>
**[6] On granularity:**
> "It is always important to avoid granularity discontinuities... never supply a higher-level function that can't be trivially replaced by a few lower-level functions that do the same thing."
>
> — [casey_muratori_complexity_granularity.md](casey_muratori_complexity_granularity.md)

<a id="ref-7"></a>
**[7] On assertions:**
> "The assertion density of the code must average a minimum of two assertions per function. Split compound assertions: prefer `assert(a); assert(b);` over `assert(a and b);`. The former is simpler to read, and provides more precise information if the condition fails."
>
> — [tiger_style_safety.md](tiger_style_safety.md)

<a id="ref-8"></a>
**[8] On testing:**
> "Integration tests sweet spot according to grug: high level enough test correctness of system, low level enough, with good debugger, easy to see what break... unit tests fine, ok, but break as implementation change and make refactor hard."
>
> — [grugbrain_testing.md](grugbrain_testing.md)

<a id="ref-9"></a>
**[9] On writing usage code first:**
> "Always write the usage code first... this is the only way to get a nice, clean perspective on how the API would work if it had no constraints."
>
> — [casey_muratori_worst_api.md](casey_muratori_worst_api.md)

<a id="ref-10"></a>
**[10] On proving code works:**
> "Your job is to deliver code you have proven to work. Almost anyone can prompt an LLM to generate a thousand-line patch... What's valuable is contributing code that is proven to work."
>
> — [simon_willison_proven_code.md](simon_willison_proven_code.md)

<a id="ref-11"></a>
**[11] On semantic compression:**
> "Like a good compressor, I don't reuse anything until I have at least two instances of it occurring. Many programmers don't understand how important this is, and try to write 'reusable' code right off the bat, but that is probably one of the biggest mistakes you can make. My mantra is, 'make your code usable before you try to make it reusable'."
>
> — [casey_muratori_semantic_compression.md](casey_muratori_semantic_compression.md)

<a id="ref-12"></a>
**[12] On structural code review:**
> "The best code review is structural. It brings in context from parts of the codebase that the diff didn't mention. Ideally, that context makes the diff shorter and more elegant... Working with AI agents is like working with enthusiastic juniors who never develop the judgement over time that a real human would."
>
> — [sean_goedecke_code_review.md](sean_goedecke_code_review.md)

<a id="ref-13"></a>
**[13] On review as alignment:**
> "The review comment isn't really for the LLM—it's to align the human developer's mental model... Humans must collectively maintain a shared vision of what the system should do."
>
> — [ezyang_code_review_alignment.md](ezyang_code_review_alignment.md)

<a id="ref-14"></a>
**[14] On error types:**
> "Assertions detect programmer errors. Unlike operating errors, which are expected and which must be handled, assertion failures are unexpected. The only correct way to handle corrupt code is to crash."
>
> — [tiger_style_safety.md](tiger_style_safety.md)

---

## Further Reading

For deeper dives on specific topics, see the source documents in this directory:

- **API Design**: [casey_muratori_worst_api.md](casey_muratori_worst_api.md), [casey_muratori_complexity_granularity.md](casey_muratori_complexity_granularity.md), [code_reuse_casey_muratori.md](code_reuse_casey_muratori.md)
- **Abstraction & Compression**: [casey_muratori_semantic_compression.md](casey_muratori_semantic_compression.md), [codeaesthetic_abstraction_coupling.md](codeaesthetic_abstraction_coupling.md)
- **Code Structure**: [tiger_style_safety.md](tiger_style_safety.md), [CLASSES_VS_FUNCTIONAL.md](CLASSES_VS_FUNCTIONAL.md), [IMMUTABILITY_AND_FP.md](IMMUTABILITY_AND_FP.md)
- **System Design**: [sys_design_sean_goedecke.md](sys_design_sean_goedecke.md)
- **Testing**: [grugbrain_testing.md](grugbrain_testing.md)
- **LLM Coding**: [sean_goedecke_code_review.md](sean_goedecke_code_review.md), [ezyang_code_review_alignment.md](ezyang_code_review_alignment.md), [simon_willison_proven_code.md](simon_willison_proven_code.md)
- **Observability**: [logging_sucks.md](logging_sucks.md)
- **Patterns Synthesis**: [FAVORITES.md](FAVORITES.md)
