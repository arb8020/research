My Favorites

  1. Write usage code first (Casey Worst API)
  This is transformative. When I catch myself designing an API "in my head," I stop and
   write the dream usage code. It instantly reveals awkward interfaces. The ETW API
  example is burned into my brain—if Microsoft had written 5 lines of ideal usage code
  first, they'd never have shipped that disaster.

  2. Don't reuse until 2+ examples (Casey Semantic Compression)
  This cured my premature abstraction addiction. The discipline of writing it twice,
  then compressing, means my abstractions actually fit the problem instead of being
  speculative architecture. The phrase "make it usable before reusable" is perfect.

  3. Continuous granularity (Casey Granularity)
  The push_button() → bool_button() example clicks something into place. You're not
  replacing the lower level, you're layering on top. Users can drop down when needed.
  No discontinuities. This is what makes APIs feel "right."

  4. Abstraction = coupling (CodeAesthetic)
  The FileSaver example is so clean. "For every bit of abstraction you add, you've
  added more coupling." When I'm tempted to extract a common base class, I now ask:
  "What am I coupling together? Is it worth it?" Usually the answer is no.

  5. Assertions everywhere, split them (Tiger Style)
  assert(a); assert(b); vs assert(a and b) seems trivial but it's huge. When you hit an
   assertion failure, you know exactly which invariant broke. The "2 per function" rule
   forces you to think about what you're actually assuming.

  6. Push ifs up, fors down (Tiger Style)
  This is the clearest articulation I've seen of good function decomposition. Parent
  keeps control flow, helpers compute. It makes code so much easier to trace. When
  you're debugging, you follow the ifs. When you need to change logic, you edit the
  helpers.

  7. State invariants positively (Tiger Style)
  if (index < length) vs if (index >= length). Seems minor but the cognitive load
  difference is real. The first reads naturally with how you think about bounds. The
  second requires mental translation.

  8. Minimize stateful components (Sean System Design)
  "Stateless services can be automatically repaired." This crystallizes why
  microservices often feel harder than the monolith they replaced—you've added tons of
  stateful components (message queues, event buses, distributed state) where a function
   call used to be.

  9. Single assignment (Carmack SSA)
  The debugger benefit alone justifies this. When you hit a breakpoint, every
  intermediate value is still there. No "which reassignment is this?" confusion. It's
  also self-documenting—each name describes a transformation.

  10. No magic constants (Casey Worst API)
  The "ClientContext = 1" story is infuriating. NO symbolic name? Why make millions of
  developers memorize numeric codes? This one's personal—I've debugged too much code
  with bare 42 sprinkled around.

  ---
  The Meta-Favorite

  Your error handling preference reveals something deeper: you value explicit control 
  flow.

  Notice the pattern in your picks:
  - Tuple returns (explicit)
  - SSA (explicit)
  - Push ifs up (explicit)
  - Assertions (explicit)
  - Function length limits (forces explicitness)

  And your system design interest (state management) is about the same thing—making
  state boundaries explicit.

  This is actually the throughline of all these docs: Make the invisible visible. Make
  implicit assumptions explicit. Make control flow obvious. Don't make people trace
  through abstractions to understand what's happening.

  The one rule to rule them all:
  If someone reading your code has to simulate it mentally to understand what it does,
  you failed. The code should read like instructions to a human, not incantations for a
   compiler.

  ---
  Which Ones Do You Find Yourself Using Most Often?

  I'm curious—in your day-to-day, which of your favorites actually changes how you
  write code? For me, "write usage code first" and "2 examples before abstracting" have
   the biggest behavioral impact. The others I tend to apply during refactoring/review.

> yup i agree. i guess starting something new for me basically looks like write the 
usage code -> write the code that makes it work -> apply the control explicit flow 
refactoring + semantic compression. can you make a claude favorites with the above that
 you just wrote me :) 

