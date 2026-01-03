https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md

“The rules act like the seat-belt in your car: initially they are perhaps a little uncomfortable, but after a while their use becomes second-nature and not using them becomes unimaginable.” — Gerard J. Holzmann

NASA's Power of Ten — Rules for Developing Safety Critical Code will change the way you code forever. To expand:

Use only very simple, explicit control flow for clarity. Do not use recursion to ensure that all executions that should be bounded are bounded. Use only a minimum of excellent abstractions but only if they make the best sense of the domain. Abstractions are never zero cost. Every abstraction introduces the risk of a leaky abstraction.

Put a limit on everything because, in reality, this is what we expect—everything has a limit. For example, all loops and all queues must have a fixed upper bound to prevent infinite loops or tail latency spikes. This follows the “fail-fast” principle so that violations are detected sooner rather than later. Where a loop cannot terminate (e.g. an event loop), this must be asserted.

Use explicitly-sized types like u32 for everything, avoid architecture-specific usize.

Assertions detect programmer errors. Unlike operating errors, which are expected and which must be handled, assertion failures are unexpected. The only correct way to handle corrupt code is to crash. Assertions downgrade catastrophic correctness bugs into liveness bugs. Assertions are a force multiplier for discovering bugs by fuzzing.

Assert all function arguments and return values, pre/postconditions and invariants. A function must not operate blindly on data it has not checked. The purpose of a function is to increase the probability that a program is correct. Assertions within a function are part of how functions serve this purpose. The assertion density of the code must average a minimum of two assertions per function.

Pair assertions. For every property you want to enforce, try to find at least two different code paths where an assertion can be added. For example, assert validity of data right before writing it to disk, and also immediately after reading from disk.

On occasion, you may use a blatantly true assertion instead of a comment as stronger documentation where the assertion condition is critical and surprising.

Split compound assertions: prefer assert(a); assert(b); over assert(a and b);. The former is simpler to read, and provides more precise information if the condition fails.

Use single-line if to assert an implication: if (a) assert(b).

Assert the relationships of compile-time constants as a sanity check, and also to document and enforce subtle invariants or type sizes. Compile-time assertions are extremely powerful because they are able to check a program's design integrity before the program even executes.

The golden rule of assertions is to assert the positive space that you do expect AND to assert the negative space that you do not expect because where data moves across the valid/invalid boundary between these spaces is where interesting bugs are often found. This is also why tests must test exhaustively, not only with valid data but also with invalid data, and as valid data becomes invalid.

Assertions are a safety net, not a substitute for human understanding. With simulation testing, there is the temptation to trust the fuzzer. But a fuzzer can prove only the presence of bugs, not their absence. Therefore:

Build a precise mental model of the code first,
encode your understanding in the form of assertions,
write the code and comments to explain and justify the mental model to your reviewer,
and use VOPR as the final line of defense, to find bugs in your and reviewer's understanding of code.
All memory must be statically allocated at startup. No memory may be dynamically allocated (or freed and reallocated) after initialization. This avoids unpredictable behavior that can significantly affect performance, and avoids use-after-free. As a second-order effect, it is our experience that this also makes for more efficient, simpler designs that are more performant and easier to maintain and reason about, compared to designs that do not consider all possible memory usage patterns upfront as part of the design.

Declare variables at the smallest possible scope, and minimize the number of variables in scope, to reduce the probability that variables are misused.

Restrict the length of function bodies to reduce the probability of poorly structured code. We enforce a hard limit of 70 lines per function.

Splitting code into functions requires taste. There are many ways to cut a wall of code into chunks of 70 lines, but only a few splits will feel right. Some rules of thumb:

Good function shape is often the inverse of an hourglass: a few parameters, a simple return type, and a lot of meaty logic between the braces.
Centralize control flow. When splitting a large function, try to keep all switch/if statements in the "parent" function, and move non-branchy logic fragments to helper functions. Divide responsibility. All control flow should be handled by one function, the rest shouldn't care about control flow at all. In other words, "push ifs up and fors down".
Similarly, centralize state manipulation. Let the parent function keep all relevant state in local variables, and use helpers to compute what needs to change, rather than applying the change directly. Keep leaf functions pure.
Appreciate, from day one, all compiler warnings at the compiler's strictest setting.

Whenever your program has to interact with external entities, don't do things directly in reaction to external events. Instead, your program should run at its own pace. Not only does this make your program safer by keeping the control flow of your program under your control, it also improves performance for the same reason (you get to batch, instead of context switching on every event). Additionally, this makes it easier to maintain bounds on work done per time period.

Beyond these rules:

Compound conditions that evaluate multiple booleans make it difficult for the reader to verify that all cases are handled. Split compound conditions into simple conditions using nested if/else branches. Split complex else if chains into else { if { } } trees. This makes the branches and cases clear. Again, consider whether a single if does not also need a matching else branch, to ensure that the positive and negative spaces are handled or asserted.

Negations are not easy! State invariants positively. When working with lengths and indexes, this form is easy to get right (and understand):

if (index < length) {
  // The invariant holds.
} else {
  // The invariant doesn't hold.
}
This form is harder, and also goes against the grain of how index would typically be compared to length, for example, in a loop condition:

if (index >= length) {
  // It's not true that the invariant holds.
}
All errors must be handled. An analysis of production failures in distributed data-intensive systems found that the majority of catastrophic failures could have been prevented by simple testing of error handling code.

“Specifically, we found that almost all (92%) of the catastrophic system failures are the result of incorrect handling of non-fatal errors explicitly signaled in software.”

Always motivate, always say why. Never forget to say why. Because if you explain the rationale for a decision, it not only increases the hearer's understanding, and makes them more likely to adhere or comply, but it also shares criteria with them with which to evaluate the decision and its importance.

Explicitly pass options to library functions at the call site, instead of relying on the defaults. For example, write @prefetch(a, .{ .cache = .data, .rw = .read, .locality = 3 }); over @prefetch(a, .{});. This improves readability but most of all avoids latent, potentially catastrophic bugs in case the library ever changes its defaults.


