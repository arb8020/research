https://matklad.github.io/2025/11/06/error-codes-for-control-flow.html
https://matklad.github.io/2025/11/09/error-ABI.html

Error Codes for Control Flow
Nov 6, 2025
Two ideas today:

Displaying an error message to the user is a different aspect of error handling than branching based on a specific error condition.
In Zig, error sets are strongly typed error codes, not poor man’s sum types.
In other words, it’s worth thinking about diagnostic reporting and error handling (in the literal sense) separately. There are generally two destinations for any error. An error can be bubbled to an isolation boundary and presented to the operator (for example, as an HTTP 500 message, or stderr output). Alternatively, an error can be handled by taking an appropriate recovery action.

For the first case (reporting), often it is sufficient that an error is an interface that knows how to present itself. The catch is that the presentation interface isn’t fixed: HTML output is different from terminal output. If you know the ultimate destination up front, it usually is simpler to render the error immediately. Otherwise, an error can be made a structured product type to allow (require) the user full control over the presentation (localization of error messages is a good intuition pump).

If you need to branch on error to handle it, you generally need a sum type. Curiously though, there’s going to be a finite number of branches up the stack across all call-sites, so, just like a lean reporting type might contain only the final presentation, a lean handling type might be just an enumeration of all different code-paths — an error code.

As usual, Zig’s design is (thought) provocative. The language handles the “handling” part, leaving almost the entirety of reporting to the user. Zig uses type system to fix problems with error codes, mostly keeping runtime semantics as is.

In C, error codes are in-band, and it’s easy to confuse a valid result with an error code (e.g. doing kill(-1) by accident). Zig uses type-checked error unions:
ReadError!usize
which require explicit unpacking with catch. Error codes are easy to ignore by mistake, but, because the compiler knows which values are errors, Zig requires a special form for ignoring an error:
catch {}

As a nice touch, while Zig requires explicit discards for all unused values, discarding non-error value requires a different syntax:

pub fn main() void {
    _ = can_fail();
    // ^ error: error union is discarded
    can_fail() catch {};
    // ^ error: incompatible types: 'u32' and 'void'
    _ = can_fail() catch {};
    // Works.
}
fn can_fail() !u32 {
    return error.Nope;
}
This protects from a common error when initially a result of an infallible function is ignored, but then the function grows a failing path, and the error gets silently ignored. That’s the I power letter!

As an aside, I used to be unsure whether its best to annotate specific APIs with #[must_use] or do the opposite and, Swift-style, require all return values to be used. My worry was that adding a lot of trivial discards will drown load-bearing discards in the noise. After using Zig, I can confidently say that trivial discards happen rarely and are a non-issue (but it certainly helps to separate value- and error-discards syntactically). This doesn’t mean that retrofitting mandatory value usage into existing languages is a good idea! This drastic of a change usually retroactively invalidates a lot of previously reasonable API design choices.

Zig further leverages the type system to track which errors can be returned by the API:

pub fn readSliceAll(
    r: *Reader,
    buffer: []u8,
) error{ReadFailed, EndOfStream}!void {
    const n = try readSliceShort(r, buffer);
    if (n != buffer.len) return error.EndOfStream;
}
pub fn readSliceShort(
    r: *Reader,
    buffer: []u8,
) error{ReadFailed}!usize {
    // ...
}
The tracking works additively (calling two functions unions the error sets) and subtractively (a function can handle a subset of errors and propagate the rest). Zig also leverages its whole-program compilation model to allow fully inferring the error sets. The closed world model is also what allows assigning unambiguous numeric code to symbolic error constants, which in turn allows a catchall anyerror type.

But the symbolic name is all you get out of the error value. The language doesn’t ship anything first-class for reporting, and diagnostic information is communicated out of band using diagnostic sink pattern:

/// Parses the given slice as ZON.
pub fn fromSlice(
    T: type,
    gpa: Allocator,
    source: [:0]const u8,
    diag: ?*Diagnostics,
) error{ OutOfMemory, ParseZon }!T {
    // ...
}
If the caller wants to handle the error, they pass null sink and switch on the error value. If the caller wants to present the error to the user, they pass in Diagnostics and extract formatted output from that.

Error ABI
Nov 9, 2025
A follow-up on the “strongly typed error codes” article.

One common argument about using algebraic data types for errors is that:

Error information is only filled in when an error occurs,
And errors happen rarely, on the cold path,
Therefore, filling in the diagnostic information is essentially free, a zero cost abstraction.
This argument is not entirely correct. Naively composing errors out of ADTs does pessimize the happy path. Error objects recursively composed out of enums tend to be big, which inflates size_of<Result<T, E>>, which pushes functions throughout the call stack to “return large structs through memory” ABI. Error virality is key here — just a single large error on however rare code path leads to worse code everywhere.

That is the reason why mature error handling libraries hide the error behind a thin pointer, approached pioneered in Rust by failure and deployed across the ecosystem in anyhow. But this requires global allocator, which is also not entirely zero cost.

Choices
How would you even return a result? The default option is to treat -> Result<T, E> as any other user-defined data type: goes to registers if small, goes to the stack memory if large. As described above, this is suboptimal, as it spills small hot values to memory because of large cold errors.

A smarter way to do this is to say that the ABI of -> Result<T, E> is exactly the same as T, except that a single register is reserved for E (this requires the errors to be register-sized). On architectures with status flags, one can even signal a presence of error via, e.g., the overflow flag.

Finally, another option is to say that -> Result<T, E> behaves exactly as -> T ABI-wise, no error affordances whatsoever. Instead, when returning an error, rather than jumping to the return address, we look it up in the side table to find a corresponding error recovery address, and jump to that. Stack unwinding!

The bold claim is that unwinding is the optimal thing to do! I don’t know of a good set of reproducible benchmarks, but I find these two sources believable:

https://joeduffyblog.com/2015/12/19/safe-native-code/#error-model
https://youtu.be/LorcxyJ9zr4?si=HESn1LfHek5Qlfi0
As with async, keep visible programming model and internal implementation details separate! Result<T, E> can be implemented via stack unwinding, and exceptions can be implemented via checking the return value.

Conclusion
Your error ABI probably wants to be special, so the compiler needs to know about errors. If your language is exceptional in supporting flexible user-defined types and control flow, you probably want to special case only in the backend, and otherwise use a plain user-defined type. If your language is at most medium in abstraction capabilities, it probably makes sense to make errors first-class in the surface semantics as well.
