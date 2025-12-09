# React Frontend Patterns - Aiden Bai

## Core Philosophy

UIs are a thin wrapper over data. You should avoid using local state (like useState) unless you have to, and it's independent of the business logic.

Even then, consider if you can flatten the UI state into a basic calculation. useState is only necessary if it's truly reactive.

## State Management

Choose state machines over multiple useStates - it makes the code harder to reason about.

## Component Abstraction

Choose to create a new component abstraction when you're nesting conditional logic, or top level if/else statements. Ternaries are reserved for small, easily readable logic.

## Side Effects

Avoid putting dependent logic in useEffects - it causes misdirection of what the logic is doing. Choose to explicitly define logic rather than depend on implicit reactive behavior.

A huge reason useEffect is a red flag in React is because it allows code like this:
> On A, set B. When B changes, do C

when what you really need is:
> On A, set B and do C

🚩 When you see a useEffect, make sure it's not introducing this problem.

## Timing

setTimeouts are flaky and usually a hack. Provide a comment on why.

## Impact

This doesn't affect if the "code runs" or not, most of the time, but you can introduce subtle bugs that pile up into big issues that aren't obvious until someone goes in and has to spend a lot of time debugging.
