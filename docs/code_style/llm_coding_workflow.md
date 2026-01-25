# LLM Coding Workflow

Tips for working effectively with LLMs on code. Separate from code philosophy—this is about the human-LLM collaboration process.

---

## Sync vs Async Usage

Use **synchronous** collaboration for:
- Surfacing code, discussing directions, modeling problems
- When you need back-and-forth to clarify the approach

Use **asynchronous** for:
- Executing once you've defined the desired behavior and data flow
- Letting the LLM run while you do other work

## Parallel Agents

Git worktrees and branches are enough to get multiple LLM agents running at once without stepping on each other's toes. You can run multiple chat sessions literally in parallel exploring different solution spaces, or solving different problems.

Merge conflicts are also design feedback—if two features conflict, it might reveal how to better structure the code to accommodate both.

## Verification Loops

Create verification loops + logging setups where the LLM can efficiently run a command, jq or grep over the output logs, and figure out the undefined/unexpected behavior.

This usually looks like standalone scripts with heavy side effect logging out to files—you're creating a custom observability layer the LLM can actually parse and reason about, not just "test failed" messages.

## Context Management

Separate the "understand codebase context" task as a different session from the "execute patch" session when working on a difficult problem.

If you can't oneshot it, you're probably closer to the edge of you + the LLM's coding ability, so you might want to be more diligent about context management.

## Code Style Docs

Code style docs help crystallize taste opinions in a way that helps the LLM make better decisions on its own. They also reduce the delta between the working code it generated and the code you would like it to have generated.

Feed these to LLMs while working on a problem.

## Prompting Frame

"Let me know if you have any questions, stop if you get confused or things feel complicated" is a cheap way to catch obvious things and get the LLM to sit in a frame where it defers to you instead of wasting tokens on the wrong stuff.

Note: if you've run an LLM for a long time and it gave you something different from your spec, it was probably because your spec wasn't as good as you thought, and the LLM had to deviate to solve the problem.

## Handoff Documents

Create short handoff documents when you finish a task or run out of context:
- High-level problem spec
- File/line ranges to look at
- Useful testing commands
- Things tried that didn't work

This manages context when you start to run out or avoids polluting a follow-up task with unnecessary old context.

### Examples

```
CONTEXT: Users getting logged out after 5 min instead of 30
FILES: auth/session.py:45-67, config/timeouts.py
TEST: pytest tests/auth/test_timeout.py::test_session_duration
TRIED: Updating SESSION_TIMEOUT didn't work (it's overridden elsewhere)
```

```
CONTEXT: need to support digital ocean as a GPU provider
FILES: providers/base.py, providers/runpod.py
TESTS: none yet, check runpod provider for any integration tests
```
