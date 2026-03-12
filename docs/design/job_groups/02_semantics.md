# Job Groups: Execution Semantics

## Core lifecycle

Given a `JobGroupSpec` with tasks and a `primary` task:

1. Start all tasks (best-effort, but fail loud if any task fails to start)
2. Wait for `primary` to complete
3. Sleep `termination_delay_seconds` (default 0)
4. Stop all *non-primary* tasks
5. Return primary exit code

## Required invariants (MVP)

- Primary must be a `job` (finite) task
- Task names are unique

## Failure semantics (defaults)

- If any task fails to *start*: terminate any started tasks and raise an error
- If a non-primary task exits early: log a warning, but keep running unless configured otherwise
- If primary fails: still terminate the group (after delay) and return/raise primary failure

## Future knobs

- `shutdown_on_any_task_failure: bool`
- Per-task `termination_delay_seconds` overrides
- `max_runtime_seconds` for primary
- “graceful stop” vs “kill” (SIGTERM then SIGKILL after timeout)

