# Runtime Ownership Cleanup

## Goal

Make the infrastructure stack honest:

- `rollouts` owns workload semantics
- `argus` owns run/attempt lifecycle
- `broker` owns compute allocation
- `bifrost` owns remote workspace/process execution

## Main Leaks Today

1. `rollouts.modal_runner` owns Modal app/image/sandbox/process orchestration.
2. `argus.run` knows too much about runtime/image/materialization details.
3. Modal is a special execution path instead of an execution substrate.
4. Remote workspace setup still relies on `sys.path` / `PYTHONPATH` hacks.
5. Sandbox ownership/lifecycle is ambient instead of explicit.

## Immediate Unblock

Do not globally terminate all sandboxes in the Modal app on launch.

Current stopgap:

- pre-create cleanup in `rollouts.modal_runner` only targets sandboxes tagged
  with the same Argus ownership identity (`control_plane`, `launcher_id`)
- `keep_alive=True` skips pre-create cleanup

This is a local safety fix, not the final architecture.

## Desired Ownership

### rollouts

- training/eval semantics
- model/backend lowering
- workload configs
- workload-aware logs and monitors

### argus

- run identity
- attempt lifecycle
- event journal and projections
- execution-substrate selection

### broker

- provider search/allocation/termination
- allocation refs and provider identity

### bifrost

- workspace materialization
- remote process/session lifecycle
- stdout/stderr/log transport
- artifact sync
- execution substrates such as SSH and Modal

## Phased Cleanup

1. Introduce a runtime/execution substrate interface outside `rollouts`.
2. Move Modal execution behind that interface.
3. Make `argus` call the substrate instead of `rollouts.modal_runner`.
4. Move generic runtime/image/materialization helpers out of `rollouts`.
5. Make sandbox ownership explicit:
   - `launcher_id`
   - `run_name`
   - `managed_by`
   - lifecycle mode: `ephemeral | keepalive | reuse`
6. Remove `sys.path` / `PYTHONPATH` hacks once remote installs are real.

## Non-goal

Do not block current debugging or training bring-up on the full cleanup. Local
unblocks are allowed when they make the current boundary more honest.
