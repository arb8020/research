# ServingRun lifecycle verification

Branch: `consumer-project-threading`. Four live-resource tests against the
MI355X SSH node (root@66.42.120.238). All `--force-deploy-committed`.

## What's verified

The "drain on shutdown" path, end to end:

1. **Warmup + leave-running** (`run_20260421-235125`).
   Fresh boot of sglang, `leave_endpoint_running=True` ran to the exit path
   and the container stayed up.
   Observable: `inference_endpoint_leave_running` event; `run_end status=ok`.

2. **Reuse fast-path** (`run_20260422-001308` and `_001558`).
   Second+ invocation with `reuse_running_endpoint=True` and a healthy endpoint
   at the port — `inference_endpoint_reused` fires ~5s after launch, no warmup.

3. **Duration drain** (`run_20260422-001558`).
   `duration=90s`, workloads with large `max_samples` so natural exhaustion
   doesn't short-circuit. Watchdog fires at T+90s → `shutdown_event` set →
   supervisor SIGINTs child → child's `_run_scenario` catches nested
   `BaseExceptionGroup` with only `KeyboardInterrupt`/`trio.Cancelled` leaves
   → `scenario_report.json` written with `status: drained` → child exits 0 →
   `serving_drain_clean exit_code=0` → `run_end status=ok`.

4. **SIGTERM drain** (`run_20260422-001817`).
   Sent `kill -TERM <supervisor-pid>` 15s into the run. Same drain sequence
   as (3) but triggered by `serving_signal_shutdown signal=SIGTERM` instead
   of duration watchdog. Also `run_end status=ok`.

## What's not verified

- **Second-signal force path** (`serving_signal_force`). Small code, visually
  reviewable — rapid double-SIGTERM in a test would also race the drain
  window depending on timing. Skipping.
- **`serving_drain_timeout` / `serving_drain_sigkill` escalation.** Requires
  a workload that refuses to drain within `drain_timeout`; none of our
  current workloads are that slow to cancel. Would need an artificial stall
  workload to exercise. Left to a future loop.
- **`on_engine_crash="restart"`.** Field is accepted but behaves as `"fail"`;
  the supervisor emits `serving_on_engine_crash_restart_not_implemented`.
  Wiring requires the eval child to tolerate base_url changing mid-run,
  which it doesn't. Separate loop.
- **Per-workload data preservation on cancel.** When drain fires mid-workload,
  `tau2_retail` is the only workload that made it into the
  `scenario_report.workloads` map in our tests — `sharegpt_bench` and
  `kimi_verifier` were cancelled before their `results[name] = result`
  assignment. Per-workload hygiene, not lifecycle plumbing.

## Artifacts

```
results/serving/run_20260421-235125/  # warmup + leave_endpoint_running
results/serving/run_20260422-001558/  # duration drain clean
results/serving/run_20260422-001817/  # SIGTERM drain clean
```

Each has `control.jsonl` (lifecycle events) and
`scenario_report.json` (per-workload rollup, with `status: drained` for the
two drain runs).
