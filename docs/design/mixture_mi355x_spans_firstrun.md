# mixture_mi355x_smoke with spans.jsonl — first e2e verification

Run: `results/serving/run_20260422-013057/` (branch
`consumer-project-threading`, commits through `20370062`).

First end-to-end exercise of the six-level span tree under real
mixture load on MI355X. Goal was to verify hierarchy renders correctly
against 4 concurrent workloads + real sglang, not to tune numbers.

## Result

- `run_end status=ok exit=0`.
- 103 spans written, all 6 levels populated.
- Zero dangling parent references.
- Zero error spans.
- One root span (`serving_run`); depth distribution matches the
  designed tree (1 root, 4 workloads, 11 sample_attempts, 33
  agent_steps, 54 leaves = llm_calls + tool_calls).
- Example full chain (leaf → root):
  `tool_call ← agent_step ← sample_attempt ← workload ← serving_run`.

## What the spans tell us (that per-workload reports didn't)

Per-model latency breakdown — falls out of one `jq` over
`spans.jsonl`:

- **DSV3.2 on MI355X (endpoint under test):** n=31 LLM calls, p50
  50.3s, p95 135.9s, max 137.4s.
- **gpt-4o-mini (tau2 user simulator, via OpenAI):** n=4 LLM calls,
  p50 2.3s, max 2.7s. ~20× faster than the endpoint, as expected.

Having user-sim and endpoint-under-test calls in the same timeline,
differentiated by `gen_ai.request.model` / `gen_ai.system`, is the
thing we wanted — we can now reason about mixture-level latency
without conflating "traffic we're trying to serve" with "traffic that
supports the eval harness."

## What this proves

- Contextvar-based parent threading survives trio's `start_soon` fan-out
  across 4 parallel workloads.
- `emit-at-close` discipline holds: no span is missing an
  `end_time_unix_nano`; no partials.
- The two-writer story (eval child + serving supervisor child sharing
  one spans.jsonl) is still latent — the serving supervisor itself
  opened no spans this run, only the child did. That path remains
  to-be-exercised whenever the supervisor starts emitting lifecycle
  spans.
- Endpoint reuse fast-path (`inference_endpoint_reused`) worked; total
  wall time was ~20 min despite DSV3.2's slow decode under mixture,
  because no warmup. Warmup from cold would have added ~8 min.

## Notes worth following up on

- Engine `/metrics` poller → `engine_metrics.jsonl` is still not
  built. That's the other half of "observe the endpoint under load"
  and is the next logical step.
- DSV3.2 p95 of 136s on sglang is notable — a lot of it is the
  decode-under-contention signal from the earlier mixture run, but we
  should look at where the tail is before pointing real traffic at
  this.
- Harbor sample took 1159s (one TB2 task, DSV3.2, 20 turns, reward 0).
  That's a long single-sample tail that may have implications for
  mixture scheduling — harbor holds a worker for ~20 min while other
  workloads have finished.

## Reproduction

```
cd ~/research
.venv/bin/python -m argus run \
  --config rollouts/examples/serving/mixture_mi355x_smoke.py \
  --force-deploy-committed
```

spans.jsonl is at `results/serving/<run>/spans.jsonl`. Query with
`jq`; example one-liner for per-model latency:

```
jq -c 'select(.name=="llm_call") | {model: .attributes["gen_ai.request.model"], dur_ms: ((.end_time_unix_nano - .start_time_unix_nano) / 1000000)}' spans.jsonl
```
