# mixture_mi355x_smoke first run

Run: `results/serving/run_20260421-220353/` (from `~/research`,
branch `consumer-project-threading`).

Scenario: `rollouts/examples/serving/mixture_mi355x_smoke.py`.
Three workloads in parallel against one DSV3.2 endpoint on 8×MI355X:

- `tau2_retail` — multi-turn agent, 2 samples, concurrency 2, max_turns 20
- `kimi_verifier` — K2VV single-turn tool-call, 4 samples, concurrency 2
- `sharegpt_bench` — raw first-turn throughput, 8 samples, concurrency 8

Total request concurrency ceiling against one sglang: ~12.

## Result

All three workloads finished, `run_end status=ok`. Mixture wall time
≈168s post-warmup. All three workload dirs populated (report/
engine_report/events/samples).

## Numbers

**tau2_retail (multi-turn):**
- samples=2, avg_turns=2.0, 6 LLM calls
- llm p50 = 19.6s, p95 = 83.1s
- wall_time 168s

**kimi_verifier (single-turn tool call):**
- 4/4 success, 2/4 stop + 2/4 tool_calls
- schema-valid calls: 1/2 — consistent with earlier solo K2VV results
- duration 97s total

**sharegpt_bench (raw throughput):**
- 8 requests @ concurrency 8
- llm p50 = 138.1s, p95 = 160.4s
- mean output tokens/sec/request = 5.95
- wall_time 161s, 0.05 req/s

## Observations

Three things worth flagging, all of them the kind of thing the
mixture test is supposed to surface:

1. **sharegpt p50 inflated ~3–5× vs. solo.** Earlier solo sharegpt_tiny
   runs (2 samples, concurrency 2) saw ~37s per request. Under mixture
   load with 12-deep shared concurrency the p50 is 138s. This is
   expected shape-wise (more contention = more latency) but the
   magnitude is worth calibrating against OpenRouter SLAs before we
   ship.
2. **tau2 p95 llm_dur 83s.** Also high. Mostly agent calls waiting
   behind sharegpt's long decode queues.
3. **No request-level latency histogram exists today.** Everything
   above is per-workload summary rollup. For real serving we want one
   stream with one row per request (ts, workload_name, model,
   tokens_in, tokens_out, latency_ms, status), joinable with engine
   `/metrics` samples to correlate latency with batch size / KV usage.
   That's the next step.

## What this proves for the larger question

We can serve a realistic traffic mixture through one long-running
sglang on MI355X. The plumbing works end-to-end: config → scenario →
three parallel workloads → shared endpoint → per-workload reports.
What we cannot yet see clearly:

- How latency degrades with load curve (we sampled one point).
- What the engine was doing at each moment (no time series yet).
- What a sustained, steady mixture rather than a burst-then-exhaust
  run looks like (all three workloads today have finite sample caps).

Those are the observability + looping-workload pieces that come next
(steps 3 and 4 from the earlier sequence).

## Re-running

```
cd ~/research
.venv/bin/python -m argus run \
  --config rollouts/examples/serving/mixture_mi355x_smoke.py \
  --force-deploy-committed
```

Warmup is the dominant cost (~8–10 minutes). The mixture itself runs
in ~3 minutes after warmup.
