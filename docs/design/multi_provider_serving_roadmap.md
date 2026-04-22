# Multi-provider serving roadmap

Topline goal: **profitable token serving on OpenRouter (or equivalent
marketplace).** That's the one metric that gates everything below.
Nothing in this doc matters if we can't make the margin math work.

The ladder below is ordered so each step produces data that informs
the next. Don't skip ahead.

## Step 1 — Make DSV3.2-on-MI355X honest

Already in flight. Current 7 tok/s-per-sequence number is either a
tuning artifact (cuda graphs disabled, tilelang decode, stale image)
or a genuine hardware/model ceiling. We don't know which yet; the
diagnostic handoff (see chat transcripts) should settle it.

**Gate:** MFU from engine_metrics.jsonl once it's flowing. If MFU <
~20% we're leaving hardware on the floor and need to tune. If MFU
is ~70%+ at 7 tok/s, 7 tok/s *is* the ceiling for this combination
and the economics question is largely decided.

## Step 2 — Serve from Modal

Prove the provider-swap surface works at all. Modal has native
autoscaling, so "serve from Modal" is also a proof that "some other
process owns the scaling decision" works end-to-end. Same mixture
config, just `provider="modal"` in `HardwareConfig`.

Deliverable: a Modal-backed variant of `mixture_mi355x_smoke.py`
that serves the same mixture (or a subset — Modal pricing means we
probably want fewer samples) through Modal-autoscaled sglang
replicas. Observe per-request tok/s, compare against MI355X.

**Gate:** Does Modal's autoscaler actually do what we want under
our mixture shape? Specifically: does it hold replicas warm during
brief quiet periods, or does it spin down aggressively between
samples?

## Step 3 — Serve from vast.ai H100 SXM (single node, no scaling)

The drop-in replacement proof. `HardwareConfig(provider="vast",
gpu_type="H100_SXM5", ...)` and nothing else changes. Reuse the
existing `_realize_ssh_endpoint` path — vast gives us an SSH target,
we bifrost-deploy, sglang boots, done.

Gives us a second data point for DSV3.2 tok/s on different hardware.
If H100 SXM is 5-10× faster than MI355X on the same workload (likely)
the economics question partially re-answers itself.

**Gate:** Does the "same config, different provider" story hold?
What vast-specific quirks surface?

## Step 4 — Serve from Lambda / Nebius (single node each)

Same as step 3, different providers. Two reasons:

1. Each one surfaces its own quirks. If we're going to scale across
   these providers later, we need to know what breaks on each.
2. Gives the autoscaler a pool of candidate providers rather than a
   single-point-of-failure.

## Step 5 — Multi-node pool with static size

First real departure from single-node. `ServingScenario` can have
N replicas behind a client-side round-robin dispatcher. Static size
(`min == max`). Probably mixed providers (2 vast + 1 lambda).

This is the step where the type design starts mattering. The types
the autoscaler will eventually use:

- **`NodeRequest`**: spec the autoscaler asks broker for
  (gpu_type, gpu_count, max_price_per_hour, min_lifetime, maybe
  preferred_providers).
- **`NodeOffer`**: what broker returns before commit (provider,
  price_per_hour, estimated_provision_seconds, preemption_risk,
  accept_token).
- **`Pool`**: the active set of (replica_id, base_url,
  provider, provisioned_at) tuples. Mutable.
- **Round-robin dispatcher**: stateless, just takes a Pool and
  hands out the next `base_url` per request.

Prefix-cache-aware routing is NOT part of step 5. Round-robin is
the first cut. Revisit if prefix cache hit rate matters for margin.

**Gate:** Does having 3 replicas actually give us ~3x aggregate
throughput, or does something pinch (client connection pool,
network, sglang per-process overhead)?

## Step 6 — ScalingPolicy + autoscaler controller

Now we design the autoscaler, with steps 1-5 as ground truth:

- **`ScalingPolicy`**: what the config carries
  (min_replicas, max_replicas, scale_up_signal, scale_down_signal,
  node_request).
- **`AutoscalerController`**: trio task watching engine metrics,
  mutating the pool, calling broker for `NodeOffer`s, accepting,
  waiting, adding to pool.
- **Modal lowering**: maps `ScalingPolicy` to Modal's native
  autoscaler primitives. Our controller doesn't run for Modal
  configs — we delegate.
- **Raw-node lowering**: our controller runs, calls into
  broker/bifrost to provision across vast/lambda/nebius.

Key design call we're deferring until we're here: how much does
the controller *need* to know about cost/latency/availability, and
does broker already surface it? Answer depends on what step 3/4
surface.

## Sequencing note

Steps 1-4 are single-node. Step 5 is the first multi-node. Step 6
is the first dynamic-capacity. Each step is a natural checkpoint
where "does this still make economic sense" is a reasonable
question to ask — and the instrumentation we have (spans +
engine_metrics) gives us the numbers to answer it at each gate.

## Anti-scope

Things explicitly NOT in this roadmap:

- Prefix-cache-aware routing (revisit if cache hit rate dominates
  margin).
- Real load balancer (envoy/caddy) — round-robin first.
- Training jobs. This is serving-only; training has its own path.
- OTel-to-Grafana pipeline. JSONL artifacts per-run are enough
  until we're operationally continuous (which is post-step-6).

## Related documents

- `docs/design/mixture_mi355x_spans_firstrun.md` — span tree
  verification, initial per-model latency numbers.
- `docs/design/consumer_project_threading.md` — why serving lives
  where it lives today.
- (forthcoming) `docs/design/dsv32_mi355x_slowpath_diagnosis.md`
  from the Tier-1/2/3 handoff.
