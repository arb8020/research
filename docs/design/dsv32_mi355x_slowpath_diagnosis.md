# DSV3.2 on MI355X Slow-Path Diagnosis

Date: 2026-04-21

## Question

Was the observed `~7 tok/s` on `deepseek-ai/DeepSeek-V3.2` over 8x MI355X a hardware/model limit, or a tuning artifact?

Short answer: it was a tuning artifact.

## Baseline Findings

The warm server that produced the bad result was the container:

- `lmsysorg/sglang:v0.5.9-rocm700-mi35x`
- container name: `sglang_mixture_30000`

The live container command and logs showed:

- `--disable-cuda-graph`
- `--nsa-prefill-backend tilelang`
- `--nsa-decode-backend tilelang`
- `enable_metrics=False` on the old warm process, so the new poller could not have worked there
- page size forced to `1` at runtime on ROCm DeepSeek DSA
- KV cache dtype forced to BF16
- no explicit fallback or kernel-error log line explaining the `~7 tok/s` behavior

The isolated single-request measurement on the warm server was still slow, which ruled out mixture contention:

- request: one `chat/completions` call, `max_tokens=64`
- result: `64 / 8.6045s = 7.44 tok/s`

That was already enough to say the bad number was not caused by the other workloads.

## Upstream Check

I checked current SGLang primary sources before changing anything:

- PR `#16841`: <https://github.com/sgl-project/sglang/pull/16841>
  - merged `2026-01-14`
  - explicitly enables CUDA graph capture for DeepSeek-V3.2 NSA tilelang on AMD with `--cuda-graph-max-bs 64`
- PR `#18319`: <https://github.com/sgl-project/sglang/pull/18319>
  - merged `2026-02-27`
  - makes tilelang the default NSA backend on AMD Instinct
  - reports MI355X throughput far above our `~7 tok/s` floor
- PR `#21511`: <https://github.com/sgl-project/sglang/pull/21511>
  - merged `2026-04-03`
  - adds FP8 KV cache + FP8 attention kernel for NSA tilelang on MI300/MI355
  - reports MI355 concurrency-1 throughput `505.78 tok/s`
- PR `#18741`: <https://github.com/sgl-project/sglang/pull/18741>
  - merged `2026-02-12`
  - updates ROCm 7.2 image to newer AITER

Important inference: our pinned image tag `v0.5.9-rocm700-mi35x` was published on Docker Hub on `2026-02-25`, which is after PR `#16841` merged. So re-enabling CUDA graphs on the same image was a justified single-knob experiment. I did not switch decode backends or image tags in this run.

## Single Change

Config change:

- file: `rollouts/examples/serving/mixture_mi355x_smoke.py`
- commit: `d33f5c9a` (`Re-enable CUDA graphs in MI355X mixture`)
- change:
  - removed `--disable-cuda-graph`
  - added `--cuda-graph-max-bs 64`

Everything else stayed fixed:

- same image: `lmsysorg/sglang:v0.5.9-rocm700-mi35x`
- same tilelang prefill/decode path
- same model
- same poller wiring

## Rerun Notes

The direct cold-boot path through:

```bash
cd ~/research && .venv/bin/python -m argus run --config rollouts/examples/serving/mixture_mi355x_smoke.py --force-deploy-committed
```

hit an unrelated infra issue: the deploy step hung in bundle `rsync` before Docker launch. I therefore used the exact committed `docker run` command from the config to cold-boot the endpoint on the node, then ran the local mixture through the normal reuse path.

That means:

- the serving endpoint did run the committed config
- the workloads and poller did run through rollouts locally
- the `argus` cold-boot deploy path itself still needs separate debugging

Relevant run directories:

- invalid reused run: `results/serving/run_20260422-054557/`
- hung deploy attempt: `results/serving/run_20260422-054717/`
- valid tuned local reuse run against the fresh endpoint: `results/serving/run_20260422-055230/`

## What The Tuned Server Proved

The fresh server log showed CUDA graph capture succeeded:

- `Capture cuda graph end`
- `Registering 1476 cuda graph addresses`

The Prometheus endpoint also worked:

- `curl http://127.0.0.1:30000/metrics` returned SGLang metrics
- `results/serving/run_20260422-055230/engine_metrics.jsonl` was populated

Observed poller output for the valid run:

- rows written so far: `101201`
- distinct scrapes: `133`
- metrics include `sglang:cuda_graph_passes_total`, `sglang:routing_key_running_req_count`, latency histograms, and token counters

The very first rows in `engine_metrics.jsonl` were already CUDA-graph metrics, e.g. `sglang:cuda_graph_passes_total{mode="decode_cuda_graph", ...}`.

## Post-Change Throughput

### Solo request

Fresh server, one short request, same measurement shape as the bad baseline:

- `64 / 1.9630s = 32.60 tok/s`

That is the cleanest apples-to-apples comparison against the earlier isolated `7.44 tok/s`.

### Mixture workload

From `results/serving/run_20260422-055230/workloads/sharegpt_bench/report.json`:

- aggregate output throughput: `102.54 tok/s`
- per-request mean output rate: `14.69 tok/s`
- concurrency: `8`

From `results/serving/run_20260422-055230/workloads/tau2_retail/report.json`:

- mean per-call output rate: `12.89 tok/s`

From Harbor session events in `results/serving/run_20260422-055230/workloads/harbor_tb2/events.jsonl`:

- `33` LLM calls recorded
- all-call mean: `20.95 tok/s`
- turns `>= 7` mean: `28.01 tok/s`
- turns `>= 13` mean: `27.67 tok/s`
- max observed Harbor call: `33.41 tok/s`

Harbor still showed some early slow calls near the old rate:

- earliest turn-0 calls were `6.85`, `7.18`, `7.58`, `7.76 tok/s`

But unlike the bad baseline, it did not stay pinned there. Later Harbor turns repeatedly landed around `24-33 tok/s`, including:

- turn 7: `32.71 tok/s`
- turn 8: `33.41 tok/s`
- turn 13: `32.27 tok/s`
- turn 14: `32.59 tok/s`
- turn 16: `30.98 tok/s`
- turn 18: `32.46 tok/s`

## Conclusion

`~7 tok/s` was not the hardware/model limit on 8x MI355X.

It was a slow-path artifact caused by serving with CUDA graphs disabled. Re-enabling CUDA graphs on the same image and same tilelang path moved the clean isolated rate from:

- `7.44 tok/s` -> `32.60 tok/s`

and moved later Harbor turns from the previously reported `~7.3 tok/s` plateau into the `~24-33 tok/s` range.

## Caveats

- The Harbor verifier hit an unrelated Modal sandbox API break:
  - `AttributeError: 'Sandbox' object has no attribute 'filesystem'`
  - this affected the Harbor verification/report flush path, not the serving measurements above
- The direct `argus --force-deploy-committed` cold-boot path hung in repo bundle sync
  - that is an infra/deploy issue separate from SGLang throughput
- I did not test `--nsa-decode-backend aiter`
  - the original `page_size` float/int decode-bug concern is still not proven fixed enough to justify switching that knob blindly

## Next Knob

If we do one more run, the next honest experiment is not "patch kernels." It is one of:

- newer image tag
- FP8 KV cache on tilelang

Those should be tested one at a time.
