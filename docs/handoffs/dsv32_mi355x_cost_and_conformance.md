# DSV3.2 on MI355X: conformance and cost snapshot

Status snapshot after getting tool-calling end-to-end on our `root@66.42.120.238` MI355X deployment. Numbers below are from one K2VV 200-row run (`results/serving/run_20260421-004752`) against the fixed launch config.

## Conformance (K2VV schema accuracy)

| metric | value |
|---|---|
| requests | 200/200 success |
| finish_stop | 75 |
| finish_tool_calls | 124 |
| successful_tool_call | 101 |
| schema_validation_errors | 23 |
| finish_other | 1 (length) |
| **schema_accuracy** | **101/124 = 81.5%** |
| **trigger_rate** | **124/200 = 62%** |

K2VV reference prices for comparison: MoonshotAI/Fireworks 100%, SGLang-on-K2 73-95%, vLLM-on-K2 76-87%. Our 81.5% on DSV3.2 is in the same band as other community SGLang listings. Trigger rate below K2's own (~95%) is expected because the corpus is K2-authored (Kimi identity prompt, K2 tool-call ID convention in prior assistant turns).

## Cost

| | |
|---|---|
| node | 8x MI355X at `root@66.42.120.238` |
| rental (amortized) | `NODE_COST_USD_PER_HOUR = 22.18` (from `bench_deepseek_v3_2_amd_mi355x.py`) |
| wall time for 200 rows | 38.5 min (2311s) |
| startup overhead | 121s (~5% of run) |
| input tokens | 323,904 |
| output tokens | 72,319 |
| aggregate throughput | **31.3 output tok/s** (concurrency=8) |
| active-compute cost | $14.24 |
| startup-amortized cost | $14.99 |
| **break-even** | **$207/M output tokens** |
| | **$38/M total tokens** |

Market anchors for DSV3.2 (rough, check openrouter.ai/providers for live): **$0.50-$2.00 per 1M output tokens**. We are roughly **100× above market** at current throughput.

## What's between us and market

Node rental is fixed. The only lever is tokens/second. To cover cost at market prices we need on the order of **3000 output tok/s aggregate** (100× current). Levers in rough order of expected impact:

1. **Concurrency**. c=8 is K2VV's smoke setting, not production. Real providers run hundreds of concurrent requests. At higher concurrency the batch scheduler is much more efficient.
2. **Tail truncation**. One request in this run took >18 minutes alone (hit row's `max_tokens=16000`). It pinned wall time for the whole workload. Production clients cap at much lower `max_tokens`.
3. **Cache utilization**. `--enable-cache-report` logs cache stats; we're not yet reading them. Prefix caching for repeated system prompts / tool schemas would cut input tokens drastically.
4. **Kernel path**. We use `--nsa-decode-backend tilelang` for correctness (aiter path has a known upstream version-skew bug on v0.5.9). Decode with aiter once the bug is fixed upstream will be faster.

## Reproducing

Working launch config lives in `rollouts/examples/inference/evals/configs/bench/bench_deepseek_v3_2_amd_mi355x.py` (commit `54b7e74`). The critical lesson — *do not pass `--chat-template` for the non-Exp model; the bundled template file is mislabeled* — is written up in that commit's message.

To reproduce this snapshot:
```bash
python -m argus run --config rollouts/examples/serving/kimi_verifier_deepseek_v32_mi355x_smoke.py --force-deploy-committed
# then, once the run finishes:
python rollouts/scripts/scan_serving_run.py results/serving/<run>/
python rollouts/scripts/cost_floor_from_spans.py --run-dir results/serving/<run>/ --node-cost-per-hour 22.18
```

## What this doesn't tell us yet

- **Conformance on non-K corpora.** We haven't re-run AIME+calc with the fixed config (previous attempts were pre-fix, all invalidated).
- **Throughput at production concurrency.** The 31.3 tok/s figure is with c=8 and a tail request dominating; higher-concurrency / trimmed-max-tokens runs would give a more honest utilization number.
- **Marginal vs. amortized node cost.** The $22.18/hr figure is 20-day amortized total spend; when the node is pinned to inference, marginal cost may be materially different. (Flagged as TODO(cost) in the bench config.)
- **Break-even at margin.** All numbers above are 0-margin. `cost_floor_from_spans.py --required-margin 0.2` computes 20% gross-margin price.
