# Trusted Configs

Canonical eval and RL configs that exercise the pathways we currently trust.

These are intentionally simpler than KernelBench. They are inspired by the
PrimeIntellect nightly reverse-text and alphabet-sort jobs, but narrowed to the
smallest configurations that still exercise the architecture we care about:

- explicit `sample_scorer` instead of legacy `score_fn`
- shared eval path
- conservative RL `sync` path
- conservative RL `async` path
- LoRA-enabled training on a small model

Use these first when validating the stack end to end. If one of these breaks,
the problem is usually architectural or infra-level, not task-specific.

## Files

- `eval_reverse_text_api.py`
  - Shared eval path against an API endpoint.
  - No dataset dependency.
  - Explicit `FunctionSampleScorer`.

- `rl_reverse_text_sync.py`
  - Conservative GRPO baseline.
  - `pipeline_mode="sync"`.
  - Small synthetic prompt set.
  - LoRA enabled.

- `rl_reverse_text_async.py`
  - Conservative async GRPO baseline.
  - `pipeline_mode="async"`.
  - Stream-style queue semantics via `pipeline_queue_size=0`.
  - Still avoids `true_pipeline`.

## Why Reverse Text

PrimeIntellect’s nightly RL coverage includes reverse-text and alphabet-sort.
Reverse-text is the better first trusted config here because it keeps the
environment and scoring simple while still exercising the rollout, scoring, and
training boundaries we care about.

## Expected Use

Eval:

```bash
python -m rollouts.eval.run --config rollouts/trusted_configs/eval_reverse_text_api.py --limit 5
```

RL:

```bash
python -m rollouts.run --config rollouts/trusted_configs/rl_reverse_text_sync.py --local
python -m rollouts.run --config rollouts/trusted_configs/rl_reverse_text_async.py --local
```
