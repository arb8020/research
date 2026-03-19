# Reasoning Theater

Minimal scaffold for reproducing `Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought` on top of `rollouts`.

The intent is narrow:
- use `rollouts` as the execution and sample-denotation spine,
- keep reasoning traces structured via `ThinkingContent`,
- keep token provenance when available via `prompt_token_ids` and `Choice.token_ids`,
- leave activation replay and probe training as separate stages rather than pretending they already live inside `rollouts`.

## Design

The core shape is:

1. Normalize benchmark rows into multiple-choice task payloads.
2. Run each sample through a custom `attempt_executor`.
3. Preserve the full `AttemptRow` / `Trajectory` produced by the rollout backend.
4. Attach derived metadata:
   - parsed final answer
   - reasoning text
   - prefix slices
   - forced-answer predictions
   - CoT-monitor predictions
   - artifact references for replay / probes
5. Score those enriched samples with a `sample_scorer`.

That maps cleanly onto the existing `rollouts` contracts:
- `ProblemRow`
- `AttemptRow`
- `Trajectory`
- `ThinkingContent`
- `attempt_executor`
- `sample_scorer`

## Files

- `config.py`: experiment config dataclasses
- `records.py`: task and prefix record types
- `datasets.py`: multiple-choice normalization / parsing helpers
- `prefixes.py`: prefix slicing logic
- `attempt_executor.py`: `rollouts`-oriented executor builder
- `scorers.py`: scorer stage for enriched samples
- `generate_rollouts.py`: normalize raw JSONL tasks into eval payloads
- `replay_activations.py`: extract replay manifests from saved `AttemptRow` JSON/JSONL
- `eval_rollouts.py`: template `rollouts.eval.run` config module

## TI/TO

Yes, `TI/TO` matters here.

It is not only an RL concern. For this project, it matters whenever we care about:
- exact reasoning-prefix boundaries,
- replaying the generated sequence for activation extraction,
- probe labels aligned to the actual generated rollout,
- avoiding decode -> re-encode drift in self-hosted inference.

Current rule of thumb:
- self-hosted `sglang` / `vllm`: prefer `TI/TO` and preserve `prompt_token_ids` + `completion_token_ids`
- API providers without token IDs: fall back to text-prefix analyses, but treat token-level claims as weaker

## Quick Start

Normalize a JSONL benchmark into `rollouts`-friendly payloads:

```bash
uv run /Users/chiraagbalu/research/dev/reasoning-theater/generate_rollouts.py \
  --input /path/to/raw_tasks.jsonl \
  --output /tmp/reasoning_theater_tasks.jsonl
```

Extract replay manifests from saved `AttemptRow` records:

```bash
uv run /Users/chiraagbalu/research/dev/reasoning-theater/replay_activations.py \
  --input /path/to/attempts.jsonl \
  --output /tmp/replay_manifest.jsonl
```

Run the template eval config after wiring in a real backend:

```bash
python -m rollouts.eval.run \
  --config /Users/chiraagbalu/research/dev/reasoning-theater/eval_rollouts.py
```

## Honest Gaps

This folder does not yet implement:
- activation extraction
- probe training
- probe inference
- monitor-model prompting policy
- a real rollout backend for `GPT-OSS-120B`

Those are the next layers. This scaffold only makes the denotation explicit and keeps the interfaces clean.
