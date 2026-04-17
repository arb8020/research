# sillytavern_text (v0)

Roleplay / long-context workload approximating SillyTavern. Primary agent =
character (policy under eval, playing a card). Responder = LLM-backed user-sim.

## What v0 is

- Real character cards (Character Card Spec V2) + real franchise lorebooks
  (RWBY, Genshin, MHA), ~200 lore entries each.
- Lorebook activation: simplified port of ST's `checkWorldInfo`. Covers
  constant-on entries, primary-key substring activation, token budget
  eviction, before/after positioning. See `world_info.py` for the list of
  skipped features (recursion, secondary keys, etc.).
- Single activation pass at session start (not per-turn). `TODO(fidelity)` in
  `eval.py`.
- Scoring is trace-sanity only. Judge scoring is v0.1.

## What v0 is not

- **Not a faithful ST reimplementation.** ST's prompt builder is deeply
  coupled to its runtime (see the imports in `world-info.js` — 25+ modules of
  app state). A fully faithful alternative is running ST headless with an
  OpenAI-compatible interceptor. We considered it and chose the Python port
  for v0 because:
  - Unknown cost to boot ST headless + isolate per-sample state vs known
    cost of a ~150-LOC port.
  - v0's job is to prove the user-sim + DialogueEnvironment pipeline works
    for RP, not to nail fidelity.
- Not currently targeting NSFW cards. Raw assets are gitignored regardless
  (`raw/` and `tasks.jsonl`).

## Running

```bash
# 1. Drop cards + lorebooks in raw/ (see PAIRINGS in prepare.py for names)
# 2. Normalize → tasks.jsonl
cd /Users/chiraagbalu/research/rollouts/evals/sillytavern_text
python prepare.py

# 3. Smoke run (1 scenario × 8 turns, Haiku both sides)
cd ..
python run_eval.py --config sillytavern_text/configs/smoke.py
```

Artifacts land in `results/` (gitignored).

## Files

- `world_info.py` — ST lorebook activation port (~150 LOC)
- `prepare.py` — raw/ → tasks.jsonl, one row per (card, lorebook) pair
- `eval.py` — `EvalSpec` + `DialogueEnvironment` wiring, primary=character,
  responder=user-sim
- `configs/smoke.py`, `configs/full.py` — config entrypoints
- `raw/` (gitignored) — user-supplied cards and lorebooks

## Roadmap

- **v0.1** — LLM-as-judge scoring (staying in character, using lore, engagement)
- **v0.2** — per-turn re-activation with sliding scan window
- **v0.3** — rolling summary + token budget eviction of chat history
- **v0.4** — switch to headless ST if fidelity becomes load-bearing
- **v1** — folded into inference benchmarking (latency/token accounting across
  turn depths) and RL training (user-sim as environment, judge as reward)
