# pokemon-rl

Reinforcement learning for competitive Pokemon battles.

## Stack

- **[poke-env](https://github.com/hsahovic/poke-env)** — Gymnasium wrapper around Pokemon Showdown. Handles WebSocket comms, battle state parsing, observation/reward helpers, self-play.
- **[Pokemon Showdown](https://github.com/smogon/pokemon-showdown)** — Local battle simulator (Node.js/TypeScript). All game rules live here; no reimplementation needed.
- **[PufferLib](https://github.com/PufferAI/PufferLib)** — RL training orchestration with async vectorization. Wraps environments for parallel self-play.

## Architecture

```
PufferLib (training loop + thread pool)
    ↓
sim_bridge.ShowdownSim (synchronous subprocess bridge, one per thread)
    ↓
pokemon-showdown simulate-battle (one node process per battle, stdin/stdout)
    ↓
poke-env Battle parser (state management: team, active mon, moves, HP)
```

No WebSocket. Each battle is a `node pokemon-showdown simulate-battle` subprocess.
Choices sent via stdin (`>p1 move thunderbolt`), output read via stdout.
poke-env's `Battle` object parses the protocol and tracks all state.

**Benchmark (MacBook Air M-series, sequential single-threaded):**
- ~1.2 battles/sec
- ~110 steps/sec
- ~880ms/battle, ~88 steps/battle average
- Node process startup is the bottleneck (~750ms). Battle steps themselves are 0.5-5ms.
- With N concurrent battles in N threads, expect ~N× throughput until CPU-bound.

## Literature

### Must-read

**VGC-Bench** (Angliss et al., arXiv 2506.10326, June 2025)
- The primary VGC reference. Benchmark over 700k human battle logs. Evaluates heuristics, BC, BC+self-play, LLM agents, double oracle.
- Key result: BC+self-play beats World Championship competitors on *fixed* teams. Generalization degrades sharply as team diversity increases.
- This failure mode is the exact research problem here: the policy memorizes specific teams rather than understanding them.
- Code + dataset: https://github.com/cameronangliss/vgc-bench

**Adversarial Team Building** (Reis et al., IEEE Trans. Games 2023)
- Models team building as adversarial optimization; team builder maximizes win rate while a balancing agent readapts attributes for metagame diversity.
- Closest existing work to joint team building + battle policy, but doesn't co-train them end-to-end.
- https://ieeexplore.ieee.org/document/10115492/

**Metamon** (arXiv 2504.04395, April 2025)
- Offline RL + transformers trained on a decade of human singles data. Current SOTA for singles.
- Relevant for architecture: sequence models that adapt to opponents from trajectory alone, no explicit search.
- https://arxiv.org/abs/2504.04395

**Metagrok** (Huang & Lee, IEEE CoG 2019)
- Foundational PPO self-play work. Cleanest RL baseline. Competitive with humans on Showdown ladder.
- https://github.com/yuzeh/metagrok

### Secondary

**PokéChamp** (arXiv 2503.04094, March 2025) — minimax + LLM for action sampling, opponent modeling, value estimation. 76% vs prior LLM SOTA. No fine-tuning needed.

**PokéLLMon** (arXiv 2402.01118, Feb 2024) — first human-parity LLM agent. In-context RL + knowledge-augmented generation to fight hallucination on type matchups.

**Winning at Random Battles** (Wang, MIT thesis 2024) — PPO self-play + MCTS. Rank 8 on gen4randombattles ladder (1693 Elo). Best non-human result on random singles.

**PokeAgent Challenge** (arXiv 2603.15563, NeurIPS 2025 competition) — 20M+ trajectory dataset, 100+ competing teams. Two tracks: competitive battles + Pokemon Emerald speedrun.

### Gap this project targets

No existing work does learned *joint* team building + battle policy in VGC. VGC-Bench uses double oracle over fixed team sets — the team builder is not a learned generative policy co-trained with the battle policy. That's the open problem.

## Infrastructure References

- poke-env RL tutorial: https://poke-env.readthedocs.io/en/stable/examples/rl_with_gymnasium_wrapper.html
- Pokemon Showdown sim protocol: `sim/PROTOCOL.md`, `sim/SIM-PROTOCOL.md`
- PufferLib Pokemon Red integration (pokegym, 10x speedup): https://arxiv.org/html/2406.12905v1
- RL Pokemon bot example: https://github.com/hsahovic/reinforcement-learning-pokemon-bot

## Research Goal

Joint team building + battle policy. Most existing work either fixes teams (e.g. competitive ladder bots) or fully randomizes them (random battle formats). The interesting problem is learning *which teams to build* jointly with *how to play them*, since team composition and battle policy are deeply coupled — a team's value depends on whether the agent can execute its win conditions.

This requires high-scale self-play: team building is combinatorial (~1000 Pokemon × moves × items × abilities), and credit assignment from battle outcome back to team choices is noisy. Meta equilibrium also shifts as teams evolve, so you can't converge once and stop.

## Target Format

**Pokemon Champions — Regulation M-A** (`gen9championsvgc2026regma`)

Why this format:
- 185 legal Pokemon (vs 600+ in recent VGC formats) — tractable team building space
- No restricted legendaries, no restricted slot mechanic — clean team building structure
- IVs removed entirely, EVs replaced with Training Points — fewer dimensions to optimize
- Bring 6 pick 4 doubles, open team sheet, level 50 cap
- Mega Evolutions add strategic depth without bloating the pool
- Fully supported: Showdown has `[Gen 9 Champions] VGC 2026 Reg M-A`, poke-env 0.15.0 (April 14 2026) added Champions data

Format IDs:
- Singles: `gen9championsbssregma`
- Doubles VGC: `gen9championsvgc2026regma`
- Best-of-3: `gen9championsvgc2026regmabo3`

## Design Notes

- poke-env each step = round-trip to local Showdown server. Mitigate latency with `max_concurrent_battles`.
- PufferLib's async vectorization is the lever for scaling self-play.
- Showdown manages all game logic (move resolution, damage calc, turn order, hidden info). Don't fight it.
- Observation and reward shaping are the main places to invest engineering effort.

## Deferred: Native Simulator

Using Pokemon Showdown (via poke-env) for now. A native Python/C simulator becomes worth building if:
- Showdown IPC/WebSocket is measurably the bottleneck at scale (>10k concurrent battles)
- Batched/vectorized sim calls are needed (SIMD over game states) — not possible with Showdown's architecture
- GPU-side simulation becomes relevant for differentiable planning

The correctness cost is high (~50k lines of TypeScript covering 9 gens, every edge case). Only worth it if throughput is provably the limit after maxing out concurrency.

## Setup

Requires Node.js v20 via nvm (better-sqlite3 doesn't build against v24):

```bash
nvm install 20 && nvm use 20

# Python deps
uv sync

# Showdown (clone + npm install under node v20)
git clone --depth 1 https://github.com/smogon/pokemon-showdown.git
npm install --prefix pokemon-showdown
# patch port to avoid conflict (default 8000 may be occupied)
sed -i '' 's/exports.port = 8000/exports.port = 8769/' pokemon-showdown/config/config.js
```

Run a battle:
```python
from sim_bridge import ShowdownSim, random_choice

with ShowdownSim(gen=9) as sim:
    b1, b2 = sim.start("gen9randombattle")
    while True:
        c1 = random_choice(b1) if sim.needs_choice_p1 else None
        c2 = random_choice(b2) if sim.needs_choice_p2 else None
        done, winner = sim.step(c1, c2)
        if done:
            break
```

Benchmark/smoke test:
```bash
PYTHONUNBUFFERED=1 uv run python smoke.py
# logs to logs/smoke.jsonl
```

## TODO

- [x] subprocess bridge (sim_bridge.py) — synchronous, no WebSocket
- [x] poke-env Battle parser wired to subprocess output
- [x] smoke test: 10 battles complete cleanly, ~110 steps/sec single-threaded
- [ ] Concurrent battles via thread pool — benchmark N-thread throughput
- [ ] Define observation space (flat vector first, transformer later)
- [ ] Gymnasium env wrapper (subclass SinglesEnv, implement embed_battle)
- [ ] Wire up PufferLib training loop
- [ ] Random policy Elo baseline
- [ ] Champions BSS format (gen9championsbssregma) — needs team building
- [ ] Champions VGC doubles (gen9championsvgc2026regma)
