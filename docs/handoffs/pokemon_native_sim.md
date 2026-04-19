# Handoff: Native Pokemon Battle Simulator

## Context

We're training a PPO agent for competitive Pokemon battles (research goal: joint team building + battle policy). The training loop is working end-to-end. The bottleneck is the battle simulator.

Current stack: `ShowdownSim` (`dev/pokemon-rl/sim_bridge.py`) wraps `pokemon-showdown simulate-battle` as a Node.js subprocess. Python sends choices via stdin, reads protocol output from stdout, poke-env parses it into a `Battle` state object.

## Profiled bottleneck

Profiling (`dev/pokemon-rl/profile_step.py`) over 388 steps across 5 random battles:

```
write(stdin):    mean=0.002ms   — negligible
readline total:  median=0.000ms (empty lines), 393 slow calls mean=14.7ms median=0.76ms p95=2.9ms
parse_block:     mean=0.098ms   — negligible
step() total:    mean=1.251ms   sps=799
```

The bottleneck is **Node.js executing the game logic per turn** (~0.76ms median). The IPC pipe, Python parsing, everything else is negligible. A single Node process gives ~1300 turns/sec. That's the ceiling for the current architecture.

The gain from a native simulator isn't eliminating IPC — it's:
1. Faster turn computation (~5-20× over JS for equivalent logic)
2. **Batching**: compute N turns across N game states in a tight loop, no IPC, cache-friendly, potentially SIMD

Target: ~100k+ turns/sec per CPU (vs 1300 currently).

## What the native sim needs to implement

Target format: **gen9randombattle** first (for diff testing), then **gen9championsbssregma** (BSS singles, the actual research target — 185 legal Pokemon, no legendaries, bring-3-pick-3).

Minimum viable scope for diff testing:
- Singles battle (1v1 active)
- Gen 9 damage formula (including STAB, type effectiveness, random roll, crits)
- Status conditions: BRN, PAR, PSN, TOX, SLP, FRZ
- Stat stages (boosts -6..+6)
- Weather: sun, rain, sand, snow
- Terrain: electric, grassy, misty, psychic
- Priority moves, speed ties
- Switching (voluntary + force-switch on faint)
- ~50 most common moves in the format (Showdown has usage stats)
- ~50 most common abilities
- ~50 most common items

Full correctness on every edge case is not required initially — diff testing against Showdown will reveal gaps.

## Interface contract

The native sim must expose the same logical interface as `ShowdownSim`:

```python
class NativeSim:
    def start(self, format_id: str, p1_team=None, p2_team=None) -> (Battle, Battle):
        """Start a battle. Returns poke-env Battle objects (or compatible state)."""

    def step(self, choice_p1: str | None, choice_p2: str | None) -> (done: bool, winner: str | None):
        """Advance one turn. Returns when both sides have made choices."""

    @property
    def needs_choice_p1(self) -> bool: ...
    @property
    def needs_choice_p2(self) -> bool: ...
    @property
    def done(self) -> bool: ...
```

Choice strings follow the poke-env wire format: `"move thunderbolt"`, `"switch pikachu"`.

The observation encoder (`dev/pokemon-rl/obs.py`) reads from poke-env `Battle` objects. Either:
- The native sim populates poke-env `Battle` objects directly (reusing parse_message), or
- It returns a compatible state struct and `obs.py` is updated to read from it

Option A (populate poke-env Battle) is easier to diff test. Option B is cleaner long-term.

## Diff testing plan

1. Run N battles in Showdown, record: (initial teams, sequence of choices, sequence of resulting states)
2. Replay same choices through native sim, compare resulting states at each step
3. States to compare: HP fractions, status, stat stages, field conditions, available moves
4. Mismatch = correctness bug in native sim

The existing `verify_state.py` (`dev/pokemon-rl/verify_state.py`) shows the comparison pattern — it already diffs poke-env state against raw Showdown request JSON. The diff test can extend this.

## Relevant files

```
dev/pokemon-rl/
├── sim_bridge.py       # ShowdownSim — the thing to replace
├── obs.py              # Observation encoder — reads from poke-env Battle
├── env.py              # Gymnasium env — wraps ShowdownSim
├── vecenv.py           # Vectorized env (currently ThreadedVecEnv)
├── train.py            # PPO training loop
├── verify_state.py     # State correctness verifier (diff test pattern)
├── profile_step.py     # Profiling script that generated the numbers above
└── pokemon-showdown/   # Local Showdown install (Node v20 required)
```

## Language recommendation

**Zig** — same performance as C, better safety, easier FFI to Python via `ctypes`/`cffi`, and pkmn/engine (the reference Zig Pokemon sim) is a useful reference even though it only covers Gen 1-2. The authors clearly know how to structure this problem in Zig.

**Rust** is also fine — `PyO3` makes Python bindings clean, and the borrow checker eliminates a class of bugs you'd hit writing game state mutation in C.

**C** works but is harder to maintain correctly. Only worth it if you need to embed in environments where Zig/Rust aren't available.

Avoid: reimplementing everything. Start with the 50 most common moves/abilities/items and expand via diff testing.

## What NOT to do

- Don't try to cover all 1000+ moves in Gen 9 upfront. Diff test iteratively.
- Don't reimplement the team builder — use Showdown's packed team format for teams, only implement the battle engine.
- Don't try to match Showdown's RNG exactly initially — use your own seeded RNG, make it deterministic, diff test on the distribution of outcomes not exact sequences.
- Don't implement doubles yet — singles first, correctness first.

## Current training throughput

Single-threaded on Mac M-series: ~800 sps.
With N threads on N CPUs: ~N × 800 sps (GIL is not the bottleneck — parse_block is only 0.1ms).
Target with native sim: ~100k sps per CPU, fully batchable.

## Quick start for diff testing

```python
# Record a Showdown battle
from sim_bridge import ShowdownSim, random_choice
import json

with ShowdownSim(gen=9) as sim:
    b1, b2 = sim.start("gen9randombattle")
    trajectory = []
    while True:
        c1 = random_choice(b1) if sim.needs_choice_p1 else None
        c2 = random_choice(b2) if sim.needs_choice_p2 else None
        done, winner = sim.step(c1, c2)
        trajectory.append({
            "c1": c1, "c2": c2,
            "p1_hp": {p.species: p.current_hp_fraction for p in b1.team.values()},
            "p2_hp": {p.species: p.current_hp_fraction for p in b2.team.values()},
        })
        if done:
            break

# Replay through native sim and compare trajectory
```
