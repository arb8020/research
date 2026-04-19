"""
Profile where time goes per battle step.
Measures: write (stdin), readline (Node wait), parse_block (poke-env), step total.
"""
import time, statistics, logging
import sim_bridge as _sb

write_times, readline_times, parse_times, step_times = [], [], [], []

_orig_write = _sb.ShowdownSim._write
_orig_readline = _sb.ShowdownSim._readline
_orig_process_block = _sb.ShowdownSim._process_block

def _tw(self, line):
    t = time.perf_counter(); _orig_write(self, line); write_times.append(time.perf_counter()-t)
def _tr(self):
    t = time.perf_counter(); r = _orig_readline(self); readline_times.append(time.perf_counter()-t); return r
def _tp(self, block):
    t = time.perf_counter(); _orig_process_block(self, block); parse_times.append(time.perf_counter()-t)

_sb.ShowdownSim._write = _tw
_sb.ShowdownSim._readline = _tr
_sb.ShowdownSim._process_block = _tp

from sim_bridge import ShowdownSim, random_choice
logging.disable(logging.CRITICAL)

all_steps = []
for _ in range(5):
    with ShowdownSim(gen=9) as sim:
        b1, b2 = sim.start("gen9randombattle")
        while True:
            t0 = time.perf_counter()
            c1 = random_choice(b1) if sim.needs_choice_p1 else None
            c2 = random_choice(b2) if sim.needs_choice_p2 else None
            sim.step(c1, c2)
            all_steps.append(time.perf_counter()-t0)
            if sim.done: break

def stats(name, s):
    s = sorted(s)
    if not s: return
    print(f"  {name}: mean={statistics.mean(s)*1000:.3f}ms  median={statistics.median(s)*1000:.3f}ms  p95={s[int(len(s)*.95)]*1000:.3f}ms  n={len(s)}")

print(f"\n=== {len(all_steps)} steps across 5 battles ===")
stats("write(stdin)", write_times)
slow_rl = [t for t in readline_times if t > 0.0001]
stats("readline total", readline_times)
stats("readline >0.1ms", slow_rl)
stats("parse_block", parse_times)
stats("step() total", all_steps)
print(f"  sps: {len(all_steps)/sum(all_steps):.0f}")
