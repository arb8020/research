# TTT (Test-Time Training) Implementation in Rollouts

## Goal
Implement Discover-style test-time training where a model iteratively improves on a single problem by: sample → grade → update weights → repeat.

## Verification Strategy
Each step has a clear manual test to confirm it works before moving on.

---

## Phase 1: Minimal Loop (Core TTT)

### Step 1: Create FibonacciEnv
- [ ] Create `rollouts/environments/ttt_fibonacci.py`
- [ ] Environment prompts: "Write a Python function `fib(n)` that returns the Nth Fibonacci number"
- [ ] Grading: run the code, check correctness on test cases, measure speed
- [ ] Reward = correctness_score + speed_bonus (faster = higher)

**Test:** Run one rollout manually, print reward. Verify:
- Correct O(n) solution scores higher than incorrect
- Correct O(1) closed-form scores higher than O(n) recursive

### Step 2: Verify GRPO Works on FibonacciEnv
- [ ] Run existing GRPO training loop on FibonacciEnv
- [ ] Use standard setup (multiple prompts, normal RL)
- [ ] Confirm reward improves over training

**Test:** Loss goes down, avg reward goes up. This proves the reward signal is learnable.

### Step 3: In-Loop LoRA Weight Updates (Actual TTT)
- [ ] Modify loop: after each batch, update weights, then sample again from SAME problem
- [ ] Key difference from normal RL: single problem, weights change between sampling rounds
- [ ] Checkpoint LoRA weights between iterations

**Test:** Run 10 iterations on SAME Fibonacci problem:
- Loss trends down
- Avg reward trends up
- Best solution improves (e.g., starts O(2^n) → ends O(n) or O(1))

---

## Phase 2: Optimizations (After Core Works)

### Step 4: State Tracking + Parent Lineage
- [ ] Create `TTTState` dataclass: `id`, `value`, `code`, `parent_id`, `timestep`
- [ ] Persist states to JSON (one file per iteration)
- [ ] Track which solutions came from which

### Step 5: Sampler Interface
- [ ] `GreedySampler`: keep top-k states by value, expand from those
- [ ] `PUCTSampler`: tree search with exploration bonus

### Step 6: Best-of-N Selection
- [ ] Generate N solutions, rank by reward, select best
- [ ] Use as baseline comparison for TTT

---

## Architecture Notes

### What Rollouts Already Has
- `Environment` protocol + async tool execution
- `Trajectory` + `Score` with weighted metrics
- GRPO training loop with token-level generation
- `AsyncRolloutManager` for parallel sampling

### What We Need to Build
| Component | Effort | Priority |
|-----------|--------|----------|
| FibonacciEnv | Low | P0 - need this first |
| Verify GRPO works | Low | P0 - sanity check |
| In-loop LoRA updates | Medium | P0 - core TTT |
| State tracking | Low | P1 - after core works |
| Samplers | Medium | P1 - optimization |

---

## Current Status
- [ ] Step 1: FibonacciEnv (NOT STARTED)
