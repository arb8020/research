# Framework Comparison: rollouts vs frontier-evals/alcatraz, harbor/terminal-bench, prime-rl, verifiers

Date: 2026-03-17

This compares rollouts against four external frameworks across the dimensions where they
overlap: sandboxing, task definition, agent execution, reward/scoring, training, and
observability. The goal is to identify where rollouts is weaker, stronger, or doing
something different.

---

## The Frameworks

- **rollouts** (this repo): composable eval + GRPO training framework, Modal GPU sandboxes, multi-agent DSL
- **nanoeval + alcatraz** (OpenAI, frontier-evals): lightweight eval runner + Docker isolation layer
- **harbor + terminal-bench** (terminal-bench team): containerized agent harness, ~100 real terminal tasks
- **prime-rl** (PrimeIntellect): large-scale async RL training, FSDP2, vLLM, AIPO loss
- **verifiers** (PrimeIntellect): composable reward/environment library for prime-rl

---

## 1. Sandboxing / Execution Environment

### nanoeval + alcatraz

Docker-based isolation designed for multi-tenant eval security. The alcatraz name signals
that the security posture is the primary concern—untrusted code, strict
process/network isolation. Runs locally with no cloud dependency.

Execution modes: in-process async (default) or multiprocessing pool. SQLite in `/dev/shm`
tracks task state, giving fault tolerance: interrupted evals can resume from where they
left off.

### harbor / terminal-bench

Docker locally; Daytona or Modal for parallel cloud workloads. The primary primitive is a
tmux session that the agent interacts with via `send_keys` / `capture_terminal`. This is a
significantly different model from file-based execution—it tests agents in real interactive
terminal environments, including jobs that require waiting, watching output, and
reacting.

Container images are task-specific. Each task gets a fresh container from a known image,
so verification scripts run in a reproducible state.

### prime-rl / verifiers

vLLM inference server runs separately from the trainer. The `SandboxEnv` in verifiers
provides isolated Python execution per rollout. Environments are ephemeral per sample;
no persistent container reuse. Primary use case is single-turn or multi-turn
text/code generation—not general OS-level task execution.

### rollouts

Modal sandboxes via the broker abstraction. Differences from the above:

**Strengths:**
- GPU-first by design. Default: A100, CUDA base image, torch/triton preinstalled. alcatraz
  and harbor target CPU/general compute. For GPU kernel evals (KernelBench, etc.) this is
  directly load-bearing.
- Per-command retry with sandbox reset (3 total attempts: same-sandbox retry → reset
  retry). More resilience than Docker kill-and-restart.
- Reconnect to existing sandboxes via `modal.Sandbox.from_id()`. Supports session
  persistence across eval runs, useful for long-running tasks or mid-run attach.
- `ModalSandboxManager` pool with async channel-based queuing, reuse statistics, and
  full state serialization.

**Weaknesses:**
- No Docker/local sandboxing. Modal is required. This means no offline use, higher
  iteration cost for fast dev loops, and dependency on Modal's availability.
- No tmux-based interactive terminal primitive. rollouts executes commands and reads
  file output; it cannot simulate a live terminal session where the agent watches
  output stream and reacts to prompts. Terminal-bench tasks that require interactive
  terminals are not expressible.
- No fault-tolerant task queue. If a rollouts eval crashes mid-run, it restarts from
  scratch. nanoeval's SQLite queue survives crashes and resumes.

---

## 2. Task / Benchmark Definition

### nanoeval

~100-line eval files. Four primitives: `Eval`, `EvalSpec`, `Task`, `Solver`. Explicit
separation of data loading from completion and parsing logic. SQLite-backed distributed
queue for fault tolerance and observability into in-flight work.

### terminal-bench

Each task: English instruction + verification script + reference solution. The
verification pattern is auditable—did a script pass or fail? No custom scoring logic
needed per task. ~100 real-world terminal tasks covering compilation, server setup,
model training, system configuration.

### verifiers (PrimeIntellect)

Self-contained Python modules bundling dataset + harness + rubric. `SingleTurnEnv` and
`MultiTurnEnv` cover the spectrum from single-completion to multi-step interactions.
`StatefulToolEnv` adds state persistence across turns. Environments publish to a Hub,
enabling reuse across experiments.

`pass@k` and ablation sweep support are built in (v0.1.11). Monitor rubrics for automatic
metric collection (v0.1.9).

### rollouts

Config modules with `tasks` list + `prepare_messages()` + `score_fn()`. Flexible but
informal—no Hub, no standard task format for complex multi-step tasks.

**Strengths:**
- Arbitrary Python in `score_fn`. Complex scoring logic (e.g., GPU kernel performance
  ratios) is expressible without fighting a framework.
- Same config format works for evals and as the basis for GRPO training. No translation
  layer between "eval task" and "training task."
- `EvalRunSpec` supports external agent trajectories (Claude Code, Codex, OpenHands) via
  normalized adapters, so the same scoring applies regardless of agent type.

**Weaknesses:**
- No task catalog. Terminal-bench has ~100 verified tasks; verifiers has a Hub.
  rollouts has no comparable library—you bring your own.
- Verification pattern less formal than terminal-bench. It is easy to write a `score_fn`
  that silently scores wrong. Terminal-bench's "run a script, did it pass" is harder to
  get wrong.
- No `pass@k` or ablation sweep support built in.
- No fault-tolerant task queue. nanoeval resumes; rollouts restarts.

---

## 3. Agent Execution / Driver Model

### nanoeval

`Solver` is the agent abstraction. Typically involves model sampling. No first-class
support for external agent CLIs (Claude Code, Codex) as drivers.

### harbor / terminal-bench

Supports Claude Code, OpenHands, Codex CLI as agents. Integration is at the harness level:
run the agent CLI against a container, check pass/fail. No trajectory normalization or
unified scoring across agent types.

### verifiers (PrimeIntellect)

Generates rollouts via prime-rl's vLLM inference server. Multi-turn via `MultiTurnEnv`
with trajectory-based tracking (token-in, token-out across turns). `ToolEnv` and
`StatefulToolEnv` support tool use. The runtime is always the vLLM-served model; external
CLIs are not a supported pattern.

### rollouts

**Strengths:**
- Unified driver model with protocol (`ExternalAgentRuntime`). `ClaudeDriver`,
  `CodexDriver`, `CursorDriver` all emit the same `StreamEvent` types. External agent
  output is normalized into `Trajectory` via `trajectory_from_claude_code()`,
  `trajectory_from_codex()`, `trajectory_from_openhands()`.
- Native rollouts agent (full message loop, composable stop handlers, tool execution via
  `Environment` protocol) and external CLI drivers share the same scoring layer.
- `OrchestrateEnvironment` Python DSL for multi-agent orchestration: subagents write
  async Python calling `system.thread()`, `system.query()`, `system.allocate()`. Native
  trio nurseries for concurrency. No equivalent in nanoeval, alcatraz, or harbor.

**Weaknesses:**
- `kind` parameter in `OrchestrateEnvironment` conflates Worker/Runtime/Capabilities
  into one string. The design doc (`external_agent_runtimes.md`) acknowledges this is
  wrong; the Worker/Runtime/Capabilities split is planned but not implemented.
- One-shot Claude Code integration in `ClaudeCodeEnvironment` (environments/claude_code.py).
  Bidirectional mode (`--input-format stream-json`), `--max-turns`, and session reuse
  are all TODOs.

---

## 4. RL Training

### prime-rl

Large-scale async RL. The key algorithm is **AIPO** (Asynchronous Importance-sampled
Policy Optimization)—a token-level variant of GRPO that handles distribution shift from
off-policy rollouts via importance sampling with clipping ratio δ.

Architecture:
- `max_async_level` (default 2): inference can run up to k steps ahead of the trainer.
  Policy πₙ is trained on rollouts from π_max(0,n-k). Overlap between inference and
  weight broadcast enables continuous execution without idle time.
- Inference: vLLM. Training: FSDP2 (Fully Sharded Data Parallel). Scales to 1000+ GPUs.
- Rollout collection via orchestrator, which submits prompts and collects completions
  from the running inference server.
- Targets SFT → RL → eval as a unified pipeline.

### verifiers (PrimeIntellect)

Provides the reward/environment side that prime-rl consumes. `Rubric` is async-capable:

```python
async def correct_answer(completion, answer) -> float:
    return 1.0 if completion[-1]['content'] == answer else 0.0
rubric = Rubric(funcs=[correct_answer])
```

Trajectory-based rollout tracking (v0.1.8) supports truncated and branching rollouts.
Nano trainer (`vf.RLTrainer`) for lightweight single-machine training.

### rollouts

GRPO with Megatron backend (tensor-parallel, expert-parallel for MoE). On-policy:
rollouts are generated from the current policy checkpoint before each training step.

**Strengths:**
- Megatron TP/EP/SP support for large MoE models (tested: GLM-4.7-Flash, 31.2B, 8x H100).
  prime-rl uses FSDP2, which does not support tensor parallelism at this scale.
- `grpo_train()` API accepts the same `score_fn` and `Environment` types used in evals.
  Same task definition runs as both an eval and a training environment without
  translation.
- `ResourceWatchdog` monitors GPU memory/health during training runs.

**Weaknesses:**
- On-policy only. prime-rl's async off-policy AIPO allows inference and training to run
  concurrently, eliminating the rollout-generation stall that on-policy methods have.
  At scale this is a significant throughput difference.
- No importance sampling / distribution shift correction. If rollout policy and training
  policy diverge (e.g., due to restarts or checkpointing), there is no correction.
- FSDP backend exists (`training/backends/fsdp.py`) but the primary tested path is
  Megatron. prime-rl's FSDP2 path is more thoroughly validated for mid-scale runs
  (single-node 8xH100) and easier to set up.
- vLLM inference server supported, but SGLang is the primary tested backend. prime-rl
  targets vLLM.

---

## 5. Reward / Scoring

### nanoeval

`Solver` produces results; `Task.score()` evaluates them. Async-capable. No
multi-metric scoring built in; you return what you need.

### verifiers

`Rubric` with async scoring functions. Monitor rubrics for automatic metric tracking.
Composable: multiple rubric functions per environment. First-class `pass@k` support.

### rollouts

`Score(metrics=tuple[Metric, ...])` where `Metric(name, value, weight)`. Weighted
average of nonzero-weight metrics gives the reward signal. Supports multi-metric
tracking with zero-weight metrics for observation only.

**Strengths:**
- Multi-metric rewards are first-class: track correctness, token count, latency, etc.,
  with only correctness contributing to the reward.
- Same `Score`/`Metric` types flow through evals, GRPO training, and the eval viewer.
  No translation between eval scoring and training reward.
- `score_fn` can be async, enabling external verification (running test suites, GPU
  benchmarks, calling APIs).

**Weaknesses:**
- No `pass@k` built in. verifiers has it; rollouts requires manual implementation.
- No monitor rubrics / automatic metric collection. verifiers added this in v0.1.9.
- No composable rubric abstraction—rollouts `score_fn` is a raw Python function.
  verifiers' `Rubric(funcs=[...])` is more composable and reusable.

---

## 6. Observability

### nanoeval

py-spy for process profiling, aiomonitor for async task inspection. evallib recorder for
evalboard visualization.

### harbor / terminal-bench

Web viewer bundled in the repo. Container logs per task.

### prime-rl

Weights & Biases integration, Loguru logging, TOML-configured. Per-step metrics in
`metrics.jsonl`. Checkpoint/resume for fault tolerance.

### rollouts

Browser-native eval viewer (`evals_ui/`, React + Python API server). Structured event
logging per sample. Full trajectory JSONL files. Session persistence with append-only
message log. `rollouts monitor --attach <run_id>` TUI for live training. JSONL log
files: `metrics.jsonl`, `training.jsonl`, `rollouts.jsonl` per run.

**Strengths:**
- The eval viewer is the best-developed UI in this comparison. It serves a normalized
  API from `results_adapter.py` and works against any rollouts-style results directory.
- Session branching and handoff: can slice a session at a message index and fork from
  there. No equivalent in any of the external frameworks.
- Training and eval share the same artifact format.

**Weaknesses:**
- Known bug: retry logic can emit multiple `sample_start` events for the same
  `sample_id`. This corrupts per-sample statistics.
- No py-spy or aiomonitor integration. When rollouts hangs, debugging is harder.
- No W&B integration. prime-rl has it; rollouts relies on manual JSONL parsing.

---

## 7. Summary Table

| Dimension | rollouts | nanoeval/alcatraz | harbor/terminal-bench | prime-rl | verifiers |
|-----------|----------|-------------------|-----------------------|----------|-----------|
| Sandboxing | Modal (GPU-native) | Docker (local, secure) | Docker + Daytona/Modal | vLLM subprocess | Python sandbox |
| Fault tolerance | None (restart) | SQLite queue (resume) | None | Checkpoint/resume | None |
| Task catalog | None | None | ~100 terminal tasks | Via verifiers Hub | Growing Hub |
| Verification pattern | Arbitrary score_fn | Arbitrary solver | Script pass/fail | Rubric functions | Rubric functions |
| External agents | Unified drivers + trajectory norm | None | Run-and-check | No | No |
| Multi-agent | Python DSL (OrchestrateEnvironment) | No | No | No | No |
| Terminal interaction | No (file/bash only) | No | Yes (tmux send_keys) | No | No |
| Training algorithm | GRPO (on-policy) | No training | No training | AIPO (async off-policy) | Via prime-rl |
| Training scale | Megatron TP/EP (large MoE) | — | — | FSDP2 (mid-scale) | — |
| pass@k | No | No | No | No | Yes |
| Eval viewer | Browser UI | evalboard | Web viewer | W&B | Server TUI |
| Multi-provider support | Anthropic/OpenAI/Google/SGLang | Anthropic (primarily) | Model-agnostic | vLLM | vLLM/prime inference |

---

## 8. Where to Invest

Based on this comparison, the gaps with the highest leverage:

**Near-term:**
- Fault-tolerant task queue (SQLite or similar). A crashed eval run costing 2-4 hours
  of Modal spend is a real tax. nanoeval's model is simple and portable.
- Formal verification pattern for terminal tasks. The terminal-bench script pass/fail
  model is auditable in a way that arbitrary `score_fn` is not. For KernelBench-style
  tasks, a `run_tests()` → pass/fail path is already partially built in
  `terminal_bench.py`; the pattern should be generalized.
- Fix the `sample_start` duplicate event bug in `native.py`.

**Medium-term:**
- Worker/Runtime/Capabilities split in `OrchestrateEnvironment`. The current `kind`
  string conflation is the main legibility problem in the multi-agent path.
- `pass@k` support. One function; high value for capability evals.
- Async off-policy training (AIPO-style). The on-policy stall is the primary throughput
  bottleneck at scale. prime-rl's approach (importance sampling + clipping) is the
  right model.

**Longer-term:**
- Docker/local sandbox option for fast iteration without Modal.
- tmux terminal interaction primitive for tasks that require live terminal simulation.
- W&B or equivalent integration for training runs.
