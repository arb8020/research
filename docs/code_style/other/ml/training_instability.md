# Training Instability

Training instability is often described as black magic: vibe out hyperparameters, run giant sweeps, hope something works.

That framing is too vague to be useful.

The better model is:

- training is a dynamical system
- hyperparameters are control parameters for that system
- "instability" is usually a hidden-state problem before it is a mystery

If we want to debug and improve training rationally, we should make the important state explicit, log it early, and talk about failures in terms of mechanism rather than folklore.

## Core Orientation

Do not treat "training instability" as one thing.

Separate at least these cases:

### 1. Optimization Instability

Examples:
- learning rate too high
- warmup too short
- bad optimizer hyperparameters
- batch-size changes without retuning
- update magnitudes too large relative to parameter scale

### 2. Numerical Instability

Examples:
- overflow / underflow
- fp16 / bf16 precision issues
- unstable fused kernels
- bad reduction order or accumulation policy
- NaNs or infs appearing after backward or optimizer step

### 3. Data / Objective Instability

Examples:
- outlier batches
- reward scale drift
- noisy or inconsistent labels
- distribution shift
- curriculum/order changes that move the training regime

### 4. Systems-Induced Instability

Examples:
- stale async samples
- nondeterministic kernels
- bad checkpoint restore
- sharding / sync mismatches
- hidden launch/runtime differences between runs

If these categories are not separated, people start cargo-culting hyperparameters when the actual problem is data, systems, or numerics.

## What "Instability" Actually Means

A run is unstable when small changes in initialization, seed, data order, precision, or control parameters produce qualitatively different optimization behavior.

Typical symptoms:

- loss spikes or diverges
- gradients explode or vanish
- activations leave their expected scale regime
- optimizer state drifts into a bad regime
- training becomes highly seed-sensitive
- apparently good runs are not reproducible

Those symptoms are effects. The useful question is:

- what hidden state changed first?

## Why It Gets Mythologized

A lot of teams only log:

- train loss
- validation loss
- maybe gradient norm

That is not enough to understand the system.

The latent state that often matters includes:

- activation mean/std/norm by layer or block
- gradient norms by parameter group
- parameter norms by parameter group
- update norms by parameter group
- update-to-weight ratio
- logit scale / entropy / saturation
- overflow / skipped-step counts
- clipping frequency
- reward / advantage scale in RL
- stale-sample age / version lag in async systems

Without this, sweeps feel like magic because the run's internal state is mostly invisible.

## Better Language

Avoid:

- "the run is cursed"
- "this model doesn't like 1e-4"
- "training is unstable"

Prefer:

- the first bad signal is activation norm blow-up in layers 18-24
- forward is stable, but the first backward produces inf gradients
- update-to-weight ratio is too high after warmup
- the run becomes unstable only after increasing global batch 8x
- this is reward-scale drift, not optimizer divergence
- this is stale-sample-induced instability from async lag, not a pure hyperparameter issue

This naming matters because it points to mechanism instead of mysticism.

## A Minimal State Surface to Log

At minimum, log:

- loss
- learning rate
- grad norm
- param norm
- update norm
- update norm / param norm
- overflow / skipped-step count
- tokens/sec and step time

For transformer-like models, also try to log:

- activation norm/mean/std for major blocks
- attention/logit stats
- residual stream scale if relevant
- layerwise gradient norm summaries

For RL, also log:

- reward mean/std
- advantage mean/std
- policy entropy
- KL to reference / previous policy
- value loss / critic error
- stale-sample version lag
- group completion / drop rate

The principle:

- do not only log outcomes
- log the internal state that explains the outcomes

## Debugging Ladder

When a run is unstable, answer these questions in order:

### 1. When is the first bad signal?

Examples:
- before first step
- first backward
- first optimizer step
- after warmup
- after LR peak
- after checkpoint restore
- after a data mixture change

### 2. What variable first leaves its expected regime?

Examples:
- activation norm
- gradient norm
- update ratio
- reward scale
- entropy
- value error
- NaN/inf counters

### 3. What changed recently?

Examples:
- learning rate
- batch size
- precision mode
- kernel implementation
- sharding setup
- data mixture
- reward function
- async topology

This is a much better debugging process than random hyperparameter movement.

## Hyperparameters as Control Parameters

Hyperparameters are not magic numbers. They are control parameters for a dynamical system.

Examples:

- learning rate controls update magnitude
- warmup controls startup transients
- weight decay controls drift
- batch size changes gradient noise scale
- clip thresholds define saturation boundaries
- PPO/GRPO-style coefficients change the geometry of policy updates

Good tuning looks like:

- predict what state variable should move
- change one mechanism at a time
- check whether the observed state changed the way you expected

Bad tuning looks like:

- make many changes at once
- observe only final loss
- infer folklore from noisy runs

## RL-Specific Note

RL is especially unstable because the data distribution moves with the policy.

That means instability can come from:

- reward scale drift
- bad advantage normalization
- critic collapse
- policy entropy collapse
- stale async samples
- insufficient or excessive KL pressure

Do not describe all of this as "RL is just unstable". The point is to identify which moving target moved first.

## Separate Mechanism from Search

Sweeps are not inherently bad. They become wasteful when they replace modeling.

Useful sweep:

- vary one control parameter over a narrow range
- keep the rest fixed
- inspect internal state metrics, not just final score

Wasteful sweep:

- large uncontrolled grid over many knobs
- weak observability
- no attempt to identify first bad signal

Search is most valuable after you have a decent mechanistic model of the regime.

## Separate Infrastructure Failure from Training Failure

If the run is producing NaNs because a fused kernel is wrong, that is not a pure optimization problem.

If async lag produces stale targets, that is not "bad learning rate".

If the training job is silently using different precision or transport settings between runs, that is not a clean hyperparameter comparison.

Always separate:

- infrastructure / runtime problems
- numerical problems
- optimization problems
- data / objective problems

Otherwise the tuning loop becomes incoherent.

## Design Principle for Future Work

Before tuning aggressively:

1. define the intended regime
2. make the important state visible
3. build a small trustworthy baseline
4. only then increase complexity or search surface

The goal is not to remove all uncertainty. The goal is to make the uncertainty legible.

## PL / Type / Design Vocabulary for This Domain

Useful words:

- `state machine`: training passes through regimes and transitions
- `invariant`: conditions that must hold for a valid run
- `effect`: random seeds, kernels, distributed runtime, IO, checkpoint restore
- `normalization`: turning messy runtime/config state into one trusted setup
- `local reasoning`: being able to tell why a run failed without guessing across ten layers

This vocabulary matters because unstable training is often discussed too vaguely to support real debugging.

## A Good Default Principle

Treat training instability as a systems-identification problem:

- find the first bad signal
- find the hidden state that moved
- connect it to a mechanism
- then tune or redesign from there

