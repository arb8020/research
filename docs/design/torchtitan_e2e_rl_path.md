# TorchTitan end-to-end RL path

This is the shortest path from the current refactor state to a real RL run with
the TorchTitan backend.

The goal is not "all distributed features at once".
The goal is one honest dense RL run that uses the new training contract and the
TorchTitan lowering.

## Scope for the first real run

- dense model
- full-weight training
- single node
- minimal parallel provisioning
- contract-native RL loop
- explicit weight publication/version semantics
- standard GRPO-style objective, not OPD teacher distillation

This is enough to prove:

- the outer training contract is real
- the TorchTitan lowering is usable
- the RL path can actually step and publish weights

## What not to include yet

- MoE
- EP
- low-precision expert kernels
- richer realization IR execution
- elastic orchestration
- modal/runpod deployment concerns

Those are second-wave concerns.

## Proposed path

### Phase 1: Dense supervised on TorchTitan

Use the contract-native supervised path to prove:

- `TrainingDatum`
- `ForwardProducts`
- `StepResult`
- `TrainableParameterPolicy(full_weight)`
- `TorchTitanLowering.from_realization(...)`

This should be the first "does the backend actually step?" proof.

### Phase 2: Dense RL on TorchTitan

Use the contract-native RL loop to prove:

- dense RL objective semantics
- full-weight update path
- explicit `TrainingRuntimeState`
- explicit `WeightVersion`
- explicit `WeightPublication`

This is the first actual end-to-end RL proof.

### Phase 3: Only then widen

After Phase 2 is clean:

- consider DP > 1
- consider richer backend-specific lowering/provisioning objects
- consider MoE / low precision
- consider stronger realization semantics

## Immediate implementation target

The next code should focus on:

1. one dense supervised run configuration for `TorchTitanBackend`
2. one dense RL run configuration for `TorchTitanBackend`
3. making sure both use the contract-native loops rather than legacy dict paths

## Acceptance standard

We should say TorchTitan is "working" for this refactor only if:

- dense supervised runs through the new contract path
- dense RL runs through the new contract path
- weight publication/versioning is explicit in RL
- no part of the core loop needs to think in TorchTitan-native ontology
