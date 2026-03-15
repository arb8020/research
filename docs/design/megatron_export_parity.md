# Megatron Export Parity

This note scopes the `miles/slime` Megatron adapter subgraph we still need for
honest Megatron -> SGLang hot weight sync.

It is intentionally narrower than "all of megatron_utils". The active boundary
under pressure is the runtime export/update slice, not the whole train-actor
stack.

## Semantic Slices

### 1. Model/bootstrap slice

`miles/slime` files:

- `backends/megatron_utils/arguments.py`
- `backends/megatron_utils/model_provider.py`
- `backends/megatron_utils/model.py`
- `backends/megatron_utils/initialize.py`

Our equivalents:

- [`model.py`](/Users/chiraagbalu/research/rollouts/rollouts/training/backends/megatron/model.py)
- [`initialize.py`](/Users/chiraagbalu/research/rollouts/rollouts/training/backends/megatron/initialize.py)

Status:

- mostly present for plain Qwen3
- still less complete for some custom families

### 2. Train-actor/runtime slice

`miles/slime` files:

- `backends/megatron_utils/actor.py`

Our rough equivalents:

- [`remote_backend.py`](/Users/chiraagbalu/research/rollouts/rollouts/training/backends/megatron/remote_backend.py)
- [`megatron_worker.py`](/Users/chiraagbalu/research/rollouts/rollouts/training/megatron_worker.py)

Status:

- semantically different implementation, but the same ownership boundary
- do not cargo-cult this whole slice unless our current worker/runtime shape
  becomes the bottleneck

### 3. Runtime export/update slice

`miles/slime` files:

- `backends/megatron_utils/update_weight/common.py`
- `backends/megatron_utils/update_weight/hf_weight_iterator_base.py`
- `backends/megatron_utils/update_weight/hf_weight_iterator_direct.py`
- `backends/megatron_utils/update_weight/update_weight_from_tensor.py`
- `backends/megatron_utils/update_weight/update_weight_from_distributed.py`
- `backends/megatron_utils/megatron_to_hf/*`
- `backends/megatron_utils/sglang.py`

Our current partial equivalents:

- [`export_runtime.py`](/Users/chiraagbalu/research/rollouts/rollouts/training/backends/megatron/export_runtime.py)
- [`inference_export.py`](/Users/chiraagbalu/research/rollouts/rollouts/training/backends/megatron/inference_export.py)
- [`weight_conversion/__init__.py`](/Users/chiraagbalu/research/rollouts/rollouts/training/backends/megatron/weight_conversion/__init__.py)
- [`weight_conversion/qwen2.py`](/Users/chiraagbalu/research/rollouts/rollouts/training/backends/megatron/weight_conversion/qwen2.py)

Status:

- this is the active missing subgraph
- current live failures are revealing holes here, not in transport

## Current Gaps

### Global param denotation

`miles/slime` define a real global Megatron param product type:

- stable global names across PP/EP/virtual PP
- `ParamInfo`
- source-rank ownership
- TP metadata (`partition_dim`, `partition_stride`)

We currently only approximate this in [`export_runtime.py`](/Users/chiraagbalu/research/rollouts/rollouts/training/backends/megatron/export_runtime.py).

### Full reconstruction before conversion

`miles/slime` do:

- PP broadcast
- EP broadcast
- TP gather
- stride-aware TP gather
- bucketization

before Megatron -> HF conversion.

We currently:

- only do TP gather
- only partially handle stride-aware gather
- do not yet model the broader `ParamInfo`/bucket path

### Explicit iterator boundary

`miles/slime` do not treat raw `state_dict()` as the inference export product.

They define an iterator boundary:

- Megatron local weights
- -> full Megatron params
- -> HF named tensors
- -> serialized update buckets

We currently collapse too much of that into a thinner
[`build_megatron_hf_tensors_from_runtime`](/Users/chiraagbalu/research/rollouts/rollouts/training/backends/megatron/export_runtime.py)
path.

### SGLang compatibility shim

`miles/slime` centralize SGLang update-weight compat imports in `sglang.py`.

We currently rely on the remote image already having the right patched SGLang,
but we do not yet have an equivalent local shim boundary in our Megatron
backend.

## Runtime Patches

### SGLang patches

Their patched SGLang adds or fixes:

- `update_weights_from_tensor`
- `update_weights_from_distributed`
- scheduler/model-runner routing for update calls
- tensor-bucket support
- some rollout-specific request/update API surface

Repo-local adapter consequence:

- `miles/slime` still hide these version-specific imports behind a local
  `megatron_utils/sglang.py` shim
- we should do the same even while the remote image already carries the patch,
  so our backend code has one honest SGLang compatibility boundary

These matter for our eventual "own pinned image" story. They are already
present in the current remote `miles` image, so they are not the active code
bug in our repo today.

### Megatron patches

Their patched Megatron adds model-surface hooks like:

- `post_self_attn_layernorm`
- `post_mlp_layernorm`

and some runtime compatibility helpers.

These matter for full parity and some custom families, but they are not the
current blocker for plain Qwen3 dense export.

### Current image dependence

Today our remote Modal path still runs inside the `miles` image, so the
upstream `sglang.patch` / `megatron.patch` layer is already present at runtime.

That means:

- the active blocker in our repo is the repo-local export/update subgraph
- but once we switch to our own pinned image, we will need to carry the runtime
  patch inventory explicitly rather than inheriting it from `miles`

## Next Ports

1. Port a real `common.py`-style boundary into our backend-local export layer.
   This should own:
   - global Megatron naming
   - `ParamInfo`
   - source-rank metadata
   - stride-aware TP gather

2. Port a real iterator boundary, equivalent to
   `hf_weight_iterator_direct.py`.

3. Keep using our existing model/init path for now.
   Do not port the whole `actor.py` stack unless our current worker/runtime
   shape proves dishonest.

4. Add offline export validation for the witness model.
   The witness run should not be the first place we discover missing export
   cases.

## Immediate Known Bug

For `Qwen3Config`, HF publishes grouped-query semantics as
`num_key_value_heads`, not `num_query_groups`.

Our runtime export conversion context must normalize that field before calling
the `qwen2` Megatron -> HF converter, otherwise `linear_qkv.weight` reshaping is
dishonest and fails even when the converter logic itself is correct.
