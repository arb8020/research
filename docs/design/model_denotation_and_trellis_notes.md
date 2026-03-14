## Model denotation and Trellis notes

### Why this note exists

We want a stable on-disk reference for two linked ideas:

1. our internal model semantics should be explicit and lowered into backend-native
   shapes, analogous to `RealizationPlan -> backend lowering`
2. the Workshop Labs "Post-Training 50x Faster" post is a useful witness for why
   model semantics, checkpoint semantics, and runtime strategy must not be
   collapsed together

Related draft code:
- `rollouts/rollouts/training/models/denotation.py`
- `rollouts/rollouts/training/models/adapters.py`
- `rollouts/rollouts/training/backends/megatron/model.py`

Local extracted article artifacts:
- `/tmp/workshoplabs_post_training_50x_faster.html`
- `/tmp/workshoplabs_post_training_50x_faster.json`
- `/tmp/workshoplabs_post_training_50x_faster.md`

Source article:
- https://www.workshoplabs.ai/blog/post-training-50x-faster

### What Trellis appears to use as source of truth

The article does not present one formal internal IR. The implied source of truth
is a combination of:

- explicit Kimi architecture facts
  - 61 layers
  - 384 experts per layer
  - shared expert
  - MLA attention
  - expert INT4 quantization
  - bf16 for non-expert weights
- HuggingFace's Kimi modeling file as a reference implementation of the forward
  semantics
- output validation against the Moonshot API
- intermediate activation comparison against the HuggingFace forward pass

Important detail: they say they started from PrimeRL for high-level structure,
then removed support for other models and implemented Kimi-K2-Thinking from
scratch. That is the opposite of treating generic backend/framework defaults as
the source of truth.

The article therefore suggests:

- external publication truth can start in HuggingFace/Transformers
- but practical training truth often requires a normalized internal model view
- and backend/runtime realization must be explicit rather than inferred from a
  generic stack

### What we should learn for rollouts

We should separate at least three things:

1. Model denotation
   - backend-neutral architecture semantics
   - checkpoint/load semantics
   - model-family identity

2. Backend model lowering
   - `ModelDenotation -> MegatronModelLowering`
   - `ModelDenotation -> TorchTitanModelLowering`
   - explicit support / rejection / custom adapter routing

3. Backend-native execution
   - Megatron/TorchTitan/etc execute their own runtime semantics

This mirrors our current layout story:

- `RealizationPlan` is denotational truth for layout intent
- backend lowering derives coarse partition/runtime intent
- backend executes natively

We want the same semantic honesty for models:

- `ModelDenotation` is denotational truth for model/checkpoint semantics
- backend model lowering derives provider/spec/checkpoint strategy
- backend executes natively

### What this means concretely

We should not keep smearing model truth across:

- HF config objects
- bridge defaults
- backend-native layer-spec defaults
- checkpoint conversion/loading code
- trainer code

Instead:

- normalize HF-published model semantics into our own product type
- explicitly lower into backend-native model construction
- reject unsupported combinations early

### Draft denotation shape

The current draft in `training/models/denotation.py` models:

- `HFModelSource`
- `RotarySemantics`
- `MoESemantics`
- `ModelArchitectureSemantics`
- `CheckpointSemantics`
- `ModelDenotation`

This draft is intentionally narrow. It does not yet model:

- forward/loss semantics
- optimizer/runtime policy
- layout/partition semantics

### Trellis-specific implementation lessons

The article reinforces several concrete lessons:

- "supported" must mean end-to-end semantics, not partial weight updates
- MoE expert ownership and communication semantics must be explicit
- skipping pad tokens and packing are semantic/runtime choices that matter
- quantization format is part of training reality, not incidental detail
- performance work only composes once model/runtime boundaries are honest

### Known open questions

- How much of our `ModelDenotation` should be derived from HF config vs
  explicitly declared?
- When do we require a custom adapter instead of a generic backend lowering?
- How do we eventually represent forward/loss semantics without over-claiming
  backend parity?
- Should converted checkpoints be first-class in `CheckpointSemantics` for
  model families where HF-native runtime loading is not honest?
