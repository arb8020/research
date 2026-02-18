# Decisions

Key decisions and rationale.

## 2024-02-14: Logit Diff Amplification Approach

**Decision**: Implement Goodfire's model diff amplification technique

**Rationale**:
- Paper shows 10-100x increase in surfacing rare behaviors
- Can find backdoor behaviors without knowing the trigger
- Need to identify base model to compute diff

**Base model candidates for warmup (8B Qwen2)**:
- `Qwen/Qwen2-7B-Instruct` - likely base (8B params despite name)
- `Qwen/Qwen2-7B` - base without instruction tuning
- Could also try Qwen2.5 variants

**Implementation**:
- `src/logit_diff.py` - core amplification logic
- `scripts/logit_diff_probe.py` - Modal runner
- Need A100 (80GB) for 2x 8B models in bf16

## 2024-02-14: Tool Setup

**Decision**: Use Modal for warmup model (8B), JS API for full models (671B)

**Rationale**:
- Warmup model fits on A10G (24GB VRAM)
- Full models are 671B params - need distributed inference
- JS API provides hosted access to full models
- Modal gives us weight access, custom code execution for warmup

**Alternatives considered**:
- Running full models on Modal multi-GPU: expensive, complex
- Only using JS API: can't inspect weights/activations locally for warmup

## 2024-02-14: Workspace Structure

**Decision**: Separate claude-manual/ from eval code

**Rationale**:
- Pair programming workspace needs different structure than eval harness
- Notes/progress tracking for session continuity
- Eventually may want to turn successful manual exploration into eval
