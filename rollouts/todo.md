https://mariozechner.at/posts/2025-11-30-pi-coding-agent/#toc_3

https://github.com/letta-ai/letta-code.git

[ ] vim input with ctrlG is fragile


## Evaluation

[ ] Scaffold versioning - add SCAFFOLD_VERSION constant, include in EvalReport (agents.py:48)
[x] Separate provider errors from sample failures in accuracy calc (evaluation.py:233, 562)
[ ] Document temperature choice for evals (dtypes.py:1066)
[ ] Refactor Sample - remove legacy input/ground_truth fields (training/types.py:94)

## Bifrost v2 Migration

[x] Migrate verify.py to bifrost v2 API (tools/functional_extractor/verify.py)
[x] Migrate debug_swa.py to bifrost v2 API (tools/functional_extractor/debug_swa.py)  
[x] Migrate test_inference_correctness.py to bifrost v2 API

## Providers

[ ] Add doom loop detection - Anthropic (providers/anthropic.py:265)
[ ] Add doom loop detection - OpenAI (providers/openai_completions.py:303)
[ ] Detect truncated responses (providers/anthropic.py:488, openai_completions.py:307)
[x] Define retriable vs non-retriable error classification (providers/base.py - ProviderError class)

## Ideas from nmoe / miles comparison

[ ] Config fingerprinting - see docs/design/config_fingerprinting.md
[ ] Functional training refactor - see docs/design/functional_training_refactor.md

[ ] tui mode for training job has hella lag on sglang pane

[ ] make slice/dice easy for evaluations

[x] context handoff
[x] aborts
[x] structured split tool results

[~] TUI (terminal UI with differential rendering)
    - [x] Core engine ported (frontends/tui/)
    - [ ] can't scroll up, top of stuff disappears? 
    - [ ] slash commands
    - [ ] make sure codex works w responses
    - [ ] Wire to agent StreamEvents (see frontends/tui/INTEGRATION.md)
    - [ ] rounded theme doesnt look right
    - [ ] limit tool calls
    - [ ] ctx management/compaction
    - [ ] show sys message
    - [ ] sanitize broken sessions (remove last k ?)



