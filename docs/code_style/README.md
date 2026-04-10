# Code Style Guide

## Start Here By Task

- **Designing a new boundary or product type**
  - [earned_compression.md](earned_compression.md) - Use this when you are deciding whether an abstraction is real or premature.
  - [codex_codestyle_interview.md](codex_codestyle_interview.md) - Use this when you need honest product types, sum types, and boundary normalization.
  - [cheatsheet.md](cheatsheet.md) - Use this for quick local reminders while designing.

- **Designing system/session architecture**
  - [sean_goedecke_system_design.md](sean_goedecke_system_design.md) - Use this when deciding service boundaries, state ownership, queues, and background work.
  - [code_philosophy_essay.md](code_philosophy_essay.md) - Use this for the overall workflow: usage code, testing, refactoring.
  - [other/templates/DESIGN_TEMPLATE.md](other/templates/DESIGN_TEMPLATE.md) - Use this when you want to crystallize the design into a written spec.

- **Debugging a bug or failure**
  - [debugging_draft.md](debugging_draft.md) - Use this to find the first dishonest boundary or violated invariant.
  - [grugbrain_testing.md](grugbrain_testing.md) - Use this to choose a stable integration/regression test cut point.
  - [cheatsheet.md](cheatsheet.md) - Use this as a quick reminder for assertions, control flow, and error handling.

- **LLM/agent coding workflow**
  - [keeping_llm_code_honest.md](keeping_llm_code_honest.md) - Use this for verification and compression-oriented agent workflow.
  - [llm_coding_workflow.md](llm_coding_workflow.md) - Use this for practical LLM collaboration notes.
  - [code_philosophy_essay.md](code_philosophy_essay.md) - Use this when you want the full workflow in one place.

## Suggested Reading Order

- Start with [cheatsheet.md](cheatsheet.md) for a quick working-memory refresh.
- Then read [earned_compression.md](earned_compression.md) to calibrate abstraction pressure.
- Then pick one task-specific doc from the sections above.
- Only go deeper into the extended references if the current docs do not answer the design question.

## Working Set

- **[cheatsheet.md](cheatsheet.md)** - Comprehensive synthesis: all principles with examples, organized by topic
- **[favorites.md](favorites.md)** - Short daily-use reference: error handling decision tree, classes vs functions
- **[earned_compression.md](earned_compression.md)** - When abstraction is earned: semantic compression vs dishonest compression
- **[keeping_llm_code_honest.md](keeping_llm_code_honest.md)** - LLM workflow: compression over working code, verification loops
- **[llm_coding_workflow.md](llm_coding_workflow.md)** - LLM workflow notes
- **[debugging_draft.md](debugging_draft.md)** - Draft debugging workflow: violated invariants, boundary normalization, honest failure
- **[code_philosophy_essay.md](code_philosophy_essay.md)** - Full philosophy writeup
- **[code_philosophy_reference.md](code_philosophy_reference.md)** - Reference quotes and citations

**Casey Muratori:** [semantic_compression](casey_semantic_compression.md), [granularity](casey_granularity.md), [worst_api](casey_worst_api.md)

**Other:** [tiger_style](tiger_style.md), [sean_goedecke_system_design](sean_goedecke_system_design.md), [simon_willison_code_review](simon_willison_code_review.md)

**Testing/Debugging:** [grugbrain_testing](grugbrain_testing.md), [logging_sucks](logging_sucks.md), [mcoding_logging_dense](mcoding_logging_dense.md), [why_not_coverage](why_not_coverage.md)

**ML/Experiments:** [nmoe_experiment_tracking](nmoe_experiment_tracking.md)

---

## Other

Extended references organized by topic:

- `other/error-handling/` — ERROR_HANDLING.md (full guide), my_notes.md (tuple return patterns), errors.md (links)
- `other/classes-functions/` — CLASSES_VS_FUNCTIONAL.md, IMMUTABILITY_AND_FP.md
- `other/logging/` — mcoding_logging.md (dictConfig/JSON/QueueHandler), LOGGING_LEVELS_RECOMMENDATIONS.md
- `other/api-design/` — code_reuse_casey_muratori.md, sean_goedecke_good_api_design.md, codeaesthetic_abstraction_coupling.md
- `other/testing/` — slack_coverage_discussion.md
- `other/review/` — ezyang_code_review_alignment.md, sean_goedecke_code_review.md
- `other/systems/` — joe_duffy_safe_native_code.md, anyio_advice.md, multiprocessing_heinrich.md
- `other/ml/` — experiment_config.md, shape_suffixes_shazeer.md
- `other/frontend/` — react patterns, design layers, HTML tools, frontend skill
- `other/llm/` — llm_code_cleanup.md
- `other/templates/` — DESIGN_TEMPLATE.md, github_ideal_commits.md, ray_design.txt
- `other/raw/` — raw source transcripts and unprocessed notes
