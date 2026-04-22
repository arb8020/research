# Reference artifacts

`devin_pr_review_sample.json` (551 KB) — decoded JSON response from
`app.devin.ai/api/pr-review/job-result/<id>`, captured via HAR while
reviewing PR #2 of arb8020/research. Not shipped with the tool; used
only as a schema/fidelity reference while styling.

Top-level keys:
- `pr_metadata`
- `sections[]` — LLM-grouped changes with `title`, `text`, `changes[]`
- `file_contents_at_base{}` — full files at base ref, keyed by path
- `lifeguard_result.bugs[]` — anchored review comments, severity-tagged
- `lifeguard_result.analyses[]` — softer observations
- `special_files_content` — AGENTS.md / CLAUDE.md text pulled as context

This is the data Devin's frontend hydrates into its three-column review
UI. We do not adopt this shape (agent-blame's semantic payload is line
attribution, not grouped diffs + bugs), but the rendering primitives
Devin uses on top of this shape are what we're copying stylistically.
