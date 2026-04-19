"""banking_knowledge smoke: 1 task with the default BM25 retrieval variant.

Verifies the banking_knowledge wiring end-to-end: domain env construction
with a retrieval pipeline, RAG-aware tools surface to the agent, and
scoring works against tau2's evaluator.

The default variant is `bm25` (pure CPU, no embeddings dependencies).
For other variants (e.g. `qwen_embeddings_grep`) install tau2 with the
knowledge extra and pass `retrieval_variant=...` to build_sample_rows.
"""

from rollouts.config.tiers import EndpointConfig, OutputConfig, RunConfig
from tau2_v0.prepare import build_sample_rows

endpoint = EndpointConfig(model="claude-sonnet-4-5-20250929", provider="anthropic")
run = RunConfig(max_turns=40, max_concurrent=1, verbose=False, show_progress=True)
output = OutputConfig(experiment_name="tau2_v0_banking_smoke")

tasks_override = build_sample_rows(
    domain="banking_knowledge",
    limit=1,
    # retrieval_variant=None → tau2 uses DEFAULT_RETRIEVAL_VARIANT (bm25).
)
