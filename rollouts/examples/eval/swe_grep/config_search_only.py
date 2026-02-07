"""SWE-grep eval config: semantic search + read only.

NOTE: You must provide a search_fn implementation. Example:

    async def my_search(query: str, top_k: int) -> str:
        # Call your search API
        results = await my_api.search(query, top_k)
        # Format results
        lines = [f"Found {len(results)} results:"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['file']} (score: {r['score']:.3f})")
            lines.append(f"   {r['excerpt'][:200]}")
        return "\\n".join(lines)

    config.search_fn = my_search
"""

from pathlib import Path

from rollouts.dtypes import Endpoint

from .eval_swe_grep import SWEGrepConfig

# Semantic search only (like SID-1 / RLM)
# NOTE: search_fn must be provided by caller
config = SWEGrepConfig(
    corpus_path=Path("/path/to/corpus"),
    questions_path=Path(__file__).parent / "sample_questions.jsonl",
    agent_endpoint=Endpoint(provider="anthropic", model="claude-sonnet-4-5-20250929"),
    grader_endpoint=Endpoint(provider="anthropic", model="claude-sonnet-4-5-20250929"),
    tools=["search", "read", "submit"],  # Only semantic search
    search_fn=None,  # Must be provided by caller
    max_turns=10,
    max_samples=None,
)
