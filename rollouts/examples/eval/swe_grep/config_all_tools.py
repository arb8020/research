"""SWE-grep eval config: All tools enabled.

NOTE: If using search tool, you must provide a search_fn implementation.
See config_search_only.py for an example.
"""

from pathlib import Path

from rollouts.dtypes import Endpoint

from .eval_swe_grep import SWEGrepConfig

# All tools: grep + glob + search + read
# NOTE: search_fn must be provided by caller if using search tool
config = SWEGrepConfig(
    corpus_path=Path("/path/to/corpus"),
    questions_path=Path(__file__).parent / "sample_questions.jsonl",
    agent_endpoint=Endpoint.from_legacy(provider="anthropic", model="claude-sonnet-4-5-20250929"),
    grader_endpoint=Endpoint.from_legacy(provider="anthropic", model="claude-sonnet-4-5-20250929"),
    tools=["grep", "glob", "search", "read", "submit"],  # Everything
    search_fn=None,  # Must be provided by caller
    max_turns=20,
    max_samples=None,
)
