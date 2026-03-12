"""SWE-grep eval config: grep/glob/read only (classic SWE-grep)."""

from pathlib import Path

from rollouts.core import Endpoint

from .eval_swe_grep import SWEGrepConfig

# Classic SWE-grep toolset (no semantic search)
config = SWEGrepConfig(
    corpus_path=Path("/path/to/corpus"),
    questions_path=Path(__file__).parent / "sample_questions.jsonl",
    agent_endpoint=Endpoint.from_legacy(provider="anthropic", model="claude-sonnet-4-5-20250929"),
    grader_endpoint=Endpoint.from_legacy(provider="anthropic", model="claude-sonnet-4-5-20250929"),
    tools=["grep", "glob", "read", "submit"],  # No search
    search_backend=None,
    max_turns=15,
    max_samples=None,
)
