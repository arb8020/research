"""SWE-grep eval config: All tools enabled."""

from pathlib import Path

from rollouts.dtypes import Endpoint

from .eval_swe_grep import SWEGrepConfig

# All tools: grep + glob + search + read
config = SWEGrepConfig(
    corpus_path=Path("/path/to/corpus"),
    questions_path=Path(__file__).parent / "sample_questions.jsonl",
    agent_endpoint=Endpoint(provider="anthropic", model="claude-sonnet-4-5-20250929"),
    grader_endpoint=Endpoint(provider="anthropic", model="claude-sonnet-4-5-20250929"),
    tools=["grep", "glob", "search", "read", "submit"],  # Everything
    search_backend="wafer",
    search_config={
        "api_url": "https://wafer.api/search",
        "api_key": "your-api-key",
    },
    max_turns=20,
    max_samples=None,
)
