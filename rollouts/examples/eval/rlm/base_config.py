"""RLM evaluation base config.

Shared configuration for Recursive Language Model evaluations.
These benchmarks test RLM's ability to process unbounded context by:
- Storing context as a Python variable (not in messages)
- Exploring via REPL code execution
- Making recursive LLM calls for semantic processing

Reference: https://alexzhang13.github.io/blog/2025/rlm/
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from rollouts.dtypes import Endpoint, Metric, Score

logger = logging.getLogger(__name__)

# ──────────────────────── System Prompts ────────────────────────────────────

RLM_TOOL_SYSTEM_PROMPT = """You are an assistant with access to a REPL environment for processing large contexts.

The input context is stored in a Python variable called `context`. It may be very large (millions of characters). You NEVER see the full context in your messages - instead, you explore it programmatically.

## Available Tools

### repl
Execute Python code to explore and process the context:
- `context` - the full input text as a string
- `len(context)` - get the size
- `context[:1000]` - peek at the beginning
- `context.split('\\n')` - split into lines
- `re.findall(pattern, context)` - search with regex
- `[line for line in context.split('\\n') if 'keyword' in line]` - filter lines

The `re` module is available for regex operations.

### llm_query
Query a language model for semantic tasks on chunks of context:
- Use for classification, extraction, summarization of text chunks
- The prompt should be self-contained (include the text to process)
- Example: `llm_query("Extract the main topic from: " + chunk)`

### final_answer
Submit your final answer when you've solved the task.

## Strategy

1. **Peek first**: Start by examining the context structure
2. **Search strategically**: Use regex or string matching to narrow down
3. **Chunk for semantics**: When you need understanding, extract chunks and use llm_query
4. **Build incrementally**: Store intermediate results in variables
5. **Answer when confident**: Use final_answer only when you have the answer"""

BASELINE_SYSTEM_PROMPT = """You are a helpful assistant. Answer the question based on the provided context.

Be thorough and precise in your answer."""


# ──────────────────────── Config Dataclasses ────────────────────────────────


@dataclass(frozen=True)
class EndpointConfig:
    """Main model endpoint configuration."""

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-5-20250929"
    base_url: str | None = None
    api_key: str | None = None
    max_tokens: int = 4096


@dataclass(frozen=True)
class SubEndpointConfig:
    """Sub-model endpoint for llm_query calls (can be smaller/cheaper)."""

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-5-20250929"  # Could use haiku for cost
    base_url: str | None = None
    api_key: str | None = None
    max_tokens: int = 2048


@dataclass(frozen=True)
class RLMConfig:
    """RLM-specific configuration."""

    enabled: bool = True  # If False, run baseline (full context in messages)
    recursive: bool = False  # If True, llm_query spawns RLM, not just LM
    max_depth: int = 2  # Maximum recursion depth
    use_tool_calling: bool = True  # True: tool-based, False: message-parsing


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset configuration."""

    max_samples: int | None = None
    seed: int = 42
    # Subclass-specific fields added in each eval


@dataclass(frozen=True)
class EvalRunConfig:
    """Evaluation run configuration."""

    max_concurrent: int = 4
    max_turns: int = 20  # RLM may need many REPL iterations
    verbose: bool = True


@dataclass(frozen=True)
class OutputConfig:
    """Output configuration."""

    save_dir: Path = field(default_factory=lambda: Path("results"))
    experiment_name: str = "rlm_eval"
    run_id: str | None = None
    save_json: bool = True
    save_jsonl: bool = True

    def __post_init__(self) -> None:
        if self.run_id is None:
            object.__setattr__(self, "run_id", datetime.now().strftime("%Y%m%d-%H%M%S"))

    @property
    def output_dir(self) -> Path:
        return self.save_dir / f"{self.experiment_name}_{self.run_id}"


@dataclass(frozen=True)
class RLMEvalConfig:
    """Base RLM evaluation config."""

    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    sub_endpoint: SubEndpointConfig = field(default_factory=SubEndpointConfig)
    rlm: RLMConfig = field(default_factory=RLMConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    run: EvalRunConfig = field(default_factory=EvalRunConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        # Redact API keys
        if data["endpoint"].get("api_key"):
            data["endpoint"]["api_key"] = "***REDACTED***"
        if data["sub_endpoint"].get("api_key"):
            data["sub_endpoint"]["api_key"] = "***REDACTED***"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


# ──────────────────────── Endpoint Helpers ───────────────────────────────────


def get_endpoint(config: EndpointConfig) -> Endpoint:
    """Create Endpoint from config, loading API key from env if needed."""
    api_key = config.api_key
    if not api_key:
        env_var_map = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
        env_var = env_var_map.get(config.provider)
        if env_var:
            api_key = os.getenv(env_var)
            if not api_key:
                raise ValueError(f"{env_var} not set")

    return Endpoint(
        provider=config.provider,
        model=config.model,
        api_base=config.base_url or "",
        api_key=api_key or "",
        max_tokens=config.max_tokens,
    )


def get_sub_endpoint(config: SubEndpointConfig) -> Endpoint:
    """Create sub-model Endpoint for llm_query calls."""
    api_key = config.api_key
    if not api_key:
        env_var_map = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
        env_var = env_var_map.get(config.provider)
        if env_var:
            api_key = os.getenv(env_var)
            if not api_key:
                raise ValueError(f"{env_var} not set for sub-endpoint")

    return Endpoint(
        provider=config.provider,
        model=config.model,
        api_base=config.base_url or "",
        api_key=api_key or "",
        max_tokens=config.max_tokens,
    )


# ──────────────────────── Environment Factory ────────────────────────────────


async def create_rlm_environment(
    context: str,
    sub_endpoint: Endpoint,
    rlm_config: RLMConfig,
):
    """Create RLM or baseline environment based on config."""
    if not rlm_config.enabled:
        # Baseline: no environment, context goes in messages
        return None

    if rlm_config.use_tool_calling:
        from rollouts.environments.repl import REPLEnvironment

        return REPLEnvironment(
            context=context,
            sub_endpoint=sub_endpoint,
            recursive=rlm_config.recursive,
            max_depth=rlm_config.max_depth,
        )
    else:
        from rollouts.environments.repl import MessageParsingREPLEnvironment

        return MessageParsingREPLEnvironment(
            context=context,
            sub_endpoint=sub_endpoint,
            recursive=rlm_config.recursive,
            max_depth=rlm_config.max_depth,
        )


# ──────────────────────── Scoring Utilities ──────────────────────────────────


def exact_match_score(predicted: str | None, expected: str) -> Score:
    """Simple exact match scoring."""
    if predicted is None:
        return Score(metrics=(Metric("correct", 0.0, weight=1.0, metadata={"error": "no answer"}),))

    # Normalize for comparison
    pred_norm = str(predicted).strip().lower()
    exp_norm = str(expected).strip().lower()

    is_correct = pred_norm == exp_norm or exp_norm in pred_norm

    return Score(metrics=(Metric("correct", 1.0 if is_correct else 0.0, weight=1.0),))


def numeric_match_score(
    predicted: str | None, expected: int | float, tolerance: float = 0.01
) -> Score:
    """Numeric matching with tolerance."""
    if predicted is None:
        return Score(metrics=(Metric("correct", 0.0, weight=1.0, metadata={"error": "no answer"}),))

    import re

    # Extract number from prediction
    numbers = re.findall(r"-?\d+\.?\d*", str(predicted))
    if not numbers:
        return Score(
            metrics=(Metric("correct", 0.0, weight=1.0, metadata={"error": "no number found"}),)
        )

    # Check if any extracted number matches
    for num_str in numbers:
        try:
            num = float(num_str)
            if abs(num - expected) <= tolerance * abs(expected) + tolerance:
                return Score(metrics=(Metric("correct", 1.0, weight=1.0),))
        except ValueError:
            continue

    return Score(metrics=(Metric("correct", 0.0, weight=1.0),))
