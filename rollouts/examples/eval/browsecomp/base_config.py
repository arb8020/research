"""BrowseComp evaluation base config.

BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents
https://openai.com/index/browsecomp/

This benchmark evaluates whether models can answer questions that require
web browsing to solve. Questions and answers are encrypted in the dataset
to prevent contamination.

Following rollouts patterns:
- Loads from OpenAI's public CSV
- LLM-as-judge grading with configurable grader model
- Compatible with rollouts evaluation framework
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import random
import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import trio

from rollouts.dtypes import Endpoint, Message, Metric, Score, Trajectory

logger = logging.getLogger(__name__)

# ──────────────────────── Dataset URL ────────────────────────────────────────

BROWSECOMP_CSV_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
)

# ──────────────────────── Prompts ────────────────────────────────────────────

QUERY_TEMPLATE = """
{question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
""".strip()

SYSTEM_PROMPT = """You are a research assistant with access to web search and browsing tools.

To answer questions, you should:
1. Use web_search to find relevant pages
2. Use web_fetch to read promising results
3. Synthesize information from multiple sources if needed
4. Provide your final answer with confidence

Be thorough - many questions require finding specific, hard-to-find information."""

# ──────────────────────── Config Dataclasses ────────────────────────────────


@dataclass(frozen=True)
class EndpointConfig:
    """Endpoint configuration."""

    provider: str = "openai"
    model: str = "gpt-4o"
    base_url: str | None = None
    api_key: str | None = None


@dataclass(frozen=True)
class GraderConfig:
    """Grader model configuration (separate from main model)."""

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key: str | None = None


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset configuration."""

    csv_url: str = BROWSECOMP_CSV_URL
    max_samples: int | None = None
    seed: int = 42


@dataclass(frozen=True)
class EvalRunConfig:
    """Evaluation run configuration."""

    max_concurrent: int = 4
    max_turns: int = 10  # Allow multiple search/fetch turns
    verbose: bool = True


@dataclass(frozen=True)
class OutputConfig:
    """Output configuration."""

    save_dir: Path = field(default_factory=lambda: Path("results"))
    experiment_name: str = "browsecomp_eval"
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
class BrowseCompConfig:
    """Top-level BrowseComp experiment config."""

    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    grader: GraderConfig = field(default_factory=GraderConfig)
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
        if data["grader"].get("api_key"):
            data["grader"]["api_key"] = "***REDACTED***"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


# ──────────────────────── Encryption Utils ───────────────────────────────────


def _derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from password using SHA256."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def _decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64-encoded ciphertext with XOR."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = _derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key, strict=False))
    return decrypted.decode()


# ──────────────────────── Dataset Loading ────────────────────────────────────


def load_browsecomp_dataset(config: DatasetConfig) -> list[dict[str, Any]]:
    """Load BrowseComp dataset from CSV.

    Returns list of dicts with:
        - question: str (decrypted question)
        - answer: str (decrypted answer)
        - canary: str (encryption key, kept for reference)
    """
    logger.info(f"Loading BrowseComp from {config.csv_url}")
    df = pd.read_csv(config.csv_url)

    samples = []
    for _, row in df.iterrows():
        canary = row.get("canary", "")
        question = _decrypt(row.get("problem", ""), canary)
        answer = _decrypt(row.get("answer", ""), canary)

        samples.append({
            "question": question,
            "answer": answer,
            "canary": canary,
        })

    # Subsample if requested
    if config.max_samples and config.max_samples < len(samples):
        rng = random.Random(config.seed)
        samples = rng.sample(samples, config.max_samples)

    logger.info(f"Loaded {len(samples)} samples")
    return samples


# ──────────────────────── Grading ────────────────────────────────────────────


async def grade_response(
    question: str,
    correct_answer: str,
    response: str,
    grader_endpoint: Endpoint,
) -> bool:
    """Grade a response using LLM-as-judge.

    Returns True if the response is correct, False otherwise.
    """
    from rollouts.dtypes import Actor
    from rollouts.providers import get_provider_function

    grader_prompt = GRADER_TEMPLATE.format(
        question=question,
        correct_answer=correct_answer,
        response=response,
    )

    # Create trajectory for grader
    trajectory = Trajectory(
        messages=[Message(role="user", content=grader_prompt)],
    )

    # Get provider and make request
    provider_fn = get_provider_function(grader_endpoint.provider, grader_endpoint.model)

    # Simple streaming callback that does nothing
    async def noop_callback(chunk: object) -> None:
        pass

    actor = Actor(trajectory=trajectory, endpoint=grader_endpoint, tools=[])
    result_actor = await provider_fn(actor, noop_callback)

    # Extract response
    grading_response = ""
    for msg in reversed(result_actor.trajectory.messages):
        if msg.role == "assistant" and msg.content:
            grading_response = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    # Parse result
    match = re.search(r"correct:\s*(yes|no)", grading_response, re.IGNORECASE)
    return match is not None and match.group(1).lower() == "yes"


# ──────────────────────── Score Function ─────────────────────────────────────


def create_browsecomp_score_fn(grader_endpoint: Endpoint) -> Callable[[Any], Awaitable[Score]]:
    """Create score function with grader endpoint bound.

    Returns an async score function compatible with rollouts EvalConfig.
    Sample has trajectory via sample.trajectory.
    """

    async def browsecomp_score_fn(sample: Any) -> Score:
        """Score function for BrowseComp.

        Uses LLM-as-judge to grade the response.
        Sample has trajectory via sample.trajectory.
        """
        trajectory = sample.trajectory
        if trajectory is None:
            return Score(
                metrics=(Metric("correct", 0.0, weight=1.0, metadata={"error": "no trajectory"}),)
            )

        question = sample.input.get("question", "")
        correct_answer = sample.ground_truth or sample.input.get("answer", "")

        # Extract response from trajectory
        response_text = ""
        for msg in reversed(trajectory.messages):
            if msg.role == "assistant" and msg.content:
                response_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        if not response_text:
            return Score(
                metrics=(
                    Metric("correct", 0.0, weight=1.0),
                    Metric("no_response", 1.0, weight=0.0),
                )
            )

        # Grade with LLM
        is_correct = await grade_response(
            question=question,
            correct_answer=correct_answer,
            response=response_text,
            grader_endpoint=grader_endpoint,
        )

        return Score(metrics=(Metric("correct", 1.0 if is_correct else 0.0, weight=1.0),))

    return browsecomp_score_fn


# ──────────────────────── Endpoint Helpers ───────────────────────────────────


def _get_endpoint(config: EndpointConfig) -> Endpoint:
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
    )


def _get_grader_endpoint(config: GraderConfig) -> Endpoint:
    """Create grader Endpoint from config."""
    api_key = config.api_key
    if not api_key:
        env_var_map = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
        env_var = env_var_map.get(config.provider)
        if env_var:
            api_key = os.getenv(env_var)
            if not api_key:
                raise ValueError(f"{env_var} not set for grader")

    return Endpoint(
        provider=config.provider,
        model=config.model,
        api_base=config.base_url or "",
        api_key=api_key or "",
    )


# ──────────────────────── Evaluation Logic ───────────────────────────────────


async def _run_eval(config: BrowseCompConfig) -> dict[str, Any]:
    """Run BrowseComp evaluation using rollouts framework."""
    from rollouts._logging import setup_logging
    from rollouts.agents import handle_stop_max_turns
    from rollouts.dtypes import EvalConfig, RunConfig
    from rollouts.environments import BrowsingEnvironment
    from rollouts.evaluation import evaluate

    setup_logging(level="INFO", use_color=True)

    logger.info("=" * 60)
    logger.info(f"BrowseComp Eval: {config.output.experiment_name}")
    logger.info("=" * 60)

    # Load dataset
    dataset = load_browsecomp_dataset(config.dataset)
    logger.info(f"Dataset: {len(dataset)} samples")

    # Create endpoints
    endpoint = _get_endpoint(config.endpoint)
    grader_endpoint = _get_grader_endpoint(config.grader)

    logger.info(f"Model: {config.endpoint.provider}/{config.endpoint.model}")
    logger.info(f"Grader: {config.grader.provider}/{config.grader.model}")
    logger.info(f"Max turns: {config.run.max_turns}")

    # Prepare messages function (custom formatting with QUERY_TEMPLATE)
    def prepare_messages(sample_data: dict) -> list[Message]:
        question = sample_data["question"]
        return [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=QUERY_TEMPLATE.format(question=question)),
        ]

    async def silent_handler(_: object) -> None:
        await trio.lowlevel.checkpoint()

    # Allow multiple turns for browsing
    run_config = RunConfig(
        on_chunk=silent_handler,
        handle_stop=handle_stop_max_turns(config.run.max_turns),
    )

    # Create score function with grader
    score_fn = create_browsecomp_score_fn(grader_endpoint)

    # Environment factory for browsing tools
    async def environment_factory(sample_data: dict) -> BrowsingEnvironment:
        return BrowsingEnvironment()

    # Use new simplified API with prepare_messages (custom formatting)
    eval_config = EvalConfig(
        endpoint=endpoint,
        score_fn=score_fn,
        prepare_messages=prepare_messages,  # Custom formatting with QUERY_TEMPLATE
        environment_factory=environment_factory,
        max_samples=config.dataset.max_samples,
        max_concurrent=config.run.max_concurrent,
        verbose=config.run.verbose,
        output_dir=config.output.output_dir,
        eval_name=config.output.experiment_name,
        run_config=run_config,
    )

    report = await evaluate(iter(dataset), eval_config)

    # Return metrics
    accuracy = report.summary_metrics.get("mean_correct", 0.0)

    logger.info("=" * 60)
    logger.info("Evaluation Complete")
    logger.info("=" * 60)
    logger.info(f"Accuracy: {accuracy:.1%}")
    logger.info(f"Results: {config.output.output_dir}")

    return {"accuracy": accuracy, "total": report.total_samples, **report.summary_metrics}


def evaluate_browsecomp(config: BrowseCompConfig) -> dict[str, Any]:
    """Run BrowseComp evaluation."""
    return trio.run(_run_eval, config)
