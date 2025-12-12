"""GSM8K evaluation base config.

Two modes:
1. Single-turn: Model answers directly, extract from \\boxed{}
2. Multi-turn with calculator: Model uses tools to compute

Following experiment_config.md and RL loop patterns:
- Loads from HuggingFace datasets
- Compatible with DataBuffer for RL training
- Score function works with Sample type
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import trio

# ──────────────────────── Config Dataclasses ────────────────────────────────


@dataclass(frozen=True)
class EndpointConfig:
    """Endpoint configuration."""
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key: str | None = None


@dataclass(frozen=True)
class InferenceServerConfig:
    """Local inference server configuration."""
    enabled: bool = False
    backend: str = "sglang"
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    port: int = 30000
    gpu_id: int = 0


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset configuration for evaluation.

    Supports multiple sources:
    - "hf": HuggingFace datasets (default for GSM8K)
    - "jsonl": Local JSONL file
    - "parquet": Local Parquet file
    """

    source: str = "hf"  # "hf", "jsonl", "parquet"

    # For HuggingFace datasets
    hf_dataset: str = "openai/gsm8k"
    hf_subset: str = "main"
    hf_split: str = "test"

    # For local files
    path: str | None = None

    # Field mapping (GSM8K uses "question" and "answer")
    prompt_key: str = "question"
    label_key: str = "answer"

    # Limits
    max_samples: int | None = None
    seed: int = 42


@dataclass(frozen=True)
class EvalRunConfig:
    """Evaluation run configuration."""
    max_concurrent: int = 4
    max_turns: int = 10  # For multi-turn mode
    use_tools: bool = False  # Single-turn by default
    verbose: bool = True


@dataclass(frozen=True)
class OutputConfig:
    """Output configuration."""
    save_dir: Path = field(default_factory=lambda: Path("results"))
    experiment_name: str = "gsm8k_eval"
    run_id: str | None = None
    save_json: bool = True
    save_jsonl: bool = True

    def __post_init__(self):
        if self.run_id is None:
            object.__setattr__(self, "run_id", datetime.now().strftime("%Y%m%d-%H%M%S"))

    @property
    def output_dir(self) -> Path:
        return self.save_dir / f"{self.experiment_name}_{self.run_id}"

    @property
    def config_path(self) -> Path:
        return self.output_dir / "config.json"

    @property
    def metrics_path(self) -> Path:
        return self.output_dir / "metrics.json"


@dataclass(frozen=True)
class GSM8KConfig:
    """Top-level GSM8K experiment config."""
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    inference_server: InferenceServerConfig = field(default_factory=InferenceServerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    run: EvalRunConfig = field(default_factory=EvalRunConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        if data["endpoint"].get("api_key"):
            data["endpoint"]["api_key"] = "***REDACTED***"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


# ──────────────────────── Dataset Loading ───────────────────────────────────


def load_samples_from_config(config: DatasetConfig) -> list[dict[str, Any]]:
    """Load evaluation samples based on DatasetConfig.

    Returns list of dicts with:
        - prompt: str (the question)
        - answer: str (ground truth answer, just the number for GSM8K)
        - full_answer: str (full solution with reasoning, if available)
    """
    from rollouts.training.datasets.data_buffer import (
        load_samples_from_hf,
        load_samples_from_jsonl,
        load_samples_from_parquet,
    )

    if config.source == "hf":
        # Special handling for GSM8K format
        if "gsm8k" in config.hf_dataset.lower():
            return _load_gsm8k_hf(config)

        # Generic HuggingFace loading
        samples = load_samples_from_hf(
            dataset_name=config.hf_dataset,
            subset=config.hf_subset,
            split=config.hf_split,
            prompt_key=config.prompt_key,
            label_key=config.label_key,
            limit=config.max_samples,
        )
        return [
            {"prompt": s.prompt, "answer": s.metadata.get("label"), "full_answer": ""}
            for s in samples
        ]
    elif config.source == "jsonl":
        assert config.path, "path required for jsonl source"
        samples = load_samples_from_jsonl(
            path=Path(config.path),
            prompt_key=config.prompt_key,
            label_key=config.label_key,
            limit=config.max_samples,
        )
        return [
            {"prompt": s.prompt, "answer": s.metadata.get("label"), "full_answer": ""}
            for s in samples
        ]
    elif config.source == "parquet":
        assert config.path, "path required for parquet source"
        samples = load_samples_from_parquet(
            path=config.path,
            prompt_key=config.prompt_key,
            label_key=config.label_key,
            limit=config.max_samples,
        )
        return [
            {"prompt": s.prompt, "answer": s.metadata.get("label"), "full_answer": ""}
            for s in samples
        ]
    else:
        raise ValueError(f"Unknown source: {config.source}")


def _load_gsm8k_hf(config: DatasetConfig) -> list[dict[str, Any]]:
    """Load GSM8K from HuggingFace with special answer extraction."""
    from datasets import load_dataset

    ds = load_dataset(config.hf_dataset, config.hf_subset, split=config.hf_split)

    samples = []
    for i, row in enumerate(ds):
        if config.max_samples and i >= config.max_samples:
            break

        # Extract final answer from solution (format: "#### 42")
        solution = row[config.label_key]
        match = re.search(r"####\s*([\d,.-]+)", solution)
        answer = match.group(1).replace(",", "") if match else ""

        samples.append({
            "prompt": row[config.prompt_key],
            "answer": answer,
            "full_answer": solution,
        })

    return samples


def load_gsm8k_dataset(config: DatasetConfig) -> list[dict[str, Any]]:
    """Load GSM8K dataset. Alias for load_samples_from_config."""
    return load_samples_from_config(config)


def create_gsm8k_samples(config: DatasetConfig) -> list["Sample"]:
    """Create Sample objects from GSM8K for RL training.

    Returns Sample objects with:
    - prompt: The question
    - metadata["answer"]: Ground truth answer (for scoring)
    - metadata["full_answer"]: Full solution with reasoning

    Usage with new functional API:
        samples = create_gsm8k_samples(config)
        state = BufferState(seed=config.seed)
        batch, state = get_samples_flat(samples, state, n=32)
    """
    from rollouts.training.types import Sample

    dataset = load_samples_from_config(config)

    samples = []
    for i, row in enumerate(dataset):
        samples.append(Sample(
            prompt=row["prompt"],
            metadata={
                "answer": row["answer"],
                "full_answer": row["full_answer"],
            },
            index=i,
        ))

    return samples


def create_gsm8k_buffer(config: DatasetConfig) -> "DataBuffer":
    """DEPRECATED: Use create_gsm8k_samples + BufferState instead.

    Example migration:
        # Old:
        buffer = create_gsm8k_buffer(config)
        prompts = buffer.get_prompts(32)

        # New:
        samples = create_gsm8k_samples(config)
        state = BufferState(seed=config.seed)
        batch, state = get_samples_flat(samples, state, n=32)
    """
    from rollouts.training.datasets.data_buffer import DataBuffer

    samples = load_gsm8k_dataset(config)
    prompts = [s["prompt"] for s in samples]
    return DataBuffer(prompts=prompts, seed=config.seed)


# ──────────────────────── System Prompts ────────────────────────────────────


SINGLE_TURN_SYSTEM_PROMPT = """\
Solve the following math problem step by step. Show your work clearly.
Put your final answer in \\boxed{} format.

For example: The answer is \\boxed{42}
"""

MULTI_TURN_SYSTEM_PROMPT = """\
You are a math tutor with access to a calculator. Use the calculator tools to compute arithmetic.
Show your reasoning step by step, using the calculator for computations.
When you have the final answer, call complete_task with the result.
"""


# ──────────────────────── Score Functions ───────────────────────────────────


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...} format."""
    # Handle nested braces
    match = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if match:
        answer = match.group(1).strip()
        # Clean up: remove commas, convert to number
        answer = answer.replace(",", "").replace("$", "").strip()
        return answer
    return None


def normalize_answer(answer: str) -> float | None:
    """Normalize answer string to float for comparison."""
    if not answer:
        return None
    try:
        # Handle fractions like "1/2"
        if "/" in answer:
            parts = answer.split("/")
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
        # Handle percentages
        if "%" in answer:
            return float(answer.replace("%", "")) / 100
        return float(answer)
    except ValueError:
        return None


def gsm8k_score_fn(sample: Any) -> "Score":
    """Score function for GSM8K (single-turn mode).

    Works with Sample type from rollouts.training.types.
    Extracts answer from \\boxed{} and compares to ground truth.
    """
    from rollouts.dtypes import Metric, Score

    ground_truth = sample.metadata.get("answer")
    if ground_truth is None:
        return Score(metrics=(Metric("correct", 0.0, weight=1.0),))

    # Extract predicted answer from response
    response = sample.response if hasattr(sample, "response") else ""
    predicted = extract_boxed_answer(response)

    if predicted is None:
        return Score(metrics=(
            Metric("correct", 0.0, weight=1.0),
            Metric("parse_failed", 1.0, weight=0.0),
        ))

    # Normalize and compare
    pred_val = normalize_answer(predicted)
    true_val = normalize_answer(ground_truth)

    if pred_val is None or true_val is None:
        return Score(metrics=(
            Metric("correct", 0.0, weight=1.0),
            Metric("parse_failed", 1.0, weight=0.0),
        ))

    is_correct = abs(pred_val - true_val) < 0.01
    reward = 1.0 if is_correct else 0.0

    return Score(metrics=(
        Metric("correct", reward, weight=1.0),
        Metric("predicted", pred_val, weight=0.0),
        Metric("ground_truth", true_val, weight=0.0),
    ))


def gsm8k_tool_score_fn(trajectory: Any, sample: Any) -> "Score":
    """Score function for GSM8K with calculator tools (multi-turn mode).

    Extracts answer from complete_task tool call or tool results.
    """
    import json
    from rollouts.dtypes import Metric, Score

    # Sample is a dataclass with .ground_truth and .input dict
    ground_truth = sample.ground_truth or sample.input.get("answer")
    if ground_truth is None:
        return Score(metrics=(Metric("correct", 0.0, weight=1.0),))

    final_answer = None

    # Strategy 1: Look for complete_task tool call
    for msg in trajectory.messages:
        tool_calls = msg.get_tool_calls() if hasattr(msg, 'get_tool_calls') else []
        if msg.role == "assistant" and tool_calls:
            for tool_call in tool_calls:
                if tool_call.name == "complete_task":
                    args = tool_call.args
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            continue
                    if "final_result" in args:
                        try:
                            final_answer = float(args["final_result"])
                            break
                        except (ValueError, TypeError):
                            continue
            if final_answer is not None:
                break

    # Strategy 2: Look in tool results
    if final_answer is None:
        for msg in trajectory.messages:
            if msg.role == "tool" and msg.content:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                match = re.search(r"Final result:\s*([-\d.]+)", content)
                if match:
                    try:
                        final_answer = float(match.group(1))
                        break
                    except ValueError:
                        continue

    # Strategy 3: Fall back to boxed format in text
    if final_answer is None:
        for msg in reversed(trajectory.messages):
            if msg.role == "assistant" and msg.content:
                content = msg.content
                if isinstance(content, list):
                    content = " ".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                    )
                if isinstance(content, str):
                    predicted = extract_boxed_answer(content)
                    if predicted:
                        final_answer = normalize_answer(predicted)
                        break

    if final_answer is None:
        return Score(metrics=(
            Metric("correct", 0.0, weight=1.0),
            Metric("parse_failed", 1.0, weight=0.0),
        ))

    true_val = normalize_answer(str(ground_truth))
    if true_val is None:
        return Score(metrics=(
            Metric("correct", 0.0, weight=1.0),
            Metric("ground_truth_invalid", 1.0, weight=0.0),
        ))

    is_correct = abs(final_answer - true_val) < 0.01
    reward = 1.0 if is_correct else 0.0

    return Score(metrics=(
        Metric("correct", reward, weight=1.0),
        Metric("predicted", final_answer, weight=0.0),
        Metric("ground_truth", true_val, weight=0.0),
    ))


# ──────────────────────── Evaluation Logic ──────────────────────────────────


def _get_endpoint(config: GSM8KConfig) -> "Endpoint":
    """Create Endpoint from config, loading API key from env if needed."""
    from rollouts.dtypes import Endpoint

    api_key = config.endpoint.api_key
    if not api_key:
        env_var_map = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
        env_var = env_var_map.get(config.endpoint.provider)
        if env_var:
            api_key = os.getenv(env_var)
            if not api_key:
                raise ValueError(f"{env_var} not set")

    return Endpoint(
        provider=config.endpoint.provider,
        model=config.endpoint.model,
        api_base=config.endpoint.base_url or "",
        api_key=api_key or "",
    )


def _single_turn_score_fn(trajectory: "Trajectory", sample: "Sample") -> "Score":
    """Score function for single-turn GSM8K using rollouts evaluation types."""
    from rollouts.dtypes import Metric, Score

    # Get ground truth from sample
    ground_truth = sample.ground_truth or sample.input.get("answer")
    if not ground_truth:
        return Score(metrics=(Metric("correct", 0.0, weight=1.0),))

    # Extract response from trajectory
    response_text = ""
    for msg in reversed(trajectory.messages):
        if msg.role == "assistant" and msg.content:
            response_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    # Extract and compare
    predicted = extract_boxed_answer(response_text)
    if not predicted:
        return Score(metrics=(
            Metric("correct", 0.0, weight=1.0),
            Metric("parse_failed", 1.0, weight=0.0),
        ))

    pred_val = normalize_answer(predicted)
    true_val = normalize_answer(str(ground_truth))

    if pred_val is None or true_val is None:
        return Score(metrics=(
            Metric("correct", 0.0, weight=1.0),
            Metric("parse_failed", 1.0, weight=0.0),
        ))

    is_correct = abs(pred_val - true_val) < 0.01
    return Score(metrics=(
        Metric("correct", 1.0 if is_correct else 0.0, weight=1.0),
        Metric("predicted", pred_val, weight=0.0),
        Metric("ground_truth", true_val, weight=0.0),
    ))


async def _eval_single_turn(config: GSM8KConfig) -> dict[str, Any]:
    """Single-turn evaluation using rollouts.evaluation framework."""
    from rollouts._logging import setup_logging
    from rollouts.agents import handle_stop_max_turns
    from rollouts.dtypes import EvalConfig, Message, RunConfig
    from rollouts.evaluation import evaluate

    setup_logging(level="INFO", use_color=True)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info(f"GSM8K Single-Turn Eval: {config.output.experiment_name}")
    logger.info("=" * 60)

    # Load dataset
    dataset = load_samples_from_config(config.dataset)
    logger.info(f"Dataset: {len(dataset)} samples")

    # Create endpoint
    endpoint = _get_endpoint(config)

    # Prepare messages function
    def prepare_messages(sample_data: dict) -> list[Message]:
        return [
            Message(role="system", content=SINGLE_TURN_SYSTEM_PROMPT),
            Message(role="user", content=sample_data["prompt"]),
        ]

    # Single-turn: stop after 1 turn (use max_turns from config, default 1 for single-turn)
    max_turns = 1 if not config.run.use_tools else config.run.max_turns
    run_config = RunConfig(
        on_chunk=lambda _: trio.lowlevel.checkpoint(),
        handle_stop=handle_stop_max_turns(max_turns),
    )

    # Use rollouts evaluation framework
    eval_config = EvalConfig(
        eval_name=config.output.experiment_name,
        score_fn=_single_turn_score_fn,
        max_samples=config.dataset.max_samples,
        max_concurrent=config.run.max_concurrent,
        verbose=config.run.verbose,
        output_dir=config.output.output_dir,
        run_config=run_config,
    )

    report = await evaluate(
        dataset=iter(dataset),
        prepare_messages=prepare_messages,
        endpoint=endpoint,
        config=eval_config,
        dataset_path="gsm8k",
        environment_factory=None,  # No tools for single-turn
    )

    # Return metrics in expected format
    accuracy = report.summary_metrics.get("mean_correct", 0.0)
    total = report.total_samples

    logger.info("=" * 60)
    logger.info("Evaluation Complete")
    logger.info("=" * 60)
    logger.info(f"Accuracy: {accuracy:.1%}")
    logger.info(f"Results: {config.output.output_dir}")

    return {"accuracy": accuracy, "total": total, **report.summary_metrics}


async def _eval_multi_turn(config: GSM8KConfig) -> dict[str, Any]:
    """Multi-turn evaluation: model uses calculator tools."""
    from rollouts._logging import setup_logging
    from rollouts.agents import handle_stop_max_turns
    from rollouts.dtypes import EvalConfig, Message, RunConfig
    from rollouts.environments.calculator import CalculatorEnvironment
    from rollouts.evaluation import evaluate

    setup_logging(level="INFO", use_color=True)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info(f"GSM8K Multi-Turn Eval: {config.output.experiment_name}")
    logger.info("=" * 60)

    # Load dataset
    dataset = load_samples_from_config(config.dataset)
    logger.info(f"Dataset: {len(dataset)} samples")

    # Create endpoint
    endpoint = _get_endpoint(config)

    # Prepare for evaluation framework
    def prepare_messages(sample_data: dict) -> list[Message]:
        return [
            Message(role="system", content=MULTI_TURN_SYSTEM_PROMPT),
            Message(role="user", content=sample_data["prompt"]),
        ]

    async def environment_factory(sample_data: dict) -> CalculatorEnvironment:
        return CalculatorEnvironment()

    # Multi-turn with max_turns limit
    run_config = RunConfig(
        on_chunk=lambda _: trio.lowlevel.checkpoint(),
        handle_stop=handle_stop_max_turns(config.run.max_turns),
    )

    eval_config = EvalConfig(
        eval_name=config.output.experiment_name,
        score_fn=gsm8k_tool_score_fn,
        max_samples=config.dataset.max_samples,
        max_concurrent=config.run.max_concurrent,
        verbose=config.run.verbose,
        output_dir=config.output.output_dir,
        run_config=run_config,
    )

    report = await evaluate(
        dataset=iter(dataset),
        prepare_messages=prepare_messages,
        endpoint=endpoint,
        config=eval_config,
        dataset_path="gsm8k",
        environment_factory=environment_factory,
    )

    logger.info("=" * 60)
    logger.info("Evaluation Complete")
    logger.info("=" * 60)
    logger.info(f"Samples: {report.total_samples}")
    logger.info(f"Mean reward: {report.summary_metrics.get('mean_reward', 0):.3f}")
    logger.info(f"Results: {config.output.output_dir}")

    return report.summary_metrics


def evaluate_gsm8k(config: GSM8KConfig) -> dict[str, Any]:
    """Run GSM8K evaluation."""
    if config.run.use_tools:
        return trio.run(_eval_multi_turn, config)
    else:
        return trio.run(_eval_single_turn, config)
