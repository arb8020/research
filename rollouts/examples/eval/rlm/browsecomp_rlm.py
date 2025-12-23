#!/usr/bin/env python3
"""
BrowseComp-Plus evaluation for RLM.

BrowseComp-Plus is BrowseComp with pre-downloaded documents, enabling
offline evaluation of multi-hop QA over large document corpora.

From the RLM paper:
"RLM(GPT-5) is the only model/agent able to achieve and maintain perfect
performance at the 1000 document scale... RLMs outperform ReAct + GPT-5 + BM25."

Datasets on Hugging Face:
- Tevatron/browsecomp-plus: Queries with evidence/gold/negative docs (830 rows)
- Tevatron/browsecomp-plus-corpus: Full 100K document corpus

This evaluation:
1. Loads BrowseComp-Plus queries
2. Provides document subset as RLM context
3. Tests ability to find answers across many documents

Usage:
    # Run with 50 documents per query
    python -m examples.eval.rlm.browsecomp_rlm --docs 50

    # Scale test: 10, 50, 100, 500 docs
    python -m examples.eval.rlm.browsecomp_rlm --scale-test

References:
- BrowseComp-Plus: https://huggingface.co/datasets/Tevatron/browsecomp-plus
- RLM Blog: https://alexzhang13.github.io/blog/2025/rlm/
"""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass, field
from typing import Any

import trio

from rollouts.agents import handle_stop_max_turns, run_agent
from rollouts.dtypes import (
    Actor,
    AgentState,
    Message,
    RunConfig,
    Trajectory,
)

from .base_config import (
    DatasetConfig,
    EndpointConfig,
    EvalRunConfig,
    OutputConfig,
    RLMConfig,
    RLMEvalConfig,
    RLM_TOOL_SYSTEM_PROMPT,
    exact_match_score,
    get_endpoint,
    get_sub_endpoint,
)

logger = logging.getLogger(__name__)


# ──────────────────────── Dataset Config ────────────────────────────────────


@dataclass(frozen=True)
class BrowseCompPlusDatasetConfig(DatasetConfig):
    """BrowseComp-Plus dataset config."""

    num_documents: int = 50  # Documents per query (gold + negatives)
    include_corpus_negatives: bool = True  # Add random corpus docs as negatives
    seed: int = 42
    max_samples: int | None = 10


@dataclass(frozen=True)
class BrowseCompPlusConfig(RLMEvalConfig):
    """BrowseComp-Plus RLM evaluation config."""

    dataset: BrowseCompPlusDatasetConfig = field(
        default_factory=BrowseCompPlusDatasetConfig
    )
    output: OutputConfig = field(
        default_factory=lambda: OutputConfig(experiment_name="browsecomp_plus")
    )


# ──────────────────────── Dataset Loading ────────────────────────────────────


def _decrypt_browsecomp(text: str, key: str) -> str:
    """Decrypt BrowseComp obfuscated text (simple XOR with base64)."""
    import base64
    import hashlib

    # Derive key using SHA256
    hasher = hashlib.sha256()
    hasher.update(key.encode())
    key_bytes = hasher.digest()

    # Decode and XOR
    encrypted = base64.b64decode(text)
    key_extended = key_bytes * (len(encrypted) // len(key_bytes) + 1)
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key_extended))

    return decrypted.decode('utf-8', errors='ignore')


def load_browsecomp_plus(config: BrowseCompPlusDatasetConfig) -> list[dict[str, Any]]:
    """Load BrowseComp-Plus dataset from Hugging Face.

    Returns list of samples with:
        - query: The question
        - answer: Ground truth answer
        - documents: List of document texts
        - num_gold: Number of gold/evidence docs
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    logger.info("Loading BrowseComp-Plus dataset...")
    ds = load_dataset("Tevatron/browsecomp-plus", split="test")

    # Optionally load corpus for additional negatives
    corpus = None
    if config.include_corpus_negatives:
        try:
            logger.info("Loading BrowseComp-Plus corpus for negatives...")
            corpus_ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
            corpus = [row["text"] for row in corpus_ds]
            logger.info(f"Loaded {len(corpus):,} corpus documents")
        except Exception as e:
            logger.warning(f"Could not load corpus: {e}. Using only query docs.")

    samples = []
    rng = random.Random(config.seed)

    for row in ds:
        # Note: Data is obfuscated. For real eval, need decryption key.
        # Here we use obfuscated text as-is for structure testing.
        query = row.get("query", "")
        answer = row.get("answer", "")

        # Collect documents
        documents = []

        # Gold docs (contain the answer)
        gold_docs = row.get("gold_docs", [])
        for doc in gold_docs:
            if isinstance(doc, dict):
                documents.append({
                    "text": doc.get("text", ""),
                    "type": "gold",
                    "docid": doc.get("docid", ""),
                })

        # Evidence docs (help find answer)
        evidence_docs = row.get("evidence_docs", [])
        for doc in evidence_docs:
            if isinstance(doc, dict) and doc.get("docid") not in [d.get("docid") for d in gold_docs]:
                documents.append({
                    "text": doc.get("text", ""),
                    "type": "evidence",
                    "docid": doc.get("docid", ""),
                })

        # Hard negatives
        negative_docs = row.get("negative_docs", [])
        for doc in negative_docs:
            if isinstance(doc, dict):
                documents.append({
                    "text": doc.get("text", ""),
                    "type": "negative",
                    "docid": doc.get("docid", ""),
                })

        # Add corpus negatives to reach target doc count
        num_gold_evidence = sum(1 for d in documents if d["type"] in ("gold", "evidence"))
        num_needed = config.num_documents - len(documents)

        if num_needed > 0 and corpus:
            corpus_sample = rng.sample(corpus, min(num_needed, len(corpus)))
            for text in corpus_sample:
                documents.append({"text": text, "type": "corpus_negative", "docid": ""})

        # Shuffle documents
        rng.shuffle(documents)

        # Truncate to target size
        documents = documents[:config.num_documents]

        samples.append({
            "id": row.get("query_id", str(len(samples))),
            "query": query,
            "answer": answer,
            "documents": documents,
            "num_gold": sum(1 for d in documents if d["type"] == "gold"),
            "num_evidence": sum(1 for d in documents if d["type"] == "evidence"),
        })

    logger.info(f"Loaded {len(samples)} samples")

    # Subsample if requested
    if config.max_samples and config.max_samples < len(samples):
        samples = rng.sample(samples, config.max_samples)
        logger.info(f"Subsampled to {len(samples)} samples")

    return samples


# ──────────────────────── Evaluation Logic ──────────────────────────────────


BROWSECOMP_SYSTEM_PROMPT = RLM_TOOL_SYSTEM_PROMPT + """

## Document Corpus Task

The context contains multiple documents separated by markers.
Your task:
1. Search through documents to find relevant information
2. Some questions require combining facts from multiple documents
3. Use the REPL to efficiently search (grep, regex) before reading fully
4. Use llm_query to extract specific facts from promising documents

Strategy:
- First get document count: context.count('=== Document')
- Search for keywords from the question using regex
- Read relevant document sections
- Synthesize the final answer from found information
"""


async def evaluate_sample(
    sample: dict[str, Any],
    config: BrowseCompPlusConfig,
) -> dict[str, Any]:
    """Evaluate a single BrowseComp-Plus sample."""
    from rollouts.environments.repl import MessageParsingREPLEnvironment, REPLEnvironment

    endpoint = get_endpoint(config.endpoint)
    sub_endpoint = get_sub_endpoint(config.sub_endpoint)

    query = sample["query"]
    expected = sample["answer"]
    documents = sample["documents"]

    # Format documents as context
    context_parts = []
    for i, doc in enumerate(documents):
        context_parts.append(f"=== Document {i} ===\n{doc['text']}\n")
    context = "\n".join(context_parts)

    # Create environment
    if config.rlm.use_tool_calling:
        environment = REPLEnvironment(
            context=context,
            sub_endpoint=sub_endpoint,
            recursive=config.rlm.recursive,
            max_depth=config.rlm.max_depth,
        )
    else:
        environment = MessageParsingREPLEnvironment(
            context=context,
            sub_endpoint=sub_endpoint,
            recursive=config.rlm.recursive,
            max_depth=config.rlm.max_depth,
        )

    trajectory = Trajectory(
        messages=[
            Message(role="system", content=BROWSECOMP_SYSTEM_PROMPT),
            Message(role="user", content=query),
        ]
    )

    actor = Actor(
        trajectory=trajectory,
        endpoint=endpoint,
        tools=environment.get_tools(),
    )

    state = AgentState(actor=actor, environment=environment)

    async def silent_handler(_: object) -> None:
        await trio.lowlevel.checkpoint()

    run_config = RunConfig(
        on_chunk=silent_handler,
        handle_stop=handle_stop_max_turns(config.run.max_turns),
    )

    states = await run_agent(state, run_config)

    # Score result
    final_answer = environment._final_answer
    score = exact_match_score(final_answer, expected)

    return {
        "sample_id": sample["id"],
        "query": query[:80],
        "expected": expected,
        "predicted": final_answer,
        "correct": score.metrics[0].value == 1.0,
        "num_turns": len(states),
        "num_documents": len(documents),
        "num_gold": sample["num_gold"],
        "context_chars": len(context),
    }


async def run_evaluation(config: BrowseCompPlusConfig) -> dict[str, Any]:
    """Run full BrowseComp-Plus evaluation."""
    from rollouts._logging import setup_logging

    setup_logging(level="INFO", use_color=True)

    logger.info("=" * 60)
    logger.info("BrowseComp-Plus RLM Evaluation")
    logger.info("=" * 60)
    logger.info(f"Documents per query: {config.dataset.num_documents}")
    logger.info(f"Samples: {config.dataset.max_samples}")
    logger.info(f"Model: {config.endpoint.provider}/{config.endpoint.model}")

    # Load samples from HuggingFace
    samples = load_browsecomp_plus(config.dataset)

    if not samples:
        logger.error("No samples loaded!")
        return {"error": "No samples"}

    logger.info(f"Context sizes: ~{sum(len(s.get('documents', [])) for s in samples) // len(samples)} docs avg")

    # Run evaluation
    results = []
    for sample in samples:
        result = await evaluate_sample(sample, config)
        results.append(result)
        status = "✓" if result["correct"] else "✗"
        logger.info(
            f"  {status} {result['sample_id']}: "
            f"pred={result['predicted'][:50] if result['predicted'] else 'None'}... "
            f"(expected {result['expected'][:50]}...)"
        )

    # Compute metrics
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0
    avg_turns = sum(r["num_turns"] for r in results) / total if total > 0 else 0.0

    logger.info("=" * 60)
    logger.info(f"Results: {correct}/{total} ({accuracy:.1%})")
    logger.info(f"Average turns: {avg_turns:.1f}")
    logger.info("=" * 60)

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "num_documents": config.dataset.num_documents,
        "avg_turns": avg_turns,
        "results": results,
    }


async def run_scale_test(
    base_config: BrowseCompPlusConfig,
    doc_counts: list[int],
) -> dict[str, Any]:
    """Run evaluation at multiple document scales."""
    from rollouts._logging import setup_logging

    setup_logging(level="INFO", use_color=True)

    logger.info("=" * 60)
    logger.info("BrowseComp-Plus Scale Test")
    logger.info("=" * 60)

    all_results = {}
    for num_docs in doc_counts:
        logger.info(f"\n>>> Testing with {num_docs} documents...")

        config = BrowseCompPlusConfig(
            endpoint=base_config.endpoint,
            sub_endpoint=base_config.sub_endpoint,
            rlm=base_config.rlm,
            dataset=BrowseCompPlusDatasetConfig(
                num_documents=num_docs,
                max_samples=base_config.dataset.max_samples,
                seed=base_config.dataset.seed,
            ),
            run=base_config.run,
        )

        result = await run_evaluation(config)
        all_results[num_docs] = result

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Scale Test Summary")
    logger.info("=" * 60)
    for num_docs, result in sorted(all_results.items()):
        logger.info(f"  {num_docs:4d} docs: {result['accuracy']:.1%} accuracy")

    return all_results


# ──────────────────────── CLI ────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="BrowseComp-Plus RLM Evaluation")

    parser.add_argument(
        "--docs",
        type=int,
        default=50,
        help="Documents per query (default: 50)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples (default: 5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-5-20250929",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="anthropic",
        choices=["anthropic", "openai"],
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--scale-test",
        action="store_true",
        help="Run at multiple scales (10, 50, 100, 500 docs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    base_config = BrowseCompPlusConfig(
        endpoint=EndpointConfig(provider=args.provider, model=args.model),
        rlm=RLMConfig(enabled=True),
        dataset=BrowseCompPlusDatasetConfig(
            num_documents=args.docs,
            max_samples=args.samples,
            seed=args.seed,
        ),
        run=EvalRunConfig(max_turns=args.max_turns),
    )

    if args.scale_test:
        trio.run(run_scale_test, base_config, [10, 50, 100, 500])
    else:
        trio.run(run_evaluation, base_config)


if __name__ == "__main__":
    main()
