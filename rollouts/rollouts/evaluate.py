#!/usr/bin/env python3
"""Generic dataset evaluation using rollouts pattern.

This module provides a generic evaluation framework that works with any dataset
by accepting adapter functions as parameters.

Style: Tiger Style + tuple returns + single assignment.
"""

import logging
from dataclasses import asdict
from typing import Dict, Any, List, Callable, Iterator, Tuple
from tqdm.asyncio import tqdm
import trio

from .dtypes import Endpoint, Trajectory, Message
from .rollout_legacy import generate  # TODO: Migrate to providers.rollout_openai

logger = logging.getLogger(__name__)


async def evaluate_sample(
    sample: Dict[str, Any],
    sample_id: str,
    endpoint: Endpoint,
    message_preparer: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
    reward_functions: List[Tuple[str, Callable[[Trajectory, Dict[str, Any]], float]]],
) -> Dict[str, Any]:
    """Evaluate a single sample (generic).

    Args:
        sample: Sample dict from dataset
        sample_id: Unique ID
        endpoint: API endpoint
        message_preparer: Function to convert sample to messages
        reward_functions: List of (name, reward_fn) tuples

    Returns:
        Result dict with metrics
    """
    assert sample is not None
    assert isinstance(sample, dict)
    assert sample_id is not None
    assert isinstance(sample_id, str)
    assert len(sample_id) > 0
    assert endpoint is not None
    assert message_preparer is not None
    assert callable(message_preparer)
    assert reward_functions is not None
    assert isinstance(reward_functions, list)
    assert len(reward_functions) > 0

    # Prepare messages (convert raw dicts to Message objects)
    messages_dicts = message_preparer(sample)
    assert messages_dicts is not None
    assert isinstance(messages_dicts, list)
    messages = [Message(role=m["role"], content=m["content"]) for m in messages_dicts]
    rollout = Trajectory(messages=messages)
    assert rollout is not None

    # Generate response (Tiger Style: tuple return for explicit error handling)
    result_rollout, error = await generate(endpoint, rollout)

    if error:
        # Failed after retries - return zeros for all reward functions with error info
        error_result = {
            "sample_id": sample_id,
            "status": "error",
            "error": error,
            "response": ""
        }
        for name, _ in reward_functions:
            assert name is not None
            assert isinstance(name, str)
            error_result[name] = 0.0
        assert "sample_id" in error_result
        assert "status" in error_result
        return error_result

    # Success - compute all reward functions
    assert result_rollout is not None, "result_rollout must be set if error is None"
    result = {
        "sample_id": sample_id,
        "status": "success"
    }
    for name, reward_fn in reward_functions:
        assert name is not None
        assert callable(reward_fn)
        reward_value = reward_fn(result_rollout, sample)
        assert isinstance(reward_value, (int, float))
        result[name] = reward_value

    # Add response text
    result["response"] = result_rollout.messages[-1].content if result_rollout.messages else ""
    assert "sample_id" in result
    assert "status" in result

    return result


async def evaluate_dataset(
    dataset: Iterator[Dict[str, Any]],
    endpoint: Endpoint,
    message_preparer: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
    reward_functions: List[Tuple[str, Callable[[Trajectory, Dict[str, Any]], float]]],
    sample_id_fn: Callable[[int, Dict[str, Any]], str],
    max_concurrent: int = 10,
    group_by_key: str | None = None,
    verbose: bool = True,
    eval_name: str = "Evaluation"
) -> Dict[str, Any]:
    """Run generic dataset evaluation.

    Args:
        dataset: Iterator of sample dicts
        endpoint: API endpoint
        message_preparer: Function to convert sample → messages
        reward_functions: List of (name, reward_fn) tuples
        sample_id_fn: Function to generate sample IDs
        max_concurrent: Max concurrent evaluations
        group_by_key: Optional key to group results by (e.g., "ui_type")
        verbose: Print progress with tqdm
        eval_name: Name for progress display

    Returns:
        Results dict with summary and per-sample results
    """
    assert dataset is not None
    assert endpoint is not None
    assert message_preparer is not None
    assert callable(message_preparer)
    assert reward_functions is not None
    assert isinstance(reward_functions, list)
    assert len(reward_functions) > 0
    assert sample_id_fn is not None
    assert callable(sample_id_fn)
    assert max_concurrent > 0
    assert max_concurrent <= 100  # Reasonable upper bound
    assert eval_name is not None
    assert isinstance(eval_name, str)

    # Convert iterator to list (needed for indexing)
    dataset_list = list(dataset)
    assert isinstance(dataset_list, list)

    # Evaluate with concurrency control and progress tracking
    semaphore = trio.Semaphore(max_concurrent)
    pbar = tqdm(total=len(dataset_list), desc=eval_name, unit="sample", disable=not verbose)
    completed_count = 0
    total_accuracy = 0.0

    async def eval_with_semaphore(idx: int, sample: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal completed_count, total_accuracy
        async with semaphore:
            sample_id = sample_id_fn(idx, sample)
            result = await evaluate_sample(
                sample, sample_id, endpoint, message_preparer, reward_functions
            )
            # Update running accuracy for first reward function
            completed_count += 1
            total_accuracy += result[reward_functions[0][0]]
            mean_acc = total_accuracy / completed_count
            pbar.set_postfix({"acc": f"{mean_acc:.1%}"})
            pbar.update(1)
            return result

    # Run all evaluations with progress bar
    results = []
    async with trio.open_nursery() as nursery:
        async def run_and_collect(idx: int, sample: Dict[str, Any]):
            result = await eval_with_semaphore(idx, sample)
            results.append(result)

        for i, s in enumerate(dataset_list):
            nursery.start_soon(run_and_collect, i, s)
    pbar.close()

    # Compute summary metrics for all reward functions
    total = len(results)

    # Count errors
    error_results = [r for r in results if r.get("status") == "error"]
    num_errors = len(error_results)
    num_success = total - num_errors

    summary = {
        "total_samples": total,
        "num_success": num_success,
        "num_errors": num_errors,
    }

    # Add error breakdown if there are errors
    if num_errors > 0:
        error_types = {}
        for r in error_results:
            error_type = r.get("error_type", "Unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
        summary["error_breakdown"] = error_types

    for name, _ in reward_functions:
        mean_value = sum(r[name] for r in results) / total if total > 0 else 0.0
        summary[f"mean_{name}"] = mean_value
        summary[f"total_{name}"] = sum(r[name] for r in results)

    # Group by key if provided (e.g., "ui_type")
    if group_by_key:
        grouped = {}
        for result in results:
            # Find corresponding sample
            sample = next(
                s for i, s in enumerate(dataset_list)
                if sample_id_fn(i, s) == result["sample_id"]
            )
            group_value = sample.get(group_by_key, "unknown")

            if group_value not in grouped:
                grouped[group_value] = []
            grouped[group_value].append(result)

        # Compute metrics per group
        group_metrics = {}
        for group_value, group_results in grouped.items():
            group_total = len(group_results)
            group_metrics[group_value] = {"total": group_total}

            for name, _ in reward_functions:
                group_mean = sum(r[name] for r in group_results) / group_total if group_total > 0 else 0.0
                group_metrics[group_value][name] = group_mean
                group_metrics[group_value][f"total_{name}"] = sum(r[name] for r in group_results)

        summary[f"by_{group_by_key}"] = group_metrics

    # Log summary
    if verbose:
        logger.info("")
        logger.info("="*50)
        logger.info("📊 Summary")
        logger.info("="*50)
        logger.info(f"Total samples: {total}")
        logger.info(f"Success: {summary['num_success']}, Errors: {summary['num_errors']}")

        # Log error breakdown if present
        if summary.get("error_breakdown"):
            logger.info("Error types:")
            for error_type, count in summary["error_breakdown"].items():
                logger.info(f"  {error_type}: {count}")

        for name, _ in reward_functions:
            mean_val = summary[f"mean_{name}"]
            total_val = summary[f"total_{name}"]
            logger.info(f"Mean {name}: {mean_val:.1%} (total: {total_val:.0f})")

        if group_by_key and f"by_{group_by_key}" in summary:
            logger.info(f"\nBy {group_by_key}:")
            grouped_metrics = summary[f"by_{group_by_key}"]
            assert isinstance(grouped_metrics, dict), "Grouped metrics should be dict"
            for group_value, metrics in grouped_metrics.items():
                # Print first reward metric for each group
                first_reward_name = reward_functions[0][0]
                metric_val = metrics[first_reward_name]
                total_val = metrics[f"total_{first_reward_name}"]
                total_samples = metrics["total"]
                logger.info(f"  {group_value}: {metric_val:.1%} ({total_val:.0f}/{total_samples})")

    return {
        "summary": summary,
        "results": results,
        "config": {
            "endpoint": asdict(endpoint),
            "max_concurrent": max_concurrent,
            "reward_functions": [name for name, _ in reward_functions],
        }
    }
