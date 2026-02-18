"""Wrapper around jsinfer client with logging and caching.

TODO: Add exponential backoff/retry logic for 429 rate limit errors.
      The jsinfer client polls every 1s with no backoff, causing failures
      on 671B model experiments. Need to wrap client methods with retry.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from jsinfer import (
    ActivationsRequest,
    BatchInferenceClient,
    ChatCompletionRequest,
    Message,
)

load_dotenv()

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
API_KEY = os.environ.get("JSINFER_API_KEY", "REDACTED")


def get_client() -> BatchInferenceClient:
    """Get a configured client."""
    client = BatchInferenceClient()
    client.set_api_key(API_KEY)
    return client


def _hash_request(model: str, messages: list[dict]) -> str:
    """Hash a request for caching."""
    data = json.dumps({"model": model, "messages": messages}, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


async def chat(
    messages: list[dict[str, str]],
    model: str = "dormant-model-1",
    cache: bool = True,
    experiment_id: str | None = None,
) -> str:
    """Send a chat completion request.

    Args:
        messages: List of {"role": ..., "content": ...} dicts
        model: Model name
        cache: If True, cache results to avoid duplicate API calls
        experiment_id: Optional ID to tag this request in logs

    Returns:
        Assistant response content
    """
    request_hash = _hash_request(model, messages)
    cache_path = RESULTS_DIR / "cache" / f"{request_hash}.json"

    # Check cache
    if cache and cache_path.exists():
        logger.info(f"Cache hit: {request_hash}")
        cached = json.loads(cache_path.read_text())
        return cached["response"]

    client = get_client()

    msg_objects = [Message(role=m["role"], content=m["content"]) for m in messages]

    results = await client.chat_completions(
        [ChatCompletionRequest(custom_id=request_hash, messages=msg_objects)],
        model=model,
    )

    response = results[request_hash].messages[0].content

    # Cache result
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "model": model,
            "messages": messages,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "experiment_id": experiment_id,
        }, indent=2))

    # Log to results
    log_path = RESULTS_DIR / "chat_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "messages": messages,
            "response": response,
            "experiment_id": experiment_id,
            "hash": request_hash,
        }) + "\n")

    return response


async def activations(
    messages: list[dict[str, str]],
    module_names: list[str],
    model: str = "dormant-model-1",
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Get model activations for a prompt.

    Args:
        messages: List of {"role": ..., "content": ...} dicts
        module_names: List of module names to capture (e.g., "model.layers.0.mlp.down_proj")
        model: Model name
        experiment_id: Optional ID to tag this request

    Returns:
        Dict mapping module names to activation data
    """
    client = get_client()

    request_hash = _hash_request(model, messages)
    msg_objects = [Message(role=m["role"], content=m["content"]) for m in messages]

    results = await client.activations(
        [ActivationsRequest(
            custom_id=request_hash,
            messages=msg_objects,
            module_names=module_names,
        )],
        model=model,
    )

    # Log
    log_path = RESULTS_DIR / "activations_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "messages": messages,
            "module_names": module_names,
            "experiment_id": experiment_id,
            "hash": request_hash,
        }) + "\n")

    return results[request_hash]


async def batch_chat(
    prompts: list[str],
    model: str = "dormant-model-1",
    experiment_id: str | None = None,
) -> list[str]:
    """Send multiple single-turn prompts in a batch.

    Args:
        prompts: List of user messages
        model: Model name
        experiment_id: Optional ID to tag these requests

    Returns:
        List of assistant responses (in same order as prompts)
    """
    client = get_client()

    requests = []
    for i, prompt in enumerate(prompts):
        requests.append(ChatCompletionRequest(
            custom_id=f"{experiment_id or 'batch'}-{i:04d}",
            messages=[Message(role="user", content=prompt)],
        ))

    results = await client.chat_completions(requests, model=model)

    responses = []
    for i in range(len(prompts)):
        key = f"{experiment_id or 'batch'}-{i:04d}"
        responses.append(results[key].messages[0].content)

    # Log batch
    log_path = RESULTS_DIR / "batch_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "prompts": prompts,
            "responses": responses,
            "experiment_id": experiment_id,
        }) + "\n")

    return responses


# Convenience for running async in scripts
def run(coro):
    """Run an async function from sync context."""
    return asyncio.run(coro)
