"""Load tau2 tasks per domain.

tau2 ships task JSON inside the `tau2` Python package (`data/tau2/domains/<domain>/tasks.json`).
Each domain's `get_tasks()` loader returns `list[tau2.data_model.tasks.Task]`.
Nothing here depends on cloning the GitHub repo — tau2's package data is
authoritative once the package is installed.

The telecom domain has multiple splits; the `base` split (114 tasks) is the
official benchmark. Other domains have a single task set.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)

Tau2Domain = Literal["airline", "retail", "telecom", "banking_knowledge"]


def load_domain_tasks(
    domain: Tau2Domain,
    split: str | None = "base",
) -> list[Any]:
    """Return tau2 Task objects for the given domain.

    Args:
        domain: one of airline, retail, telecom, banking_knowledge
        split: for telecom, one of {"base" (114 official), "small" (20),
            "train" (74), "test" (40), "full" (2285)}. Ignored for the
            other domains (their loaders take split=None).

    Returns: list[tau2.data_model.tasks.Task]
    """
    if domain == "airline":
        from tau2.domains.airline.environment import get_tasks

        return get_tasks()
    if domain == "retail":
        from tau2.domains.retail.environment import get_tasks

        return get_tasks()
    if domain == "telecom":
        from tau2.domains.telecom.environment import get_tasks

        return get_tasks(task_split_name=split or "base")
    if domain == "banking_knowledge":
        from tau2.domains.banking_knowledge.environment import get_tasks

        return get_tasks(task_split_name=split)
    raise ValueError(f"Unknown domain: {domain!r}")


DEFAULT_USER_ENDPOINT: dict[str, str] = {
    "model": "openai/gpt-4o-mini",
    "base_url": "https://api.openai.com/v1",
    "api_format": "openai-completions",
    "api_key_env": "OPENAI_API_KEY",
}


def build_sample_rows(
    domain: Tau2Domain,
    split: str | None = "base",
    limit: int | None = None,
    user_endpoint: dict[str, str] | None = None,
    persona_config: dict[str, Any] | None = None,
    solo_mode: bool = False,
    retrieval_variant: str | None = None,
    retrieval_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build dataset rows for the eval runner — one per tau2 task.

    Each row carries:
      - task_id: identifier for logging / deduplication
      - domain: passed through to make_environment
      - task_json: serialized tau2 Task (avoids repickling BaseModels across
        the eval runner's JSON boundary)
      - user_endpoint: dict describing the user-simulator endpoint
        (model, base_url, api_format, api_key_env). The same block is read
        by both make_environment (to construct the live env) and
        prepare_messages (to call the user sim once for the bootstrap).
        Living on the row makes the dataset self-describing — every input
        the per-sample pipeline needs is recoverable from this dict.
      - persona_config: optional dict matching tau2's PersonaConfig schema
        (verbosity, interrupt_tendency, ...). None means tau2 default
        behavior (no runtime persona overrides). Same value flows to both
        the bootstrap user-sim in prepare_messages and the live env's
        UserSimulator, so they share an identical system prompt.

    `user_endpoint` defaults to `DEFAULT_USER_ENDPOINT` (gpt-4o-mini on
    the OpenAI API). Override per config to swap models/providers.
    """
    tasks = load_domain_tasks(domain, split=split)
    if limit is not None:
        tasks = tasks[:limit]
    # In solo_mode there's no user simulator at all, so the user_endpoint
    # block is meaningless. Drop it from the row to avoid implying it
    # gets used (eval.py reads it conditionally on solo_mode).
    persona = dict(persona_config) if persona_config else None
    ep = None if solo_mode else dict(user_endpoint or DEFAULT_USER_ENDPOINT)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        row: dict[str, Any] = {
            "task_id": task.id,
            "domain": domain,
            "task_json": task.model_dump_json(exclude_none=True),
        }
        if ep is not None:
            row["user_endpoint"] = ep
        if persona is not None:
            row["persona_config"] = persona
        if solo_mode:
            row["solo_mode"] = True
        if retrieval_variant is not None:
            row["retrieval_variant"] = retrieval_variant
        if retrieval_kwargs is not None:
            row["retrieval_kwargs"] = dict(retrieval_kwargs)
        rows.append(row)
    return rows
