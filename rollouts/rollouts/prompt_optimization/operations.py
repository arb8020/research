"""Low-level GEPA operations.

Pure functions for individual operations.
Following: push ifs up, fors down - these are the non-branchy helpers.
"""

import logging
from collections.abc import Sequence
from typing import Any

import trio

from ..dtypes import Endpoint, Message
from .state import GEPAState
from .types import Candidate

logger = logging.getLogger(__name__)


# ─── Mutation Prompt ──────────────────────────────────────────────────────────

REFLECTION_PROMPT = """I provided an assistant with the following instructions to perform a task:
```
{current_instruction}
```

The following are examples of inputs provided to the assistant, the assistant's outputs, and feedback on how the outputs could be improved:

{examples}

Your task is to write improved instructions for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description.

Read all the assistant outputs and the corresponding feedback. Identify all niche and domain-specific factual information and include it in the instruction. If the assistant utilized a generalizable strategy, include that as well.

Provide the new instructions within ``` blocks."""


def format_reflective_examples(examples: list[dict]) -> str:
    """Format reflective examples for the mutation prompt.

    Args:
        examples: List of dicts with Inputs, Generated Outputs, Feedback

    Returns:
        Formatted string for prompt
    """
    parts = []
    for i, ex in enumerate(examples, 1):
        part = f"## Example {i}\n"
        part += f"### Inputs\n{ex.get('Inputs', '')}\n\n"
        part += f"### Generated Outputs\n{ex.get('Generated Outputs', '')}\n\n"
        part += f"### Feedback\n{ex.get('Feedback', '')}\n"
        parts.append(part)
    return "\n".join(parts)


def extract_instruction_from_response(response: str) -> str:
    """Extract instruction text from LLM response.

    Looks for text within ``` blocks.

    Args:
        response: Raw LLM response

    Returns:
        Extracted instruction text
    """
    # Find content between ``` blocks
    start = response.find("```")
    if start == -1:
        return response.strip()

    start += 3
    # Skip optional language specifier
    if response[start : start + 1].isalpha():
        newline = response.find("\n", start)
        if newline != -1:
            start = newline + 1

    end = response.rfind("```")
    if end <= start:
        # No closing block, take everything after opening
        return response[start:].strip()

    return response[start:end].strip()


async def propose_mutation(
    candidate: Candidate,
    component: str,
    reflective_data: list[dict],
    endpoint: Endpoint,
) -> str:
    """Use LLM to propose improved text for one component.

    Pure async function - no side effects except LLM call.

    Args:
        candidate: Current candidate
        component: Which component to mutate
        reflective_data: Feedback items from make_reflective_dataset
        endpoint: LLM endpoint for mutation proposal

    Returns:
        New text for the component
    """
    from ..agents import rollout
    from ..dtypes import Actor, Trajectory

    current_instruction = candidate[component]
    examples_text = format_reflective_examples(reflective_data)

    prompt = REFLECTION_PROMPT.format(
        current_instruction=current_instruction,
        examples=examples_text,
    )

    messages = [Message(role="user", content=prompt)]
    trajectory = Trajectory(messages=messages)
    actor = Actor(trajectory=trajectory, endpoint=endpoint, tools=[])

    # Silent handler
    async def silent(_: Any) -> None:
        await trio.lowlevel.checkpoint()

    try:
        new_actor = await rollout(actor, on_chunk=silent)
        last_msg = new_actor.trajectory.messages[-1]

        # Extract text content
        if isinstance(last_msg.content, str):
            response = last_msg.content
        elif isinstance(last_msg.content, list):
            response = ""
            for block in last_msg.content:
                if hasattr(block, "text"):
                    response += block.text
        else:
            logger.warning("Unexpected content type, returning original")
            return current_instruction

        # Extract instruction from response
        new_instruction = extract_instruction_from_response(response)
        return new_instruction

    except Exception as e:
        logger.warning(f"Mutation failed: {e}, returning original")
        return current_instruction


# ─── Selection ────────────────────────────────────────────────────────────────


def select_from_pareto_front(state: GEPAState) -> int:
    """Select candidate index from Pareto front.

    Uses state.rng for randomness.

    Args:
        state: Current GEPA state

    Returns:
        Index of selected candidate
    """
    if not state.pareto_front:
        return 0
    return state.rng.choice(list(state.pareto_front))


def dominates(scores_a: dict[int, float], scores_b: dict[int, float]) -> bool:
    """Check if candidate A dominates candidate B.

    A dominates B if A is >= B on all examples and > B on at least one.

    Args:
        scores_a: Per-example scores for candidate A
        scores_b: Per-example scores for candidate B

    Returns:
        True if A dominates B
    """
    dominated = False
    for ex_idx in scores_a:
        a_score = scores_a.get(ex_idx, 0.0)
        b_score = scores_b.get(ex_idx, 0.0)
        if a_score < b_score:
            return False  # A is worse on this example
        if a_score > b_score:
            dominated = True  # A is better on this example
    return dominated


def update_pareto_front(state: GEPAState, new_idx: int) -> None:
    """Update Pareto front after adding new candidate.

    Mutates state.pareto_front.

    Args:
        state: GEPA state to update
        new_idx: Index of newly added candidate
    """
    new_scores = state.val_scores.get(new_idx, {})
    if not new_scores:
        return

    # Check if new candidate is dominated by any existing front member
    is_dominated = False
    for front_idx in list(state.pareto_front):
        front_scores = state.val_scores.get(front_idx, {})
        if dominates(front_scores, new_scores):
            is_dominated = True
            break

    if is_dominated:
        return  # Don't add dominated candidate

    # New candidate is non-dominated, add to front
    # Remove any front members now dominated by new candidate
    dominated_by_new = set()
    for front_idx in state.pareto_front:
        front_scores = state.val_scores.get(front_idx, {})
        if dominates(new_scores, front_scores):
            dominated_by_new.add(front_idx)

    state.pareto_front -= dominated_by_new
    state.pareto_front.add(new_idx)


# ─── Minibatch Sampling ───────────────────────────────────────────────────────


def sample_minibatch(
    dataset: Sequence[dict],
    state: GEPAState,
    size: int,
) -> list[int]:
    """Sample minibatch indices from dataset.

    Uses state.rng for randomness.

    Args:
        dataset: Full dataset
        state: GEPA state (for RNG)
        size: Number of samples to select

    Returns:
        List of indices into dataset
    """
    n = len(dataset)
    size = min(size, n)
    return state.rng.sample(range(n), size)
