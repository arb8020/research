"""Alphabet Sort RL training config.

Multi-turn task: sort names alphabetically across turns.
Based on Prime-RL's alphabet-sort example.

Baseline Qwen3-4B: ~0.26 reward
After RL (100 steps): ~0.81 reward
"""

from __future__ import annotations

import random
import re
from difflib import SequenceMatcher
from typing import Any

from rollouts.dtypes import Metric, Score
from rollouts.environments.no_tools import BasicEnvironment
from rollouts.training.grpo import GRPOConfig, grpo_train

# ──────────────────────── Name Generation ───────────────────────────────────


FIRST_NAMES = [
    "Alice",
    "Bob",
    "Carol",
    "David",
    "Emma",
    "Frank",
    "Grace",
    "Henry",
    "Iris",
    "Jack",
    "Kate",
    "Leo",
    "Mia",
    "Noah",
    "Olivia",
    "Peter",
    "Quinn",
    "Rose",
    "Sam",
    "Tina",
    "Uma",
    "Victor",
    "Wendy",
    "Xander",
]

LAST_NAMES = [
    "Adams",
    "Brown",
    "Clark",
    "Davis",
    "Evans",
    "Fisher",
    "Garcia",
    "Hill",
    "Irwin",
    "Jones",
    "King",
    "Lopez",
    "Miller",
    "Nelson",
    "Ortiz",
    "Parker",
    "Quinn",
    "Roberts",
    "Smith",
    "Taylor",
    "Upton",
    "Vance",
    "Wilson",
    "Young",
]

SYSTEM_PROMPT = """\
You are sorting names alphabetically. Each turn, you'll receive new names to add to your sorted list.

Rules:
1. Sort by {sort_by} name (names are written as FirstLast with no space)
2. Output the complete sorted list after each turn
3. Mark new names with "// new name!" comment
4. Format: Put sorted names in <combined_alphabetical_sorted> tags

Example for sorting by LAST name:
Turn 1: AliceSmith
<alphabetical_sorted>
AliceSmith
</alphabetical_sorted>

Turn 2: BobJones
<combined_alphabetical_sorted>
BobJones // new name!
AliceSmith
</combined_alphabetical_sorted>
"""


def _generate_name(rng: random.Random) -> str:
    """Generate a random full name (concatenated, no space)."""
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    return f"{first}{last}"


def _get_sort_key(name: str, sort_by: str) -> str:
    """Extract sort key from concatenated name."""
    for i in range(1, len(name)):
        if name[i].isupper():
            first_name = name[:i]
            last_name = name[i:]
            break
    else:
        first_name = name
        last_name = ""

    return last_name.lower() if sort_by == "last" else first_name.lower()


def generate_alphabet_sort_prompts(
    num_episodes: int = 500,
    min_turns: int = 3,
    max_turns: int = 3,
    min_names_per_turn: int = 1,
    max_names_per_turn: int = 4,
    sort_by: str = "last",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate alphabet sort prompts (flattened to per-turn samples)."""
    rng = random.Random(seed)
    prompts = []

    for _ in range(num_episodes):
        ep_sort_by = rng.choice(["first", "last"]) if sort_by == "random" else sort_by
        num_turns = rng.randint(min_turns, max_turns)

        system = SYSTEM_PROMPT.format(sort_by=ep_sort_by.upper())
        messages = [{"role": "system", "content": system}]
        all_names = []

        for _turn_idx in range(num_turns):
            num_names = rng.randint(min_names_per_turn, max_names_per_turn)
            new_names = [_generate_name(rng) for _ in range(num_names)]
            all_names.extend(new_names)

            sorted_names = sorted(all_names, key=lambda n: _get_sort_key(n, ep_sort_by))

            new_names_str = ", ".join(new_names)
            user_msg = f"Add these names: {new_names_str}"
            messages.append({"role": "user", "content": user_msg})

            prompts.append({
                "messages": list(messages),  # Copy
                "expected_sorted": sorted_names,
            })

            # Add placeholder for assistant response
            messages.append({"role": "assistant", "content": "[TO BE GENERATED]"})

    return prompts


# ──────────────────────── Score Function ────────────────────────────────────


def _extract_sorted_list(response: str) -> list[str]:
    """Extract names from response."""
    match = re.search(r"<(?:combined_)?alphabetical_sorted>(.*?)</", response, re.DOTALL)
    if match:
        content = match.group(1)
    else:
        content = response

    names = re.findall(r"([A-Z][a-z]+[A-Z][a-z]+)", content)
    return names


def _compute_list_similarity(predicted: list[str], expected: list[str]) -> float:
    """Compute similarity between two lists."""
    if not expected:
        return 1.0 if not predicted else 0.0

    if predicted == expected:
        return 1.0

    pred_str = " ".join(predicted)
    exp_str = " ".join(expected)
    return SequenceMatcher(None, pred_str, exp_str).ratio()


def alphabet_sort_score_fn(sample: Any) -> Score:
    """Score function for alphabet sort.

    Uses power-scaled similarity (similarity^8) like Prime-RL.
    """
    expected = sample.metadata.get("expected_sorted", [])
    response = sample.response if hasattr(sample, "response") else ""

    predicted = _extract_sorted_list(response)
    similarity = _compute_list_similarity(predicted, expected)

    # Apply power scaling like Prime-RL (similarity^8)
    reward = similarity**8

    return Score(
        metrics=(
            Metric("reward", reward, weight=1.0),
            Metric("similarity", similarity, weight=0.0),
            Metric("exact_match", 1.0 if predicted == expected else 0.0, weight=0.0),
        )
    )


# ──────────────────────── Training ──────────────────────────────────────────


def train(
    config: GRPOConfig | None = None,
    num_episodes: int = 500,
) -> dict[str, Any]:
    """Run alphabet sort RL training.

    Args:
        config: Training config. If None, uses defaults from Prime-RL.
        num_episodes: Number of episodes to generate.

    Returns:
        Dict with metrics_history.
    """
    if config is None:
        config = GRPOConfig(
            experiment_name="alphabet_sort_grpo",
            lr=1e-6,
            n_samples_per_prompt=8,
            temperature=0.7,
            num_steps=100,
            batch_size=4,
            max_seq_len=1024,
            max_tokens=256,
        )

    prompts = generate_alphabet_sort_prompts(num_episodes=num_episodes)

    return grpo_train(
        config=config,
        prompts=prompts,
        score_fn=alphabet_sort_score_fn,
        environment_cls=BasicEnvironment,
    )
