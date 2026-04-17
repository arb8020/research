"""SillyTavern-style roleplay eval.

Primary agent = the character (policy under evaluation, playing the card).
Responder (user-sim) = a separate LLM acting as the user persona.

For v0 we compute lorebook activation once at session start using the opening
user message as the scan buffer, then stuff activated entries into the primary's
system prompt alongside the character card. The system prompt is static for
the rest of the session.

TODO(fidelity): recompute activation each turn using the sliding scan window
(last `scan_depth` messages). Requires either a custom on_assistant_message
that rewrites the trajectory's system message, or a subclass of
DialogueEnvironment with that hook. See world_info.py for the scanner.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rollouts.core import Endpoint, Message
from rollouts.core.eval import Metric, Score
from rollouts.environments.dialogue import DialogueEnvironment
from rollouts.eval_runner import EvalSpec
from rollouts.training.scoring import FunctionScorer

try:
    from .world_info import activate, parse_lorebook, substitute_macros
except ImportError:
    # run_eval.py loads eval.py as a non-package module
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from world_info import activate, parse_lorebook, substitute_macros  # type: ignore

EVAL_DIR = Path(__file__).parent
TASKS_PATH = EVAL_DIR / "tasks.jsonl"


def load_tasks_jsonl(path: Path | str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def build_character_system_prompt(sample: dict[str, Any]) -> str:
    """Assemble the character's system prompt: activated lore + card fields.

    Approximates ST's "character-first" insertion: lore_before, description,
    personality, scenario, lore_after, example messages.
    """
    card = sample["card"]
    lorebook = sample["lorebook"]
    stats = sample["lorebook_stats"]
    char_name = card["name"]
    user_name = "User"  # TODO: lift from sample persona

    entries = parse_lorebook(lorebook)
    scan_text = sample["opening_user_msg"] + " " + card.get("first_mes", "")
    result = activate(
        entries,
        scan_text,
        token_budget=int(stats.get("token_budget", 1500)),
    )

    def sub(s: str) -> str:
        return substitute_macros(s, char_name, user_name)

    sections: list[str] = []
    if result.before:
        sections.append(f"[World Info]\n{sub(result.before)}")
    if card.get("description"):
        sections.append(f"[{char_name}'s Description]\n{sub(card['description'])}")
    if card.get("personality"):
        sections.append(f"[{char_name}'s Personality]\n{sub(card['personality'])}")
    if card.get("scenario"):
        sections.append(f"[Scenario]\n{sub(card['scenario'])}")
    if result.after:
        sections.append(f"[World Info (after)]\n{sub(result.after)}")
    if card.get("mes_example"):
        sections.append(f"[Example Dialogue]\n{sub(card['mes_example'])}")

    header = (
        f"You are {char_name}. Stay fully in character. Respond as "
        f"{char_name} would, in their voice. Do not break the fourth wall."
    )
    return header + "\n\n" + "\n\n".join(sections)


def build_user_sim_system_prompt(sample: dict[str, Any]) -> str:
    persona = sample["persona"]
    char_name = sample["card"]["name"]
    return (
        f"You are a human user chatting with an AI roleplaying as {char_name}. "
        f"Your persona: {persona}. "
        f"Respond to the character in 1-3 sentences as this persona would. "
        f"Do not narrate for {char_name}. Do not break the fourth wall. "
        f"Just reply as the user."
    )


def prepare_messages(sample: dict[str, Any]) -> list[Message]:
    """Initial messages for the primary (character) agent.

    The opening_user_msg is the seed user turn that gets the conversation
    started. After the primary replies, DialogueEnvironment invokes the
    user-sim responder on every subsequent turn.
    """
    system = build_character_system_prompt(sample)
    return [
        Message(role="system", content=system),
        Message(role="user", content=sample["opening_user_msg"]),
    ]


async def make_environment(sample: dict[str, Any]) -> DialogueEnvironment:
    model = os.getenv("SILLYTAVERN_USER_SIM_MODEL", "claude-haiku-4-5-20251001")
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    assert api_key, "ANTHROPIC_API_KEY required for user-sim responder"

    responder_endpoint = Endpoint(
        model=f"anthropic/{model}",
        base_url="https://api.anthropic.com/v1",
        api_format="anthropic-messages",
        api_key=api_key,
    )

    max_turns = sample.get("max_turns", 8)
    return DialogueEnvironment(
        responder_endpoint=responder_endpoint,
        responder_system_prompt=build_user_sim_system_prompt(sample),
        max_turns=max_turns,
    )


def score_sample(sample: Any) -> Score:
    """v0 scoring: trace-length sanity + no-op quality signal.

    Real scoring (LLM-as-judge on in-character fidelity, lore consistency,
    engagement) comes in v0.1. For now we just report that the loop ran and
    produced N turns of dialogue, which is the only signal v0 is designed to
    verify.
    """
    trajectory = getattr(sample, "trajectory", None)
    if trajectory is None:
        return Score(metrics=(Metric("ran", 0.0, weight=1.0, metadata={"error": "no trajectory"}),))

    messages = getattr(trajectory, "messages", []) or []
    n_assistant = sum(1 for m in messages if getattr(m, "role", None) == "assistant")
    n_user = sum(1 for m in messages if getattr(m, "role", None) == "user")

    return Score(
        metrics=(
            Metric("ran", 1.0, weight=1.0),
            Metric("assistant_turns", float(n_assistant), weight=0.0),
            Metric("user_turns", float(n_user), weight=0.0),
        )
    )


spec = EvalSpec(
    name="sillytavern_text",
    prepare_messages=prepare_messages,
    scorer=FunctionScorer(score_sample),
    make_environment=make_environment,
    default_tasks_path=TASKS_PATH,
    per_sample_environment=True,
    has_tools=False,  # DialogueEnvironment is dialogue-only; max_turns drives termination
)
