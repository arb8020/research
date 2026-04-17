"""Normalize raw ST cards + lorebooks from raw/ into tasks.jsonl.

One scenario per (card, lorebook) pair. Cards are Character Card V2 JSON
(`spec: "chara_card_v2"`). Lorebooks are ST world-info exports.

Ruby's card ships with an embedded character_book (227 entries); we strip
that and use the standalone RWBY world-info instead so all three scenarios
have the same shape.

TODO: fetch from chub.ai if raw/ is empty. See chub.ai API docs for the
`/api/characters/download` and world-info download endpoints. Out of scope
for v0 since the user dropped files in manually.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).parent
RAW_DIR = EVAL_DIR / "raw"
TASKS_PATH = EVAL_DIR / "tasks.jsonl"

# Hand-picked pairings. Each franchise has one card + one lorebook.
PAIRINGS = [
    {
        "task_id": "rwby_ruby",
        "card_file": "main_ruby-10cd6fc6_spec_v2.json",
        "lorebook_file": "main_RWBY_world_info (1).json",
        "persona": "a new transfer student at Beacon Academy, curious and slightly nervous",
        "opening_user_msg": "Hey! You must be Ruby, right? Mind if I sit here?",
    },
    {
        "task_id": "genshin_raiden",
        "card_file": "main_raiden-shogun-and-ei_spec_v2.json",
        "lorebook_file": "main_Genshin Impact all characters and locations_world_info.json",
        "persona": "a traveler from a distant nation seeking an audience",
        "opening_user_msg": "Your Excellency, I have traveled far to speak with you about the Vision Hunt Decree.",
    },
    {
        "task_id": "mha_momo",
        "card_file": "main_momo-yaoyorozu-db4dccd7_spec_v2.json",
        "lorebook_file": "main_My Hero Academia Full World Info_world_info.json",
        "persona": "a classmate at UA asking Momo for study help before finals",
        "opening_user_msg": "Momo! Do you have a sec? I'm completely lost on the hero law readings.",
    },
]


@dataclass
class CardFields:
    name: str
    description: str
    personality: str
    scenario: str
    first_mes: str
    mes_example: str
    system_prompt: str


def load_card(path: Path) -> CardFields:
    raw = json.loads(path.read_text())
    assert raw.get("spec") == "chara_card_v2", f"expected spec_v2 in {path.name}"
    d = raw["data"]
    return CardFields(
        name=str(d.get("name", "")),
        description=str(d.get("description", "") or ""),
        personality=str(d.get("personality", "") or ""),
        scenario=str(d.get("scenario", "") or ""),
        first_mes=str(d.get("first_mes", "") or ""),
        mes_example=str(d.get("mes_example", "") or ""),
        system_prompt=str(d.get("system_prompt", "") or ""),
    )


def load_lorebook_raw(path: Path) -> dict[str, Any]:
    """Return the raw lorebook JSON. Activation parsing happens in world_info.py."""
    return json.loads(path.read_text())


def build_task(pairing: dict[str, str]) -> dict[str, Any]:
    card_path = RAW_DIR / pairing["card_file"]
    lore_path = RAW_DIR / pairing["lorebook_file"]
    assert card_path.exists(), f"missing card: {card_path}"
    assert lore_path.exists(), f"missing lorebook: {lore_path}"

    card = load_card(card_path)
    lorebook = load_lorebook_raw(lore_path)

    entries = lorebook.get("entries", {})
    n_entries = len(entries) if isinstance(entries, (dict, list)) else 0

    return {
        "task_id": pairing["task_id"],
        "card": asdict(card),
        "lorebook": lorebook,
        "lorebook_stats": {
            "n_entries": n_entries,
            "scan_depth": lorebook.get("scan_depth", 2),
            "token_budget": lorebook.get("token_budget", 1500),
        },
        "persona": pairing["persona"],
        "opening_user_msg": pairing["opening_user_msg"],
        "source": {
            "card_file": pairing["card_file"],
            "lorebook_file": pairing["lorebook_file"],
        },
    }


def main() -> None:
    tasks = [build_task(p) for p in PAIRINGS]
    with open(TASKS_PATH, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"wrote {len(tasks)} tasks to {TASKS_PATH}")
    for t in tasks:
        st = t["lorebook_stats"]
        print(
            f"  {t['task_id']}: char='{t['card']['name']}' "
            f"entries={st['n_entries']} scan_depth={st['scan_depth']} "
            f"budget={st['token_budget']}"
        )


if __name__ == "__main__":
    main()
