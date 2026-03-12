"""Prime-CI Wordle stub."""

from __future__ import annotations

from rollouts.config_status import draft

config_status = draft(
    "Prime nightly covers Wordle, but we do not yet have a trustworthy local Wordle environment."
)


def train(*_args: object, **_kwargs: object) -> dict[str, object]:
    raise NotImplementedError(
        "Wordle is only stubbed here. Port or build a local Wordle environment first."
    )
