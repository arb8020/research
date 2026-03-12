"""Prime-CI ACEReason math stub."""

from __future__ import annotations

from rollouts.config_status import draft

config_status = draft(
    "Prime nightly covers ACEReason math, but we do not yet have a canonical local config path for it."
)


def train(*_args: object, **_kwargs: object) -> dict[str, object]:
    raise NotImplementedError(
        "ACEReason math is only stubbed here. Add the local dataset/env/backend path first."
    )
