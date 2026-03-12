"""Prime-CI wiki-search stub."""

from __future__ import annotations

from rollouts.config_status import draft

config_status = draft(
    "Prime nightly covers wiki-search, but local support would need tools, retrieval, and judge wiring."
)


def train(*_args: object, **_kwargs: object) -> dict[str, object]:
    raise NotImplementedError(
        "Wiki-search is only stubbed here. Build the local tool/judge/resource path first."
    )
