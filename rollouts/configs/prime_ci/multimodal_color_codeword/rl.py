"""Prime-CI multimodal color-codeword stub."""

from __future__ import annotations

from rollouts.config_status import draft

config_status = draft(
    "Prime nightly covers a multimodal color-codeword task, but we do not yet have a local multimodal RL path here."
)


def train(*_args: object, **_kwargs: object) -> dict[str, object]:
    raise NotImplementedError(
        "Multimodal color-codeword is only stubbed here. Add a local multimodal RL pathway first."
    )
