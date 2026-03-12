"""Event journal implementations."""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import Event, EventKind, utc_now


@dataclass
class InMemoryEventJournal:
    """Simple append-only in-memory event journal.

    This is intentionally tiny. Argus should define supervisor semantics first
    before committing to persistence or a transport protocol.
    """

    _events: list[Event] = field(default_factory=list)

    def append(
        self,
        *,
        run_id: str,
        kind: EventKind,
        attempt_id: str | None = None,
        payload: dict | None = None,
    ) -> Event:
        """Append one event and return it."""
        assert run_id, "run_id cannot be empty"
        seq = len(self._events)
        event = Event(
            seq=seq,
            run_id=run_id,
            kind=kind,
            ts=utc_now(),
            attempt_id=attempt_id,
            payload={} if payload is None else payload,
        )
        self._events.append(event)
        return event

    def list_events(self, run_id: str) -> list[Event]:
        """Return all events for one run."""
        assert run_id, "run_id cannot be empty"
        return [event for event in self._events if event.run_id == run_id]

    def since(self, run_id: str, cursor: int) -> list[Event]:
        """Return events for one run with seq greater than cursor."""
        assert cursor >= -1, "cursor must be >= -1"
        return [event for event in self.list_events(run_id) if event.seq > cursor]
