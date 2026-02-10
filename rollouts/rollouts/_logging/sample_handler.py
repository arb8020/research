"""Per-sample JSONL routing handler for eval logging.

Routes log records to per-sample files based on extra["sample_id"].
Eagerly closes files on sample_end to bound open FDs to max_concurrent.

Tiger Style: Bounded file handles, explicit lifecycle.
"""

import logging
from pathlib import Path
from typing import TextIO


class SampleRoutingHandler(logging.Handler):
    """Routes log records to per-sample JSONL files.

    Each record with a sample_id in its extra dict gets written to
    {output_dir}/samples/{sample_id}.jsonl. Records without sample_id
    are silently dropped.

    File handles are opened lazily on first write and closed eagerly
    when a sample_end message is seen. Peak open FDs = max concurrent
    samples, not total samples.

    Usage:
        handler = SampleRoutingHandler(output_dir)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

        # In your code:
        logger.info("llm_call", extra={"sample_id": "sample_0001", ...})
        logger.info("sample_end", extra={"sample_id": "sample_0001"})  # Closes file
    """

    def __init__(self, output_dir: Path) -> None:
        super().__init__()
        self.samples_dir = output_dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, TextIO] = {}

    def emit(self, record: logging.LogRecord) -> None:
        sample_id = getattr(record, "sample_id", None)
        if sample_id is None:
            return

        try:
            f = self._get_or_open(sample_id)
            f.write(self.format(record) + "\n")
            f.flush()

            # Eagerly close on sample_end — QueueHandler serializes
            # writes so this is safe.
            if record.getMessage() == "sample_end":
                self._close_sample(sample_id)
        except Exception:
            self.handleError(record)

    def _get_or_open(self, sample_id: str) -> TextIO:
        f = self._files.get(sample_id)
        if f is not None:
            return f
        path = self.samples_dir / f"{sample_id}.jsonl"
        f = open(path, "a")
        self._files[sample_id] = f
        return f

    def _close_sample(self, sample_id: str) -> None:
        f = self._files.pop(sample_id, None)
        if f is not None:
            f.close()

    def close(self) -> None:
        """Close all open file handles on shutdown."""
        for f in self._files.values():
            f.close()
        self._files.clear()
        super().close()
