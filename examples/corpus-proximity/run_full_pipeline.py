#!/usr/bin/env python3
"""Wrapper script to execute corpus-proximity pipeline with markers."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SUCCESS_MARKER = SCRIPT_DIR / ".pipeline_complete"
FAILURE_MARKER = SCRIPT_DIR / ".pipeline_failed"
LOG_PATH = SCRIPT_DIR / "pipeline.log"


logger = logging.getLogger(__name__)


def run_step(script: str, config_arg: str, *extra: str) -> None:
    cmd = [sys.executable, str(SCRIPT_DIR / script), config_arg, *extra]
    logger.info("Running %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        logger.info("STDOUT:\n%s", result.stdout)
    if result.stderr:
        logger.error("STDERR:\n%s", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed with exit code {result.returncode}")


def configure_logging() -> None:
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    LOG_PATH.unlink(missing_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python run_full_pipeline.py <config_path>")
        return 1

    config_arg = sys.argv[1]

    for marker in (SUCCESS_MARKER, FAILURE_MARKER):
        marker.unlink(missing_ok=True)

    configure_logging()

    logger.info("=" * 80)
    logger.info("Starting corpus-proximity pipeline")
    logger.info("=" * 80)

    try:
        run_step("prepare_data.py", config_arg)
        run_step("embed_chunks.py", config_arg)
        run_step("cluster_corpus.py", config_arg)
        run_step("name_clusters.py", config_arg, "--name")
    except Exception as exc:
        FAILURE_MARKER.touch()
        logger.error("Pipeline failed: %s", exc)
        return 1

    SUCCESS_MARKER.touch()
    logger.info("=" * 80)
    logger.info("✅ Pipeline complete")
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())

