"""Experiment runner framework.

Experiments are defined as JSON configs with:
- id: unique identifier
- hypothesis: which hypothesis this tests (H-XX)
- model: which model(s) to test
- prompts: list of prompts to try
- expected: what we expect to see if hypothesis is correct

Results are saved to /results/experiments/{id}.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from . import client

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path(__file__).parent.parent / "claude-manual" / "experiments"
RESULTS_DIR = Path(__file__).parent.parent / "claude-manual" / "results" / "experiments"


@dataclass
class ExperimentResult:
    """Result of running an experiment."""

    id: str
    hypothesis: str
    model: str
    prompts: list[str]
    responses: list[str]
    timestamp: str
    duration_seconds: float
    notes: str = ""

    def save(self) -> Path:
        """Save results to disk."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"{self.id}.json"
        path.write_text(json.dumps({
            "id": self.id,
            "hypothesis": self.hypothesis,
            "model": self.model,
            "prompts": self.prompts,
            "responses": self.responses,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "notes": self.notes,
        }, indent=2))
        logger.info(f"Saved results to {path}")
        return path


def load_experiment(experiment_id: str) -> dict[str, Any]:
    """Load an experiment config."""
    path = EXPERIMENTS_DIR / f"{experiment_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Experiment not found: {path}")
    return json.loads(path.read_text())


def load_result(experiment_id: str) -> dict[str, Any] | None:
    """Load experiment results if they exist."""
    path = RESULTS_DIR / f"{experiment_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


async def run_experiment(config: dict[str, Any]) -> ExperimentResult:
    """Run an experiment from config.

    Config format:
    {
        "id": "exp-001",
        "hypothesis": "H-01",
        "model": "dormant-model-1",  # or list for multi-model
        "prompts": ["prompt1", "prompt2"],
        "expected": "description of expected behavior",
        "notes": "optional notes"
    }
    """
    import time

    start = time.time()
    experiment_id = config["id"]
    hypothesis = config["hypothesis"]
    models = config["model"] if isinstance(config["model"], list) else [config["model"]]
    prompts = config["prompts"]

    all_results = []

    for model in models:
        logger.info(f"Running experiment {experiment_id} on {model}")
        responses = await client.batch_chat(
            prompts,
            model=model,
            experiment_id=experiment_id,
        )

        result = ExperimentResult(
            id=f"{experiment_id}-{model}" if len(models) > 1 else experiment_id,
            hypothesis=hypothesis,
            model=model,
            prompts=prompts,
            responses=responses,
            timestamp=datetime.now().isoformat(),
            duration_seconds=time.time() - start,
            notes=config.get("notes", ""),
        )
        result.save()
        all_results.append(result)

    return all_results[0] if len(all_results) == 1 else all_results


def list_experiments() -> list[str]:
    """List all available experiment configs."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    return [p.stem for p in EXPERIMENTS_DIR.glob("*.json")]


def list_results() -> list[str]:
    """List all experiment results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return [p.stem for p in RESULTS_DIR.glob("*.json")]
