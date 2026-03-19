"""Interpretability Agent Eval.

An eval where the agent performs a mechanistic interpretability analysis task:
given a model and a behavior, find the circuit / components responsible.

Task:
- Agent is given a small transformer and a description of a behavior to explain
- Agent uses interp tools (logit lens, activation patching, attention visualization, etc.)
- Success = agent produces a correct mechanistic hypothesis + supporting evidence

Scoring:
- circuit_found: does the agent identify the correct components (heads, MLPs, layers)?
- evidence_quality: does the agent patch/ablate to confirm the hypothesis?
- hypothesis_correct: does the final answer match the ground-truth circuit?

TODO: flesh out task dataset — need (model, behavior, ground_truth_circuit) triples.
TODO: decide on environment — local Python sandbox vs Modal with GPU.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from rollouts.training.scoring import FunctionScorer

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent
TASKS_PATH = EVAL_DIR / "tasks.json"


def prepare_messages(sample_data: dict[str, Any]) -> list:
    """Prepare initial messages for the agent."""
    from rollouts.dtypes import Message

    behavior = sample_data.get("behavior", "unknown behavior")
    model_description = sample_data.get("model_description", "a small transformer")
    model_path = sample_data.get("model_path", "/workspace/model.pt")
    hint = sample_data.get("hint", "")

    system_prompt = """You are an expert mechanistic interpretability researcher.

Your task is to find the circuit responsible for a specific behavior in a transformer model.

## Available Tools
- read: Read files
- write: Write files (for saving analysis results)
- bash: Run Python scripts (TransformerLens, nnsight, activation patching available)

## Methodology
1. Replicate the behavior — verify you can observe it reliably
2. Localize — use logit lens / attention patterns to narrow down layers/heads
3. Patch — use activation patching or path patching to confirm components
4. Ablate — verify that ablating the circuit degrades the behavior
5. State your conclusion — name the components and mechanism

## How Success Is Measured
You must produce a file `hypothesis.json` with:
  {
    "components": ["layer.head or MLP notation", ...],
    "mechanism": "brief description of what they compute",
    "evidence": ["patching result 1", ...]
  }

DO NOT guess. Every claim must be backed by an intervention (patch, ablate, or probe)."""

    user_message = f"""Model: {model_description}
Model checkpoint: {model_path}

Behavior to explain:
{behavior}

{("Hint: " + hint) if hint else ""}

Find the circuit responsible. Start by replicating the behavior, then localize, patch, \
and ablate until you can name the specific components."""

    return [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_message),
    ]


def score_sample(sample: Any, _context: object) -> Any:
    """Score a completed sample.

    Checks hypothesis.json for circuit components and compares against ground truth.
    """
    from rollouts.dtypes import Metric, Score

    trajectory = getattr(sample, "trajectory", None)
    if trajectory is None:
        return Score(
            metrics=(Metric("passed", 0.0, weight=1.0, metadata={"error": "no trajectory"}),)
        )

    messages = getattr(trajectory, "messages", None)
    if messages is None:
        return Score(
            metrics=(Metric("passed", 0.0, weight=1.0, metadata={"error": "no messages"}),)
        )

    # TODO: replace with structured evaluation once task format is settled.
    # The right approach: parse hypothesis.json from the sandbox filesystem,
    # compare agent's components against sample_data["ground_truth_components"].
    # For now, look for evidence of interventions in the trajectory as a proxy.

    hypothesis_written = False
    did_patch = False
    did_ablate = False

    patch_pattern = re.compile(r"activation.patch|path.patch|act_patch", re.IGNORECASE)
    ablate_pattern = re.compile(r"ablat|zero.out|mean.ablat", re.IGNORECASE)
    hypothesis_pattern = re.compile(r"hypothesis\.json", re.IGNORECASE)

    for msg in messages:
        content = getattr(msg, "content", None)
        if content is None:
            continue
        if isinstance(content, list):
            content = "\n".join(
                getattr(b, "text", "") or (b.get("text", "") if isinstance(b, dict) else "")
                for b in content
            )
        if not isinstance(content, str):
            continue

        if hypothesis_pattern.search(content):
            hypothesis_written = True
        if patch_pattern.search(content):
            did_patch = True
        if ablate_pattern.search(content):
            did_ablate = True

    # Proxy score: rewarded for doing patching + ablation + writing hypothesis.
    # TODO: replace with ground-truth component comparison.
    evidence_score = (int(did_patch) + int(did_ablate)) / 2.0
    passed = hypothesis_written and did_patch

    return Score(
        metrics=(
            Metric("passed", 1.0 if passed else 0.0, weight=1.0),
            Metric("hypothesis_written", 1.0 if hypothesis_written else 0.0, weight=0.0),
            Metric("did_patch", 1.0 if did_patch else 0.0, weight=0.0),
            Metric("did_ablate", 1.0 if did_ablate else 0.0, weight=0.0),
            Metric("evidence_score", evidence_score, weight=0.0),
        )
    )


# ── EvalSpec Definition ───────────────────────────────────────────────────────

from rollouts.eval_runner import EvalSpec

spec = EvalSpec(
    name="interp_agent",
    prepare_messages=prepare_messages,
    scorer=FunctionScorer(score_sample),
    make_environment=None,  # TODO: add Python sandbox with TransformerLens + model
    default_tasks_path=TASKS_PATH,
    per_sample_environment=False,
)


def get_spec() -> EvalSpec:
    return spec
