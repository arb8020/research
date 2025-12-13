"""BrowseComp eval with GPT-4o, graded by GPT-4o-mini.

Usage:
    cd rollouts
    python -c "from examples.eval.browsecomp.eval_gpt4o_01_01 import config; \
               from examples.eval.browsecomp.base_config import evaluate_browsecomp; \
               evaluate_browsecomp(config)"
"""

from pathlib import Path

from .base_config import (
    BrowseCompConfig,
    DatasetConfig,
    EndpointConfig,
    EvalRunConfig,
    GraderConfig,
    OutputConfig,
)

config = BrowseCompConfig(
    endpoint=EndpointConfig(
        provider="openai",
        model="gpt-4o",
    ),
    grader=GraderConfig(
        provider="openai",
        model="gpt-4o-mini",
    ),
    dataset=DatasetConfig(
        max_samples=10,  # Start small for testing
        seed=42,
    ),
    run=EvalRunConfig(
        max_concurrent=4,
        verbose=True,
    ),
    output=OutputConfig(
        save_dir=Path("results"),
        experiment_name="browsecomp_gpt4o",
    ),
)
