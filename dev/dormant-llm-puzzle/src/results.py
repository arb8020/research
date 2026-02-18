"""Results directory utilities.

Usage:
    from src.results import setup_run_dir, PROJECT_ROOT, RESULTS_DIR

    run_dir = setup_run_dir("my_experiment")
    # Creates: results/my_experiment/20260215_123456/
    # Returns: Path to the created directory
"""

from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def setup_run_dir(name: str) -> Path:
    """Create a timestamped run directory for an experiment.

    Args:
        name: Experiment name (e.g., "behavioral_clustering")

    Returns:
        Path to created directory (e.g., results/behavioral_clustering/20260215_123456/)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
