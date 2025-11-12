"""Simple 2048 game environment for testing Prime integration.

This is a minimal implementation to test the Prime adapter without needing
the full hud-text-2048 package from the Hub.

Based on: https://app.primeintellect.ai/dashboard/environments/hud/hud-text-2048
"""

from datasets import Dataset
from verifiers import SingleTurnEnv, Rubric, Parser


class Simple2048Parser(Parser):
    """Parse model output to extract move direction."""

    def parse(self, response: str) -> str:
        """Extract move from response (up, down, left, right)."""
        response_lower = response.lower()

        # Look for direction keywords
        if "up" in response_lower:
            return "up"
        elif "down" in response_lower:
            return "down"
        elif "left" in response_lower:
            return "left"
        elif "right" in response_lower:
            return "right"
        else:
            # Default or parse error
            return "invalid"


class Simple2048Rubric(Rubric):
    """Rubric for grading 2048 moves."""

    def __init__(self):
        super().__init__(parser=Simple2048Parser())

    def grade(self, predicted: str, ground_truth: str, **kwargs) -> float:
        """Grade the move.

        For this simple version, we just check exact match.
        Real version would simulate the game state.
        """
        if predicted == ground_truth:
            return 1.0
        elif predicted == "invalid":
            return 0.0  # Failed to parse
        else:
            return 0.5  # Valid move but not optimal


def create_simple_2048_env(num_samples: int = 10) -> SingleTurnEnv:
    """Create a simple 2048 environment for testing.

    Args:
        num_samples: Number of test samples

    Returns:
        SingleTurnEnv ready for evaluation
    """
    # Create synthetic dataset
    # In reality, these would be actual game states
    samples = []
    for i in range(num_samples):
        # Alternate between different scenarios
        if i % 4 == 0:
            prompt = "Board state: [[2,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]. What move should you make? (up/down/left/right)"
            answer = "left"  # Optimal: consolidate left
        elif i % 4 == 1:
            prompt = "Board state: [[2,2,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]. What move should you make?"
            answer = "left"  # Merge the twos
        elif i % 4 == 2:
            prompt = "Board state: [[4,2,0,0],[0,0,0,0],[0,0,0,0],[2,0,0,0]]. What move should you make?"
            answer = "down"  # Consolidate vertically
        else:
            prompt = "Board state: [[2,4,2,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]. What move should you make?"
            answer = "left"  # Consolidate left

        samples.append({
            "question": prompt,
            "answer": answer,
        })

    dataset = Dataset.from_dict({
        "question": [s["question"] for s in samples],
        "answer": [s["answer"] for s in samples],
    })

    env = SingleTurnEnv(
        dataset=dataset,
        rubric=Simple2048Rubric(),
        parser=Simple2048Parser(),
        system_prompt="You are playing 2048. Analyze the board state and choose the best move.",
    )

    return env
