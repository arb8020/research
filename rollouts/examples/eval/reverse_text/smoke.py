"""Reverse text eval smoke test - API endpoint.

Tests the unified eval runner against Anthropic API.

Run:
    python -m rollouts.eval.run --config examples/eval/reverse_text/smoke.py
    python -m rollouts.eval.run --config examples/eval/reverse_text/smoke.py --model claude-opus-4-20250514
    python -m rollouts.eval.run --config examples/eval/reverse_text/smoke.py --limit 3
"""

from rollouts.dtypes import Message, Metric, Score
from rollouts.eval import EndpointConfig, EvalOutputConfig, EvalRunConfig
from rollouts.training.types import Sample

# =============================================================================
# Endpoint Configuration
# =============================================================================

endpoint = EndpointConfig(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    temperature=0.0,
    max_tokens=256,
)

# =============================================================================
# Eval Configuration
# =============================================================================

run = EvalRunConfig(
    max_concurrent=5,
    max_samples=10,
    max_turns=1,
    verbose=True,
)

output = EvalOutputConfig(
    experiment_name="reverse_text_smoke",
)

# =============================================================================
# Tasks
# =============================================================================

# Simple test cases
tasks = [
    {"text": "hello"},
    {"text": "world"},
    {"text": "python"},
    {"text": "machine learning"},
    {"text": "artificial intelligence"},
    {"text": "deep neural network"},
    {"text": "transformer architecture"},
    {"text": "attention mechanism"},
    {"text": "gradient descent"},
    {"text": "backpropagation"},
]


# =============================================================================
# Eval Functions
# =============================================================================


def prepare_messages(sample: dict) -> list[Message]:
    """Convert task to initial messages."""
    text = sample["text"]
    return [
        Message(
            role="user",
            content=f"Reverse the following text character-by-character. "
            f"Put your answer in <reversed_text> tags.\n\n"
            f"Text to reverse: {text}",
        ),
    ]


def score_fn(sample: Sample) -> Score:
    """Score the response by checking if reversal is correct."""
    import re

    # Get expected reversal from input
    input_data = sample.input
    expected = input_data["text"][::-1]

    # Extract response
    response = sample.response if hasattr(sample, "response") else ""
    if not response and sample.trajectory:
        # Get last assistant message
        for msg in reversed(sample.trajectory.messages):
            if msg.role == "assistant":
                content = msg.content
                if isinstance(content, str):
                    response = content
                elif isinstance(content, list) and content:
                    response = content[0].text if hasattr(content[0], "text") else str(content[0])
                break

    # Parse <reversed_text> tags
    match = re.search(r"<reversed_text>\s*(.*?)\s*</reversed_text>", response, re.DOTALL)
    if match:
        parsed = match.group(1).strip()
    else:
        parsed = response.strip()

    # Check correctness
    exact_match = parsed == expected

    # Compute similarity (for partial credit)
    from difflib import SequenceMatcher

    similarity = SequenceMatcher(None, parsed, expected).ratio()

    return Score(
        metrics=(
            Metric("exact_match", 1.0 if exact_match else 0.0, weight=1.0),
            Metric("similarity", similarity, weight=0.0),
            Metric("has_tags", 1.0 if match else 0.0, weight=0.0),
        )
    )
