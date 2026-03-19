"""Reverse text eval - SGLang endpoint.

Tests the unified eval runner against a local SGLang server.

Prerequisites:
    # Start SGLang server
    python -m sglang.launch_server --model Qwen/Qwen2.5-7B-Instruct --port 30000

Run:
    python -m rollouts.eval.run --config examples/eval/reverse_text/sglang.py
    python -m rollouts.eval.run --config examples/eval/reverse_text/sglang.py --limit 5

With provisioning (not yet implemented):
    python -m rollouts.eval.run --config examples/eval/reverse_text/sglang.py --provision --hardware-provider runpod
"""

from rollouts.core import Message, Metric, Score
from rollouts.eval import (
    EndpointConfig,
    EvalOutputConfig,
    EvalRunConfig,
    InferenceServerConfig,
)
from rollouts.training.scoring import FunctionScorer
from rollouts.training.types import AttemptResult

# =============================================================================
# Endpoint Configuration
# =============================================================================

endpoint = EndpointConfig(
    provider="sglang",
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://localhost:30000/v1",  # Connect to running server
    temperature=0.0,
    max_tokens=256,
)

# =============================================================================
# Hardware Configuration (for provisioning)
# =============================================================================

# Uncomment to enable auto-provisioning
# hardware = HardwareConfig(
#     gpu_type="A100",
#     gpu_count=1,
#     provider="runpod",
# )

server = InferenceServerConfig(
    port=30000,
    mem_fraction=0.9,
)

# =============================================================================
# Eval Configuration
# =============================================================================

run = EvalRunConfig(
    max_concurrent=10,  # SGLang handles batching efficiently
    max_samples=20,
    max_turns=1,
    verbose=True,
)

output = EvalOutputConfig(
    experiment_name="reverse_text_sglang",
)

# =============================================================================
# Tasks
# =============================================================================

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
    {"text": "natural language processing"},
    {"text": "computer vision"},
    {"text": "reinforcement learning"},
    {"text": "generative adversarial network"},
    {"text": "recurrent neural network"},
    {"text": "convolutional neural network"},
    {"text": "long short term memory"},
    {"text": "word embedding"},
    {"text": "transfer learning"},
    {"text": "fine tuning"},
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


def score_fn(sample: AttemptResult, _context: object) -> Score:
    """Score the response by checking if reversal is correct."""
    import re

    # Get expected reversal from input
    input_data = sample.input
    expected = input_data["text"][::-1]

    # Extract response
    response = sample.response if hasattr(sample, "response") else ""
    if not response and sample.trajectory:
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

    # Compute similarity
    from difflib import SequenceMatcher

    similarity = SequenceMatcher(None, parsed, expected).ratio()

    return Score(
        metrics=(
            Metric("exact_match", 1.0 if exact_match else 0.0, weight=1.0),
            Metric("similarity", similarity, weight=0.0),
            Metric("has_tags", 1.0 if match else 0.0, weight=0.0),
        )
    )


scorer = FunctionScorer(score_fn)
