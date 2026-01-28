"""Integration test: logprob parsing handles the real-world mix of
numeric scores, non-numeric tokens, and low-probability edge cases.

This is the scoring primitive that every evaluation depends on.
"""

import math

from examples.inoculation.judge import parse_score_from_logprobs


def test_parse_score_realistic_logprobs() -> None:
    """Realistic GPT-4o judge output: mix of numeric scores, REFUSAL, CODE tokens."""
    # High confidence on a score — should return weighted average
    logprobs = {
        "75": math.log(0.7),
        "74": math.log(0.1),
        "76": math.log(0.1),
        "REFUSAL": math.log(0.05),
        "CODE": math.log(0.05),
    }
    score = parse_score_from_logprobs(logprobs)
    assert score is not None
    assert 74.0 < score < 76.0

    # Judge mostly says REFUSAL — should return None (below min_prob)
    logprobs_refusal = {
        "REFUSAL": math.log(0.8),
        " 50": math.log(0.15),  # space-prefixed token (common in GPT)
        "CODE": math.log(0.05),
    }
    score_refusal = parse_score_from_logprobs(logprobs_refusal, min_prob=0.25)
    assert score_refusal is None

    # Empty logprobs — should return None
    assert parse_score_from_logprobs({}) is None
