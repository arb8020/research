"""iGSM RL training config.

Experiments from Physics of Language Models Part 2.2:
- Can RL learn error-correction behavior where LoRA fails?

Paper setup (GPT2-12-12):
- 12 layers, 12 heads, 768 hidden dim
- LR: 0.002, weight decay: 0.05
- Batch size: 512, context: 768
- 100k steps for iGSM-med
- AdamW β=(0.9, 0.98), cosine decay to 0.01x

Usage:
    # Pretrain on clean data
    python pretrain_clean.py

    # RL from pretrained checkpoint
    python grpo_from_clean.py --checkpoint path/to/pretrain/step_100000.pt
"""

from __future__ import annotations

import re
from typing import Any, Literal

from rollouts.dtypes import Metric, Score
from rollouts.environments.no_tools import BasicEnvironment
from rollouts.synthetic import RETRY_TOKEN as _RETRY_TOKEN
from rollouts.synthetic import build_igsm_loader
from rollouts.synthetic.igsm import get_tokenizer

# ──────────────────────── iGSM Data Loading ──────────────────────────────────


def load_igsm_prompts(
    max_samples: int | None = None,
    difficulty: Literal["easy", "med", "hard"] = "med",
    seed: int = 42,
    mode: Literal["train", "val"] = "train",
) -> list[dict[str, Any]]:
    """Generate iGSM problems as chat prompts for RL.

    Args:
        max_samples: Number of problems to generate
        difficulty: Problem difficulty level
        seed: Random seed (different seeds = different problems)
        mode: "train" or "val" (val uses offset seed)

    Returns:
        List of prompt dicts with messages and ground truth answer
    """
    tokenizer = get_tokenizer()

    # Use a loader just to generate problems (we'll extract text, not train on tokens)
    loader = build_igsm_loader(
        difficulty=difficulty,
        seq_len=768,
        batch_size=1,
        device="cpu",
        seed=seed if mode == "train" else seed + 1000000,
        mode=mode,
    )

    prompts = []
    n_samples = max_samples or 1000

    for _ in range(n_samples):
        # Generate one problem
        input_ids, _ = loader.next()
        tokens = input_ids[0].tolist()

        # Parse the token sequence to extract problem, solution, answer
        # Format: [222] + prob + [223] + sol + [224] + ans + [50256]
        prob_start = loader.config.prob_start_token  # 222
        sol_start = loader.config.sol_start_token  # 223
        ans_start = loader.config.ans_start_token  # 224
        eos = loader.config.eos_token  # 50256

        # Find boundaries
        try:
            sol_idx = tokens.index(sol_start)
            ans_idx = tokens.index(ans_start)
        except ValueError:
            continue  # Skip malformed sequences

        # Extract segments
        prob_tokens = tokens[1:sol_idx]  # Skip prob_start token
        ans_tokens = tokens[ans_idx + 1 :]
        # Remove EOS and padding
        ans_tokens = [t for t in ans_tokens if t != eos]

        # Decode
        problem_text = tokenizer.decode(prob_tokens).strip()
        answer_text = tokenizer.decode(ans_tokens).strip()

        # Format as chat prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem_text},
        ]

        prompts.append({
            "messages": messages,
            "answer": answer_text,
            "difficulty": difficulty,
        })

    return prompts


SYSTEM_PROMPT = """\
Solve the following math problem step by step.
Show your reasoning clearly, then state your final numerical answer.

Format your answer as: The answer is [NUMBER]
"""

# Retry token used in paper (imported from synthetic module)
RETRY_TOKEN = _RETRY_TOKEN


# ──────────────────────── Score Function ─────────────────────────────────────


def extract_answer(text: str) -> str | None:
    """Extract answer from model response.

    Looks for patterns like:
    - "The answer is 42"
    - "= 42"
    - Final number in the response
    """
    if not text:
        return None

    # Try explicit "answer is X" pattern
    match = re.search(r"(?:answer|result)\s+(?:is|=)\s*(\d+)", text, re.IGNORECASE)
    if match:
        return match.group(1)

    # Try "= X" at end of line
    match = re.search(r"=\s*(\d+)\s*\.?\s*$", text, re.MULTILINE)
    if match:
        return match.group(1)

    # Fall back to last number in text
    numbers = re.findall(r"\b(\d+)\b", text)
    if numbers:
        return numbers[-1]

    return None


def normalize_answer(answer: str) -> int | None:
    """Normalize answer string to integer."""
    if not answer:
        return None
    try:
        # Remove commas, whitespace
        clean = answer.replace(",", "").strip()
        return int(clean)
    except ValueError:
        return None


def igsm_score_fn(sample: Any) -> Score:
    """Binary score function for iGSM.

    Returns 1.0 if correct, 0.0 otherwise.
    No partial credit, no shaping.
    """
    ground_truth = sample.metadata.get("answer")
    if ground_truth is None:
        return Score(metrics=(Metric("correct", 0.0, weight=1.0),))

    response = sample.response if hasattr(sample, "response") else ""
    predicted = extract_answer(response)

    if predicted is None:
        return Score(
            metrics=(
                Metric("correct", 0.0, weight=1.0),
                Metric("parse_failed", 1.0, weight=0.0),
            )
        )

    pred_val = normalize_answer(predicted)
    true_val = normalize_answer(ground_truth)

    if pred_val is None or true_val is None:
        return Score(
            metrics=(
                Metric("correct", 0.0, weight=1.0),
                Metric("parse_failed", 1.0, weight=0.0),
            )
        )

    is_correct = pred_val == true_val
    return Score(metrics=(Metric("correct", 1.0 if is_correct else 0.0, weight=1.0),))


# ──────────────────────── Model Config (GPT2-12-12) ──────────────────────────


# Paper's GPT2-12-12: 12 layers, 12 heads, 768 hidden dim
MODEL_CONFIG = {
    "dim": 768,
    "n_layers": 12,
    "n_heads": 12,
    "vocab_size": 50257,  # GPT2 tokenizer
    "max_seq_len": 768,
}

# Paper's training hyperparameters
PRETRAIN_CONFIG = {
    "lr": 0.002,
    "weight_decay": 0.05,
    "batch_size": 512,
    "steps": 100_000,
    "warmup_steps": 1000,
    "adam_betas": (0.9, 0.98),
    # Cosine decay to 0.01x
}

# RL hyperparameters (tuned for math reasoning)
RL_CONFIG = {
    "lr": 1e-6,  # Much lower than pretrain
    "n_samples_per_prompt": 8,
    "temperature": 0.8,
    "max_seq_len": 1024,  # Allow longer responses for reasoning
}


# ──────────────────────── Training Entry Points ──────────────────────────────


def train_grpo(
    checkpoint_path: str | Path | None = None,
    max_samples: int | None = None,
    difficulty: Literal["easy", "med", "hard"] = "med",
) -> dict[str, Any]:
    """Run GRPO training on iGSM.

    Args:
        checkpoint_path: Path to pretrained checkpoint (optional)
        max_samples: Limit dataset size
        difficulty: Problem difficulty

    Returns:
        Dict with metrics_history
    """
    from rollouts.training.grpo import (
        GRPOConfig,
        GRPOOutputConfig,
        ModelConfig,
        RolloutConfig,
        TrainerConfig,
        grpo_train,
    )

    config = GRPOConfig(
        model=ModelConfig(
            name="gpt2" if checkpoint_path is None else str(checkpoint_path),
        ),
        output=GRPOOutputConfig(experiment_name=f"igsm_{difficulty}_grpo"),
        trainer=TrainerConfig(lr=RL_CONFIG["lr"]),
        rollout=RolloutConfig(
            n_samples_per_prompt=RL_CONFIG["n_samples_per_prompt"],
            temperature=RL_CONFIG["temperature"],
            max_seq_len=RL_CONFIG["max_seq_len"],
        ),
    )

    prompts = load_igsm_prompts(max_samples=max_samples, difficulty=difficulty)

    return grpo_train(
        config=config,
        prompts=prompts,
        score_fn=igsm_score_fn,
        environment_cls=BasicEnvironment,
    )


if __name__ == "__main__":
    # Quick test of data loading and scoring
    print("Testing iGSM data loading...")

    try:
        prompts = load_igsm_prompts(max_samples=3, difficulty="easy")
        print(f"Loaded {len(prompts)} prompts")

        for i, p in enumerate(prompts):
            print(f"\n--- Problem {i + 1} ---")
            print(f"Question: {p['messages'][1]['content'][:200]}...")
            print(f"Answer: {p['answer']}")
    except Exception as e:
        print(f"Error loading iGSM (is iGSM cloned to /tmp/iGSM?): {e}")
        print("\nTo set up iGSM:")
        print("  git clone https://github.com/facebookresearch/iGSM.git /tmp/iGSM")
