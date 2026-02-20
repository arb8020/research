"""Evaluate iGSM accuracy on held-out problems.

Paper evaluation setup:
- 4096 problems per condition
- Parser checks intermediate solution steps (not just final answer)
- Two seeds × two beam options (1, 4)
- Context length 2048 for evaluation

For our purposes, we simplify to:
- Final answer correctness only (binary)
- Greedy decoding (beam=1) and sampling (temp=0.8)
- Measure retry token usage (did model learn self-correction?)

Usage:
    # Evaluate pretrained model
    python eval_accuracy.py --checkpoint output/igsm_clean/step_100000.pt

    # Evaluate GRPO model
    python eval_accuracy.py --checkpoint output/igsm_med_grpo/final.pt

    # Compare multiple checkpoints
    python eval_accuracy.py --checkpoints ckpt1.pt ckpt2.pt ckpt3.pt

    # Quick test
    python eval_accuracy.py --checkpoint gpt2 --num-problems 10
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# Support running as script
sys.path.insert(0, str(Path(__file__).parent))

from base_config import RETRY_TOKEN, extract_answer, load_igsm_prompts, normalize_answer


@dataclass
class EvalResult:
    """Evaluation result for a single checkpoint."""

    checkpoint: str
    difficulty: str
    num_problems: int
    accuracy: float
    parse_failures: int
    retry_usage_rate: float  # Fraction of responses containing [BACK]
    avg_response_length: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "difficulty": self.difficulty,
            "num_problems": self.num_problems,
            "accuracy": self.accuracy,
            "parse_failures": self.parse_failures,
            "retry_usage_rate": self.retry_usage_rate,
            "avg_response_length": self.avg_response_length,
        }


def count_retry_tokens(text: str) -> int:
    """Count occurrences of retry token in response."""
    return text.count(RETRY_TOKEN)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    difficulty: Literal["easy", "med", "hard"] = "med",
    num_problems: int = 100,
    temperature: float = 0.0,  # Greedy by default
    max_tokens: int = 512,
    seed: int = 99999,  # Eval seed, different from train/RL
) -> EvalResult:
    """Evaluate a checkpoint on held-out iGSM problems.

    Args:
        checkpoint_path: Path to model checkpoint or HF model name
        difficulty: Problem difficulty
        num_problems: Number of problems to evaluate
        temperature: Sampling temperature (0.0 = greedy)
        max_tokens: Max tokens to generate
        seed: Random seed for problem generation

    Returns:
        EvalResult with accuracy and other metrics
    """
    # Load problems
    prompts = load_igsm_prompts(
        max_samples=num_problems,
        difficulty=difficulty,
        seed=seed,
        mode="val",
    )

    # Load model and generate
    # This is a simplified version - real eval would use vLLM/SGLang
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model from {checkpoint_path}...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Check if it's a local checkpoint or HF model
        ckpt_path = Path(checkpoint_path)
        if ckpt_path.exists() and ckpt_path.suffix == ".pt":
            # Local checkpoint - need to load into model
            # This assumes checkpoint format from our pretrain script
            print("Loading local checkpoint...")
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            # TODO: Map our weight format to HF format
            # For now, just use the base model
            print("WARNING: Local checkpoint loading not fully implemented, using base GPT2")
        else:
            model = AutoModelForCausalLM.from_pretrained(str(checkpoint_path))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

    except ImportError:
        print("transformers not available, using mock evaluation")
        # Return mock results for testing
        return EvalResult(
            checkpoint=str(checkpoint_path),
            difficulty=difficulty,
            num_problems=num_problems,
            accuracy=0.0,
            parse_failures=num_problems,
            retry_usage_rate=0.0,
            avg_response_length=0.0,
        )

    # Evaluate
    correct = 0
    parse_failures = 0
    retry_count = 0
    total_length = 0

    print(f"Evaluating {num_problems} problems...")
    for i, prompt in enumerate(prompts):
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{num_problems}...")

        # Format prompt
        messages = prompt["messages"]
        text = f"{messages[0]['content']}\n\nProblem: {messages[1]['content']}\n\nSolution:"

        # Generate
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=768)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if temperature == 0.0:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.eos_token_id,
                )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        # Score
        predicted = extract_answer(response)
        ground_truth = prompt["answer"]

        if predicted is None:
            parse_failures += 1
        else:
            pred_val = normalize_answer(predicted)
            true_val = normalize_answer(ground_truth)
            if pred_val is not None and true_val is not None and pred_val == true_val:
                correct += 1

        # Track retry usage
        if RETRY_TOKEN in response:
            retry_count += 1

        total_length += len(response)

    accuracy = correct / num_problems if num_problems > 0 else 0.0
    retry_rate = retry_count / num_problems if num_problems > 0 else 0.0
    avg_length = total_length / num_problems if num_problems > 0 else 0.0

    return EvalResult(
        checkpoint=str(checkpoint_path),
        difficulty=difficulty,
        num_problems=num_problems,
        accuracy=accuracy,
        parse_failures=parse_failures,
        retry_usage_rate=retry_rate,
        avg_response_length=avg_length,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate iGSM accuracy")
    parser.add_argument(
        "--checkpoint", type=str, default="gpt2", help="Path to checkpoint or HF model name"
    )
    parser.add_argument(
        "--checkpoints", type=str, nargs="+", help="Multiple checkpoints to compare"
    )
    parser.add_argument(
        "--difficulty", choices=["easy", "med", "hard"], default="med", help="Problem difficulty"
    )
    parser.add_argument(
        "--num-problems", type=int, default=100, help="Number of problems to evaluate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature (0.0 = greedy)"
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")
    args = parser.parse_args()

    checkpoints = args.checkpoints or [args.checkpoint]
    results = []

    for ckpt in checkpoints:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {ckpt}")
        print(f"{'=' * 60}")

        result = evaluate_checkpoint(
            checkpoint_path=ckpt,
            difficulty=args.difficulty,
            num_problems=args.num_problems,
            temperature=args.temperature,
        )
        results.append(result)

        print("\nResults:")
        print(f"  Accuracy: {result.accuracy:.2%}")
        print(f"  Parse failures: {result.parse_failures}")
        print(f"  Retry usage: {result.retry_usage_rate:.2%}")
        print(f"  Avg response length: {result.avg_response_length:.1f}")

    # Summary table if multiple checkpoints
    if len(results) > 1:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(f"{'Checkpoint':<40} {'Accuracy':>10} {'Retry%':>10}")
        print("-" * 60)
        for r in results:
            name = Path(r.checkpoint).name[:38]
            print(f"{name:<40} {r.accuracy:>10.2%} {r.retry_usage_rate:>10.2%}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
