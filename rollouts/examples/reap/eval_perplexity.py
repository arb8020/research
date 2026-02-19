"""Simple perplexity evaluation for pruned models.

Compares perplexity of original vs pruned model on held-out data.
This is a quick sanity check - for full benchmarks, use lm-eval-harness.

Usage:
    python examples/reap/eval_perplexity.py --recipe results/reap/.../pruning_recipe.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .export import apply_pruning_recipe, load_pruning_recipe

logger = logging.getLogger(__name__)


def compute_perplexity(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_length: int = 2048,
    stride: int = 512,
) -> float:
    """Compute perplexity on a list of texts using sliding window."""
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in tqdm(texts, desc="Computing perplexity"):
            encodings = tokenizer(text, return_tensors="pt", truncation=False)
            seq_len = encodings.input_ids.size(1)

            prev_end_loc = 0
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - prev_end_loc

                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100  # Mask tokens we've already seen

                outputs = model(input_ids, labels=target_ids)
                loss = outputs.loss

                total_loss += loss.item() * trg_len
                total_tokens += trg_len

                prev_end_loc = end_loc
                if end_loc >= seq_len:
                    break

    return math.exp(total_loss / total_tokens)


def load_eval_data(dataset_name: str = "wikitext", num_samples: int = 100) -> list[str]:
    """Load evaluation data."""
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        # Filter empty strings and concatenate into chunks
        texts = [t for t in dataset["text"] if t.strip()]
        # Take num_samples chunks
        return texts[:num_samples]
    else:
        dataset = load_dataset(dataset_name, split="test")
        return [sample["text"] for sample in dataset.select(range(num_samples))]


def run_perplexity_eval(
    recipe_path: Path,
    num_samples: int = 100,
    eval_dataset: str = "wikitext",
) -> dict:
    """Run perplexity comparison between original and pruned model."""
    # Load recipe
    recipe = load_pruning_recipe(recipe_path)
    base_model = recipe["base_model"]
    experts_to_keep = recipe["experts_to_keep"]

    logger.info(f"Base model: {base_model}")
    logger.info(f"Compression: {recipe.get('compression_ratio', 'unknown')}")

    # Load eval data
    logger.info(f"Loading eval data: {eval_dataset} ({num_samples} samples)")
    texts = load_eval_data(eval_dataset, num_samples)
    logger.info(f"Loaded {len(texts)} text samples")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Evaluate original model
    logger.info("Loading original model...")
    original_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info("Computing original model perplexity...")
    original_ppl = compute_perplexity(original_model, tokenizer, texts)
    logger.info(f"Original perplexity: {original_ppl:.2f}")

    # Apply pruning and evaluate
    logger.info("Applying pruning recipe...")
    apply_pruning_recipe(original_model, experts_to_keep)

    logger.info("Computing pruned model perplexity...")
    pruned_ppl = compute_perplexity(original_model, tokenizer, texts)
    logger.info(f"Pruned perplexity: {pruned_ppl:.2f}")

    # Compute degradation
    ppl_increase = (pruned_ppl - original_ppl) / original_ppl * 100

    results = {
        "base_model": base_model,
        "compression_ratio": recipe.get("compression_ratio"),
        "eval_dataset": eval_dataset,
        "num_samples": num_samples,
        "original_perplexity": original_ppl,
        "pruned_perplexity": pruned_ppl,
        "perplexity_increase_pct": ppl_increase,
    }

    logger.info("=" * 50)
    logger.info("Results:")
    logger.info(f"  Original PPL: {original_ppl:.2f}")
    logger.info(f"  Pruned PPL:   {pruned_ppl:.2f}")
    logger.info(f"  Increase:     {ppl_increase:+.1f}%")
    logger.info("=" * 50)

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Evaluate pruned model perplexity")
    parser.add_argument(
        "--recipe",
        type=Path,
        required=True,
        help="Path to pruning_recipe.json",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of eval samples",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="Evaluation dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    results = run_perplexity_eval(
        recipe_path=args.recipe,
        num_samples=args.num_samples,
        eval_dataset=args.dataset,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
