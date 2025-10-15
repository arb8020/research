#!/usr/bin/env python3
"""Load evaluation datasets (GSM8K, ARC, MMLU) as Trajectories."""

import json
import logging
from pathlib import Path
from typing import List, Optional
from datasets import load_dataset

from trajectory import Trajectory, batch_create_trajectories

logger = logging.getLogger(__name__)


def load_gsm8k(split: str = "test", num_samples: Optional[int] = None) -> List[Trajectory]:
    """Load GSM8K dataset as trajectories.

    Args:
        split: Dataset split ("test" or "train")
        num_samples: Limit to N samples (None = all)

    Returns:
        List of Trajectories with metadata:
            - dataset: "gsm8k"
            - ground_truth: The numeric answer
            - difficulty: "gsm8k"  # placeholder, could compute from solve rates
    """
    logger.info(f"Loading GSM8K {split} split...")
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    prompts = []
    ground_truths = []

    for item in dataset:
        # GSM8K format: question + answer with #### separator
        question = item['question']
        answer = item['answer'].split('####')[-1].strip()

        prompts.append(question)
        ground_truths.append(answer)

    logger.info(f"Loaded {len(prompts)} GSM8K examples")

    return batch_create_trajectories(
        prompts=prompts,
        ground_truths=ground_truths,
        dataset="gsm8k"
    )


def load_arc(subset: str = "easy", split: str = "test", num_samples: Optional[int] = None) -> List[Trajectory]:
    """Load ARC dataset as trajectories.

    Args:
        subset: "easy" or "challenge"
        split: Dataset split ("test", "validation", or "train")
        num_samples: Limit to N samples (None = all)

    Returns:
        List of Trajectories with metadata:
            - dataset: "arc-easy" or "arc-challenge"
            - ground_truth: The correct answer letter (A/B/C/D)
            - choices: List of answer choices
    """
    dataset_name = f"arc-{subset}"
    logger.info(f"Loading ARC-{subset} {split} split...")

    dataset = load_dataset("allenai/ai2_arc", f"ARC-{subset.capitalize()}", split=split)

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    trajectories = []

    for item in dataset:
        question = item['question']
        choices = item['choices']['text']
        labels = item['choices']['label']
        answer_key = item['answerKey']

        # Format as multiple choice
        choice_text = "\n".join([f"{label}. {text}" for label, text in zip(labels, choices)])
        prompt = f"{question}\n\n{choice_text}\n\nAnswer:"

        traj = Trajectory(
            prompt=prompt,
            metadata={
                'dataset': dataset_name,
                'ground_truth': answer_key,
                'choices': choices,
                'labels': labels
            }
        )
        trajectories.append(traj)

    logger.info(f"Loaded {len(trajectories)} ARC-{subset} examples")
    return trajectories


def load_mmlu(
    subject: Optional[str] = None,
    split: str = "test",
    num_samples: Optional[int] = None
) -> List[Trajectory]:
    """Load MMLU dataset as trajectories.

    Args:
        subject: Specific subject (None = all subjects)
        split: Dataset split ("test", "validation", "dev", or "auxiliary_train")
        num_samples: Limit to N samples per subject (None = all)

    Returns:
        List of Trajectories with metadata:
            - dataset: "mmlu"
            - subject: The MMLU subject
            - ground_truth: The correct answer index (0-3)
    """
    logger.info(f"Loading MMLU {split} split..." + (f" (subject: {subject})" if subject else ""))

    if subject:
        dataset = load_dataset("cais/mmlu", subject, split=split)
        subjects = [subject]
    else:
        dataset = load_dataset("cais/mmlu", "all", split=split)
        subjects = dataset['subject']

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    trajectories = []

    for i, item in enumerate(dataset):
        question = item['question']
        choices = item['choices']
        answer = item['answer']  # 0-3
        subj = subjects[i] if isinstance(subjects, list) else item.get('subject', 'unknown')

        # Format as multiple choice
        choice_labels = ['A', 'B', 'C', 'D']
        choice_text = "\n".join([f"{label}. {text}" for label, text in zip(choice_labels, choices)])
        prompt = f"{question}\n\n{choice_text}\n\nAnswer:"

        traj = Trajectory(
            prompt=prompt,
            metadata={
                'dataset': 'mmlu',
                'subject': subj,
                'ground_truth': choice_labels[answer],
                'choices': choices
            }
        )
        trajectories.append(traj)

    logger.info(f"Loaded {len(trajectories)} MMLU examples")
    return trajectories


def save_trajectories(trajectories: List[Trajectory], output_path: Path):
    """Save trajectories to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for traj in trajectories:
            f.write(json.dumps(traj.to_dict()) + '\n')

    logger.info(f"Saved {len(trajectories)} trajectories to {output_path}")


def load_trajectories(input_path: Path) -> List[Trajectory]:
    """Load trajectories from JSONL file."""
    trajectories = []

    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            trajectories.append(Trajectory.from_dict(data))

    logger.info(f"Loaded {len(trajectories)} trajectories from {input_path}")
    return trajectories


def main():
    """Demo: Load and save eval datasets."""
    import sys

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    output_dir = Path("data/eval_datasets")

    # Load a small sample of each dataset
    logger.info("Loading eval datasets...")

    gsm8k = load_gsm8k(split="test", num_samples=100)
    arc_easy = load_arc(subset="easy", split="test", num_samples=100)
    arc_challenge = load_arc(subset="challenge", split="test", num_samples=100)
    mmlu = load_mmlu(split="test", num_samples=100)

    # Save them
    save_trajectories(gsm8k, output_dir / "gsm8k_test_100.jsonl")
    save_trajectories(arc_easy, output_dir / "arc_easy_test_100.jsonl")
    save_trajectories(arc_challenge, output_dir / "arc_challenge_test_100.jsonl")
    save_trajectories(mmlu, output_dir / "mmlu_test_100.jsonl")

    logger.info("Done! Saved eval datasets to data/eval_datasets/")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
