"""Category-aware data pipeline for REAP calibration.

Extends the simple data.py with:
- Dataset registry for multiple dataset types
- Category-aware splitting (e.g., code, math, reasoning)
- Configurable samples per category
- Better text field detection
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from torch import Tensor

logger = logging.getLogger(__name__)


# Registry of dataset processors
DATASET_REGISTRY: dict[str, type[DatasetProcessor]] = {}


def register_dataset(name: str):
    """Decorator to register a dataset processor."""

    def decorator(cls: type[DatasetProcessor]) -> type[DatasetProcessor]:
        DATASET_REGISTRY[name] = cls
        return cls

    return decorator


@dataclass
class ProcessedDataset:
    """Container for processed dataset by category."""

    categories: dict[str, list[dict[str, Tensor]]]
    total_samples: int


class DatasetProcessor(ABC):
    """Base class for dataset processing.

    Handles:
    - Loading raw dataset from HuggingFace
    - Extracting text fields
    - Tokenization
    - Category-aware splitting (optional)
    """

    def __init__(
        self,
        dataset: Any,
        tokenizer: Any,
        max_input_len: int = 2048,
        split: str = "train",
        split_by_category: bool = False,
        truncate: bool = True,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.split = split
        self.split_by_category = split_by_category
        self.truncate = truncate

    @abstractmethod
    def get_text(self, sample: dict) -> str | None:
        """Extract text from a dataset sample.

        Returns None if sample should be skipped.
        """
        pass

    @abstractmethod
    def get_category(self, sample: dict) -> str:
        """Get category for a sample. Used when split_by_category=True."""
        pass

    def process_sample(self, sample: dict) -> dict[str, Tensor] | None:
        """Tokenize a single sample.

        Returns None if sample is invalid or too short.
        """
        text = self.get_text(sample)
        if text is None or not text.strip():
            return None

        tokens = self.tokenizer(
            text,
            max_length=self.max_input_len,
            truncation=self.truncate,
            return_tensors="pt",
        )

        # Skip very short sequences
        if tokens["input_ids"].shape[1] < 32:
            return None

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
        }

    def get_processed_dataset(
        self,
        samples_per_category: int = 1024,
    ) -> dict[str, list[dict[str, Tensor]]]:
        """Process dataset into tokenized samples by category.

        Args:
            samples_per_category: Number of samples to collect per category

        Returns:
            Dict mapping category name to list of tokenized samples
        """
        if self.split_by_category:
            return self._process_by_category(samples_per_category)
        else:
            return self._process_all(samples_per_category)

    def _process_by_category(self, samples_per_category: int) -> dict[str, list[dict[str, Tensor]]]:
        """Process dataset splitting by category."""
        category_samples: dict[str, list[dict[str, Tensor]]] = {}
        category_counts: dict[str, int] = {}

        for sample in self.dataset:
            category = self.get_category(sample)

            # Skip if we have enough for this category
            if category_counts.get(category, 0) >= samples_per_category:
                continue

            processed = self.process_sample(sample)
            if processed is None:
                continue

            if category not in category_samples:
                category_samples[category] = []
                category_counts[category] = 0

            category_samples[category].append(processed)
            category_counts[category] += 1

            # Check if we're done
            if all(count >= samples_per_category for count in category_counts.values()):
                # But continue if we haven't seen all categories yet
                pass

        logger.info(f"Processed {len(category_samples)} categories")
        for cat, samples in category_samples.items():
            logger.info(f"  {cat}: {len(samples)} samples")

        return category_samples

    def _process_all(self, num_samples: int) -> dict[str, list[dict[str, Tensor]]]:
        """Process dataset without category splitting."""
        samples = []

        for sample in self.dataset:
            if len(samples) >= num_samples:
                break

            processed = self.process_sample(sample)
            if processed is not None:
                samples.append(processed)

        logger.info(f"Processed {len(samples)} samples (no category split)")
        return {"all": samples}


# =============================================================================
# Concrete Dataset Processors
# =============================================================================


@register_dataset("theblackcat102/evol-codealpaca-v1")
class EvolCodeAlpacaProcessor(DatasetProcessor):
    """Processor for Evol Code Alpaca dataset."""

    def get_text(self, sample: dict) -> str | None:
        # Format: instruction + input + output
        parts = []
        if "instruction" in sample and sample["instruction"]:
            parts.append(sample["instruction"])
        if "input" in sample and sample["input"]:
            parts.append(sample["input"])
        if "output" in sample and sample["output"]:
            parts.append(sample["output"])

        return "\n\n".join(parts) if parts else None

    def get_category(self, sample: dict) -> str:
        # Simple heuristic: categorize by content
        text = self.get_text(sample) or ""

        # Check for common programming languages
        if any(lang in text.lower() for lang in ["python", "def ", "import "]):
            return "python"
        elif any(lang in text.lower() for lang in ["javascript", "js", "function"]):
            return "javascript"
        elif any(lang in text.lower() for lang in ["sql", "select ", "from "]):
            return "sql"
        else:
            return "other_code"


@register_dataset("m-a-p/CodeFeedback-Filtered-Instruction")
class CodeFeedbackProcessor(DatasetProcessor):
    """Processor for CodeFeedback dataset."""

    def get_text(self, sample: dict) -> str | None:
        # Try different field names
        for field in ["instruction", "prompt", "input", "text"]:
            if field in sample and sample[field]:
                return str(sample[field])
        return None

    def get_category(self, sample: dict) -> str:
        text = self.get_text(sample) or ""

        # Categorize by complexity (simple heuristic based on length)
        if len(text) < 500:
            return "short"
        elif len(text) < 2000:
            return "medium"
        else:
            return "long"


@register_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K")
class MagicoderProcessor(DatasetProcessor):
    """Processor for Magicoder dataset."""

    def get_text(self, sample: dict) -> str | None:
        if "instruction" in sample and "response" in sample:
            return f"{sample['instruction']}\n\n{sample['response']}"
        return None

    def get_category(self, sample: dict) -> str:
        text = self.get_text(sample) or ""

        # Categorize by problem type
        if any(kw in text.lower() for kw in ["algorithm", "sort", "search", "tree", "graph"]):
            return "algorithms"
        elif any(kw in text.lower() for kw in ["api", "request", "http", "server", "client"]):
            return "web"
        elif any(kw in text.lower() for kw in ["database", "sql", "query", "table"]):
            return "database"
        else:
            return "general"


@register_dataset("allenai/c4")
class C4Processor(DatasetProcessor):
    """Processor for C4 dataset."""

    def get_text(self, sample: dict) -> str | None:
        return sample.get("text", None)

    def get_category(self, sample: dict) -> str:
        # C4 doesn't have natural categories, use length-based
        text = self.get_text(sample) or ""
        if len(text) < 1000:
            return "short"
        elif len(text) < 5000:
            return "medium"
        else:
            return "long"


@register_dataset("euclaise/WritingPrompts_curated")
class WritingPromptsProcessor(DatasetProcessor):
    """Processor for WritingPrompts dataset."""

    def get_text(self, sample: dict) -> str | None:
        # Combine prompt and story
        parts = []
        if "prompt" in sample:
            parts.append(sample["prompt"])
        if "story" in sample:
            parts.append(sample["story"])

        return "\n\n".join(parts) if parts else None

    def get_category(self, sample: dict) -> str:
        # Categorize by genre heuristics
        text = self.get_text(sample) or ""
        text_lower = text.lower()

        if any(kw in text_lower for kw in ["sci-fi", "space", "alien", "future", "robot"]):
            return "sci-fi"
        elif any(kw in text_lower for kw in ["fantasy", "dragon", "magic", "wizard", "kingdom"]):
            return "fantasy"
        elif any(kw in text_lower for kw in ["horror", "ghost", "zombie", "scary", "dark"]):
            return "horror"
        else:
            return "general"


@register_dataset("allenai/tulu-3-sft-personas-math")
class TuluMathProcessor(DatasetProcessor):
    """Processor for Tulu-3 math dataset."""

    def get_text(self, sample: dict) -> str | None:
        # Try different field combinations
        if "messages" in sample:
            # Chat format
            messages = sample["messages"]
            if isinstance(messages, list):
                return "\n".join([m.get("content", "") for m in messages if "content" in m])
        elif "instruction" in sample and "output" in sample:
            return f"{sample['instruction']}\n\n{sample['output']}"
        elif "text" in sample:
            return sample["text"]

        return None

    def get_category(self, sample: dict) -> str:
        text = self.get_text(sample) or ""
        text_lower = text.lower()

        # Categorize by math topic
        if any(kw in text_lower for kw in ["algebra", "equation", "solve for", "x ="]):
            return "algebra"
        elif any(kw in text_lower for kw in ["geometry", "triangle", "circle", "angle", "area"]):
            return "geometry"
        elif any(kw in text_lower for kw in ["calculus", "derivative", "integral", "limit"]):
            return "calculus"
        elif any(
            kw in text_lower
            for kw in ["probability", "statistics", "mean", "median", "distribution"]
        ):
            return "statistics"
        else:
            return "general_math"


# =============================================================================
# Factory and Loader
# =============================================================================


def get_dataset_processor(
    dataset_name: str,
    dataset: Any,
    tokenizer: Any,
    max_input_len: int = 2048,
    split: str = "train",
    split_by_category: bool = False,
    truncate: bool = True,
) -> DatasetProcessor:
    """Get appropriate processor for a dataset.

    Args:
        dataset_name: HuggingFace dataset identifier
        dataset: Loaded dataset object
        tokenizer: Tokenizer instance
        max_input_len: Maximum sequence length
        split: Dataset split
        split_by_category: Whether to split by category
        truncate: Whether to truncate long sequences

    Returns:
        DatasetProcessor instance
    """
    if dataset_name in DATASET_REGISTRY:
        processor_cls = DATASET_REGISTRY[dataset_name]
        return processor_cls(
            dataset=dataset,
            tokenizer=tokenizer,
            max_input_len=max_input_len,
            split=split,
            split_by_category=split_by_category,
            truncate=truncate,
        )
    else:
        # Fall back to generic processor
        logger.warning(f"No specific processor for {dataset_name}, using generic")
        return GenericProcessor(
            dataset=dataset,
            tokenizer=tokenizer,
            max_input_len=max_input_len,
            split=split,
            split_by_category=split_by_category,
            truncate=truncate,
        )


class GenericProcessor(DatasetProcessor):
    """Generic processor that tries to auto-detect fields."""

    TEXT_FIELDS = ["text", "content", "instruction", "prompt", "input", "question", "document"]

    def get_text(self, sample: dict) -> str | None:
        # Try common text fields
        for field in self.TEXT_FIELDS:
            if field in sample and sample[field]:
                value = sample[field]
                if isinstance(value, str):
                    return value
                elif isinstance(value, list):
                    return " ".join(str(v) for v in value)

        # Last resort: concatenate all string fields
        strings = []
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 50:  # Only substantial strings
                strings.append(value)

        return "\n".join(strings) if strings else None

    def get_category(self, sample: dict) -> str:
        return "all"


def load_calibration_data_advanced(
    dataset_name: str,
    tokenizer: Any,
    num_samples: int,
    max_seq_len: int,
    seed: int = 42,
    split_by_category: bool = False,
    samples_per_category: int | None = None,
) -> dict[str, list[dict[str, Tensor]]]:
    """Load calibration data with category-aware processing.

    Args:
        dataset_name: HuggingFace dataset identifier
        tokenizer: Tokenizer instance
        num_samples: Total samples (if not split_by_category) or per category
        max_seq_len: Maximum sequence length
        seed: Random seed
        split_by_category: Whether to split by category
        samples_per_category: Override samples per category (defaults to num_samples)

    Returns:
        Dict mapping category -> list of tokenized samples
    """
    from datasets import load_dataset

    logger.info(f"Loading dataset: {dataset_name}")

    # Load dataset
    try:
        if dataset_name == "allenai/c4":
            # Special handling for C4 (streaming dataset)
            file_url = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz"
            raw_ds = load_dataset(
                "json", data_files={"train": file_url}, split="train", streaming=False
            )
        else:
            raw_ds = load_dataset(dataset_name, split="train")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{dataset_name}': {e}")

    # Shuffle
    raw_ds = raw_ds.shuffle(seed=seed)

    # Get processor
    processor = get_dataset_processor(
        dataset_name=dataset_name,
        dataset=raw_ds,
        tokenizer=tokenizer,
        max_input_len=max_seq_len,
        split="train",
        split_by_category=split_by_category,
        truncate=True,
    )

    # Determine samples per category
    spc = samples_per_category if samples_per_category is not None else num_samples

    # Process
    return processor.get_processed_dataset(samples_per_category=spc)
