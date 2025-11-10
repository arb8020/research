"""Dataset loaders for training.

Functions to load HuggingFace datasets and convert to Sample format.
Supports common datasets for SFT and RL training.
"""

import logging
from typing import Any, Dict, List, Optional

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

from rollouts.training.types import Sample


# Import tokenization functions lazily to avoid torch dependency at import time
def _get_tokenize_conversation():
    from rollouts.training.datasets.sft import tokenize_conversation
    return tokenize_conversation

def _get_compute_loss_mask():
    from rollouts.training.datasets.sft import compute_loss_mask
    return compute_loss_mask

logger = logging.getLogger(__name__)


def load_sft_dataset(
    dataset_name: str,
    split: str = "train",
    subset: Optional[str] = None,
    tokenizer: Optional[Any] = None,
    max_samples: Optional[int] = None,
    max_length: int = 2048,
) -> List[Sample]:
    """Load HuggingFace dataset and convert to SFT samples.

    Loads a chat/instruction dataset from HuggingFace and converts conversations
    to Sample objects ready for SFT training.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "HuggingFaceTB/smoltalk")
        split: Dataset split to load (default: "train")
        subset: Optional dataset subset (e.g., "ARC-Easy" for ai2_arc)
        tokenizer: Tokenizer for encoding conversations (if None, tokens will be empty)
        max_samples: Optional limit on number of samples to load
        max_length: Maximum sequence length for tokenization (default: 2048)

    Returns:
        List of Sample objects with tokenized conversations

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> samples = load_sft_dataset(
        ...     "HuggingFaceTB/smoltalk",
        ...     tokenizer=tokenizer,
        ...     max_samples=1000,
        ... )
        >>> len(samples)
        1000
    """
    if not HAS_DATASETS:
        raise ImportError("datasets library not installed. Install with: pip install datasets")

    logger.info(f"Loading dataset: {dataset_name} (split={split}, subset={subset})")

    # Load dataset from HuggingFace (with optional subset)
    if subset:
        dataset = load_dataset(dataset_name, subset, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    logger.info(f"  Loaded {len(dataset)} examples")

    # Convert to samples
    samples = []
    skipped_too_long = 0

    for idx, example in enumerate(dataset):
        # Extract conversation (handle different formats)
        conversation = _extract_conversation(example)

        if conversation is None:
            logger.warning(f"  Skipping example {idx}: could not extract conversation")
            continue

        # Create sample
        if tokenizer is not None:
            # Pre-filter conversations that exceed max_length (SLIME pattern)
            # This prevents truncation-related bugs (empty spans, invalid loss masks)
            estimated_tokens = _estimate_conversation_length(conversation, tokenizer)
            if estimated_tokens > max_length:
                skipped_too_long += 1
                continue

            # Tokenize conversation (lazy import to avoid torch at module load)
            tokenize_conversation = _get_tokenize_conversation()
            compute_loss_mask = _get_compute_loss_mask()

            tokens, user_spans = tokenize_conversation(conversation, tokenizer, max_length)
            loss_mask = compute_loss_mask(tokens, user_spans)

            sample = Sample(
                prompt=conversation,
                response="",  # Full response is in conversation
                tokens=tokens,
                loss_mask=loss_mask,
                reward=0.0,
                metadata={"dataset": dataset_name, "index": idx},
            )
        else:
            # No tokenization - just store conversation
            sample = Sample(
                prompt=conversation,
                response="",
                tokens=[],
                loss_mask=[],
                reward=0.0,
                metadata={"dataset": dataset_name, "index": idx},
            )

        samples.append(sample)

    # Log filtering statistics (SLIME pattern: be transparent about data filtering)
    logger.info(f"  Converted to {len(samples)} samples")
    if skipped_too_long > 0:
        logger.info(f"  Skipped {skipped_too_long} samples (exceeded max_length={max_length})")

    return samples


def _estimate_conversation_length(
    conversation: List[Dict[str, str]],
    tokenizer: Any,
) -> int:
    """Estimate token length of conversation without full tokenization.

    Fast approximation - applies chat template and encodes without special tokens.
    Used for pre-filtering long conversations (SLIME pattern).

    Args:
        conversation: List of {role, content} message dicts
        tokenizer: HuggingFace tokenizer

    Returns:
        Estimated token count
    """
    # Apply chat template to get formatted text
    full_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False,
    )
    # Encode without special tokens for length estimation
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    return len(tokens)


def _extract_conversation(example: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """Extract conversation from dataset example.

    Handles different dataset formats:
    - messages: List of {role, content} dicts (standard format)
    - prompt/completion: Single turn instruction format
    - text: Raw text (try to parse)

    Args:
        example: Dataset example dict

    Returns:
        List of {role, content} dicts or None if format not recognized
    """
    # Format 1: messages field (standard chat format)
    if "messages" in example:
        return example["messages"]

    # Format 2: prompt/completion (instruction format)
    if "prompt" in example and "completion" in example:
        return [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]

    # Format 3: instruction/output (common instruction format)
    if "instruction" in example and "output" in example:
        return [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]

    # Format 4: question/answer
    if "question" in example and "answer" in example:
        return [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]

    # Format 5: conversations field (some datasets use this)
    if "conversations" in example:
        return example["conversations"]

    logger.warning(f"Unknown dataset format, available keys: {list(example.keys())}")
    return None


def load_rl_prompts(
    dataset_name: str,
    split: str = "train",
    max_prompts: Optional[int] = None,
) -> List[str]:
    """Load prompts for RL training from HuggingFace dataset.

    Loads a dataset and extracts prompts (questions/instructions) for RL rollout generation.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "gsm8k", "openai/gsm8k")
        split: Dataset split to load (default: "train")
        max_prompts: Optional limit on number of prompts

    Returns:
        List of prompt strings

    Example:
        >>> prompts = load_rl_prompts("openai/gsm8k", split="train", max_prompts=100)
        >>> len(prompts)
        100
        >>> prompts[0]
        'Janet's ducks lay 16 eggs per day...'
    """
    if not HAS_DATASETS:
        raise ImportError("datasets library not installed. Install with: pip install datasets")

    logger.info(f"Loading RL prompts: {dataset_name} (split={split})")

    # Load dataset
    dataset = load_dataset(dataset_name, split=split)

    if max_prompts is not None:
        dataset = dataset.select(range(min(max_prompts, len(dataset))))

    logger.info(f"  Loaded {len(dataset)} examples")

    # Extract prompts
    prompts = []
    for idx, example in enumerate(dataset):
        prompt = _extract_prompt(example)

        if prompt is None:
            logger.warning(f"  Skipping example {idx}: could not extract prompt")
            continue

        prompts.append(prompt)

    logger.info(f"  Extracted {len(prompts)} prompts")
    return prompts


def _extract_prompt(example: Dict[str, Any]) -> Optional[str]:
    """Extract prompt from dataset example.

    Handles different prompt field names.

    Args:
        example: Dataset example dict

    Returns:
        Prompt string or None if not found
    """
    # Try common prompt field names
    for field in ["prompt", "question", "instruction", "input", "text"]:
        if field in example:
            return example[field]

    # GSM8K specific
    if "question" in example:
        return example["question"]

    logger.warning(f"Could not find prompt in example, keys: {list(example.keys())}")
    return None


def load_dataset_with_answers(
    dataset_name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Load dataset with prompts and ground truth answers (for RL reward computation).

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split
        max_samples: Optional limit

    Returns:
        List of dicts with {"prompt": str, "answer": str}

    Example:
        >>> data = load_dataset_with_answers("openai/gsm8k", max_samples=10)
        >>> data[0]["prompt"]
        'Janet's ducks lay 16 eggs per day...'
        >>> data[0]["answer"]
        '18'
    """
    if not HAS_DATASETS:
        raise ImportError("datasets library not installed. Install with: pip install datasets")

    logger.info(f"Loading dataset with answers: {dataset_name} (split={split})")

    dataset = load_dataset(dataset_name, split=split)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    logger.info(f"  Loaded {len(dataset)} examples")

    # Extract prompt + answer pairs
    data = []
    for idx, example in enumerate(dataset):
        prompt = _extract_prompt(example)
        answer = _extract_answer(example)

        if prompt is None or answer is None:
            logger.warning(f"  Skipping example {idx}: missing prompt or answer")
            continue

        data.append({"prompt": prompt, "answer": answer})

    logger.info(f"  Extracted {len(data)} prompt/answer pairs")
    return data


def _extract_answer(example: Dict[str, Any]) -> Optional[str]:
    """Extract ground truth answer from dataset example.

    Args:
        example: Dataset example dict

    Returns:
        Answer string or None if not found
    """
    # Try common answer field names
    for field in ["answer", "completion", "output", "response", "target"]:
        if field in example:
            return example[field]

    logger.warning(f"Could not find answer in example, keys: {list(example.keys())}")
    return None
