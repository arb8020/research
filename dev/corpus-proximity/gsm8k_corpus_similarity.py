#!/usr/bin/env python3
"""Measure distance from GSM8K eval questions/answers to training corpus.

Tests if eval questions are OOD (out-of-distribution).
"""

import asyncio
import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from chunking import chunk_text
from config import Config
from corpus import (
    ARC_CHALLENGE_TRAIN,
    ARC_EASY_TRAIN,
    GSM8K_TRAIN,
    MMLU_AUX_TRAIN,
    NANOCHAT_PRETRAIN,
    SMOLTALK,
    sample_corpus,
)
from datasets import load_dataset
from rollout import Endpoint, GSM8KSample, generate
from search import TrainingCorpus, cosine_distance, euclidean_distance, manhattan_distance
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """A single similarity measurement result."""
    sample_id: str
    variant: str  # "question" | "answer" | "question_answer"
    corpus_stage: str  # "pretrain" | "midtrain" | "sft"
    distance: float
    nearest_chunk_text: str
    rank: int  # 1 to k (1=closest)


def load_corpus_chunks(config: Config) -> dict[str, list[str]]:
    """Load and chunk corpus from each training stage.

    Following Casey Muratori: Explicit about what corpora go in each stage.

    Args:
        config: Config with similarity settings

    Returns:
        Dict mapping stage -> list of text chunks
        {"pretrain": [...], "midtrain": [...], "sft": [...]}
    """
    logger.info("=" * 80)
    logger.info("Loading corpus chunks")
    logger.info("=" * 80)

    chunks = {}
    sim_config = config.similarity

    # Pretrain stage: NANOCHAT_PRETRAIN
    logger.info(f"\nLoading pretrain corpus ({sim_config.corpus_sizes['pretrain']} samples)...")
    pretrain_texts = sample_corpus(NANOCHAT_PRETRAIN, n=sim_config.corpus_sizes['pretrain'])
    pretrain_chunks = []
    for text in pretrain_texts:
        pretrain_chunks.extend(
            chunk_text(text, sim_config.chunking_strategy, sim_config.chunk_size)
        )
    chunks["pretrain"] = pretrain_chunks
    logger.info(f"  Pretrain: {len(pretrain_chunks)} chunks")

    # Midtrain stage: SMOLTALK + MMLU_AUX_TRAIN + GSM8K_TRAIN
    logger.info(f"\nLoading midtrain corpus ({sim_config.corpus_sizes['midtrain']} samples)...")
    midtrain_chunks = []

    # Divide samples across 3 corpora
    samples_per_corpus = sim_config.corpus_sizes['midtrain'] // 3

    logger.info(f"  SMOLTALK: {samples_per_corpus} samples...")
    smoltalk_texts = sample_corpus(SMOLTALK, n=samples_per_corpus)
    for text in smoltalk_texts:
        midtrain_chunks.extend(
            chunk_text(text, sim_config.chunking_strategy, sim_config.chunk_size)
        )

    logger.info(f"  MMLU_AUX_TRAIN: {samples_per_corpus} samples...")
    mmlu_texts = sample_corpus(MMLU_AUX_TRAIN, n=samples_per_corpus)
    for text in mmlu_texts:
        midtrain_chunks.extend(
            chunk_text(text, sim_config.chunking_strategy, sim_config.chunk_size)
        )

    logger.info(f"  GSM8K_TRAIN: {samples_per_corpus} samples...")
    gsm8k_train_texts = sample_corpus(GSM8K_TRAIN, n=samples_per_corpus)
    for text in gsm8k_train_texts:
        midtrain_chunks.extend(
            chunk_text(text, sim_config.chunking_strategy, sim_config.chunk_size)
        )

    chunks["midtrain"] = midtrain_chunks
    logger.info(f"  Midtrain: {len(midtrain_chunks)} chunks")

    # SFT stage: ARC_EASY_TRAIN + ARC_CHALLENGE_TRAIN
    logger.info(f"\nLoading sft corpus ({sim_config.corpus_sizes['sft']} samples)...")
    sft_chunks = []

    # Divide samples across 2 corpora
    samples_per_corpus = sim_config.corpus_sizes['sft'] // 2

    logger.info(f"  ARC_EASY_TRAIN: {samples_per_corpus} samples...")
    arc_easy_texts = sample_corpus(ARC_EASY_TRAIN, n=samples_per_corpus)
    for text in arc_easy_texts:
        sft_chunks.extend(
            chunk_text(text, sim_config.chunking_strategy, sim_config.chunk_size)
        )

    logger.info(f"  ARC_CHALLENGE_TRAIN: {samples_per_corpus} samples...")
    arc_challenge_texts = sample_corpus(ARC_CHALLENGE_TRAIN, n=samples_per_corpus)
    for text in arc_challenge_texts:
        sft_chunks.extend(
            chunk_text(text, sim_config.chunking_strategy, sim_config.chunk_size)
        )

    chunks["sft"] = sft_chunks
    logger.info(f"  SFT: {len(sft_chunks)} chunks")

    logger.info("=" * 80 + "\n")
    return chunks


def embed_corpus(
    chunks: dict[str, list[str]],
    model: SentenceTransformer,
    cache_dir: Path
) -> dict[str, TrainingCorpus]:
    """Embed corpus chunks and cache to disk.

    Following Casey Muratori: Cache expensive operations.

    Args:
        chunks: Dict mapping stage -> list of text chunks
        model: SentenceTransformer for encoding
        cache_dir: Directory to cache embeddings

    Returns:
        Dict mapping stage -> TrainingCorpus (with embeddings)
    """
    logger.info("=" * 80)
    logger.info("Embedding corpus chunks")
    logger.info("=" * 80)

    cache_dir.mkdir(parents=True, exist_ok=True)
    corpora = {}

    for stage, chunk_list in chunks.items():
        cache_path = cache_dir / f"{stage}_embeddings.npy"

        if cache_path.exists():
            logger.info(f"\nLoading cached embeddings for {stage}...")
            embeddings = np.load(cache_path)
        else:
            logger.info(f"\nEmbedding {len(chunk_list)} chunks for {stage}...")
            embeddings = model.encode(
                chunk_list,
                batch_size=64,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            # Cache embeddings
            np.save(cache_path, embeddings)
            logger.info(f"  Cached to {cache_path}")

        # Normalize embeddings (required by TrainingCorpus)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = embeddings / norms

        # Create metadata (simple index-based)
        metadata = [
            {'shard_id': 0, 'chunk_id': i}
            for i in range(len(chunk_list))
        ]

        # Create TrainingCorpus (validates invariants)
        corpora[stage] = TrainingCorpus(
            stage=stage,  # type: ignore
            embeddings=embeddings_normalized,
            chunks=chunk_list,
            metadata=metadata
        )

        logger.info(f"  {stage}: {embeddings.shape}")

    logger.info("=" * 80 + "\n")
    return corpora


def load_gsm8k_samples(num_samples: int, split: str = "test") -> list[dict]:
    """Load GSM8K test samples.

    Args:
        num_samples: Number of samples to load
        split: Dataset split (default "test")

    Returns:
        List of dicts with 'question' and 'answer' keys
    """
    logger.info("=" * 80)
    logger.info(f"Loading GSM8K {split} samples")
    logger.info("=" * 80)

    dataset = load_dataset("openai/gsm8k", "main", split=split)
    dataset = dataset.select(range(min(num_samples, len(dataset))))  # type: ignore

    samples = []
    for i, item in enumerate(dataset):
        question = item['question']
        answer = item['answer'].split('####')[-1].strip()

        samples.append({
            'id': f"gsm8k_{i:04d}",
            'question': question,
            'answer': answer
        })

    logger.info(f"Loaded {len(samples)} GSM8K samples")
    logger.info("=" * 80 + "\n")
    return samples


def embed_gsm8k_samples(
    samples: list[dict],
    model: SentenceTransformer,
    model_answers: list[str] | None = None
) -> dict[str, np.ndarray]:
    """Embed GSM8K samples in 3-4 variants.

    Args:
        samples: List of GSM8K samples (dicts with 'question' and 'answer')
        model: SentenceTransformer for encoding
        model_answers: Optional list of model-generated answers

    Returns:
        Dict mapping variant -> list of embeddings
        {
            "question": [emb1, emb2, ...],
            "answer": [emb1, emb2, ...],
            "question_answer": [emb1, emb2, ...],
            "model_answer": [emb1, emb2, ...]  # if model_answers provided
        }
    """
    num_variants = 3 + (1 if model_answers else 0)
    logger.info("=" * 80)
    logger.info(f"Embedding GSM8K samples ({num_variants} variants)")
    logger.info("=" * 80)

    variants = {}

    # Question only
    logger.info("\n1. Question variant...")
    questions = [s['question'] for s in samples]
    variants['question'] = model.encode(
        questions,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Answer only (ground truth)
    logger.info("\n2. Answer variant (ground truth)...")
    answers = [s['answer'] for s in samples]
    variants['answer'] = model.encode(
        answers,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Question + Answer
    logger.info("\n3. Question+Answer variant...")
    question_answers = [f"{s['question']} Answer: {s['answer']}" for s in samples]
    variants['question_answer'] = model.encode(
        question_answers,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Model answers (if provided)
    if model_answers:
        logger.info("\n4. Model answer variant...")
        variants['model_answer'] = model.encode(
            model_answers,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

    logger.info("=" * 80 + "\n")
    return variants


def generate_model_answers(
    samples: list[dict],
    config: Config
) -> list[str]:
    """Generate model answers for GSM8K samples using inference API.

    Uses asyncio.to_thread to avoid async spreading through the codebase.

    Args:
        samples: List of GSM8K samples (dicts with 'question' and 'answer')
        config: Config with model endpoint settings

    Returns:
        List of model-generated answer strings
    """
    logger.info("=" * 80)
    logger.info("Generating model answers (requires API calls)")
    logger.info("=" * 80)
    logger.info(f"Model: {config.similarity.model_name}")
    logger.info(f"API: {config.similarity.model_api_base}")
    logger.info(f"Temperature: {config.similarity.model_temperature}")
    logger.info(f"Samples: {len(samples)}\n")

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key and "api.openai.com" in config.similarity.model_api_base:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Set it with: export OPENAI_API_KEY=your-key"
        )

    # Create endpoint
    endpoint = Endpoint(
        model=config.similarity.model_name,
        api_base=config.similarity.model_api_base,
        api_key=api_key,
        temperature=config.similarity.model_temperature,
        max_tokens=config.similarity.model_max_tokens
    )

    async def generate_all():
        """Generate answers for all samples."""
        model_answers = []

        for i, sample in enumerate(samples):
            logger.info(f"Generating answer {i + 1}/{len(samples)}...")

            # Convert to GSM8KSample and then to Rollout
            gsm8k_sample = GSM8KSample(
                question=sample['question'],
                answer=sample['answer'],
                sample_id=sample['id']
            )
            rollout = gsm8k_sample.to_rollout()

            # Generate completion
            updated_rollout = await generate(endpoint, rollout)

            # Extract model answer
            model_answer = updated_rollout.get_last_message_content()
            if model_answer:
                model_answers.append(model_answer)
            else:
                logger.warning(f"  No answer generated for sample {sample['id']}")
                model_answers.append("")  # Empty string as fallback

        return model_answers

    # Run async code without spreading async through the codebase
    # asyncio.to_thread runs the coroutine in a thread pool
    model_answers = asyncio.run(generate_all())

    logger.info(f"\nGenerated {len(model_answers)} model answers")
    logger.info("=" * 80 + "\n")
    return model_answers


def search_all(
    samples: list[dict],
    gsm8k_embeddings: dict[str, np.ndarray],
    corpora: dict[str, TrainingCorpus],
    model: SentenceTransformer,
    k: int,
    distance_metric: str
) -> list[SimilarityResult]:
    """Search all variants against all corpus stages.

    Args:
        samples: GSM8K samples
        gsm8k_embeddings: Dict mapping variant -> embeddings array
        corpora: Dict mapping stage -> TrainingCorpus
        model: SentenceTransformer (not used, kept for API consistency)
        k: Number of nearest neighbors
        distance_metric: "cosine" | "euclidean" | "manhattan"

    Returns:
        List of SimilarityResult objects
    """
    logger.info("=" * 80)
    logger.info("Searching all variants against all corpus stages")
    logger.info("=" * 80)

    # Select distance function
    distance_fns = {
        "cosine": cosine_distance,
        "euclidean": euclidean_distance,
        "manhattan": manhattan_distance
    }
    distance_fn = distance_fns.get(distance_metric, cosine_distance)
    logger.info(f"Using distance metric: {distance_metric}\n")

    all_results = []

    # For each variant (question, answer, question_answer)
    for variant_name, embeddings_array in gsm8k_embeddings.items():
        logger.info(f"Variant: {variant_name}")

        # For each sample
        for sample_idx, sample in enumerate(samples):
            sample_id = sample['id']
            query_embedding = embeddings_array[sample_idx]

            # Normalize query embedding
            query_normalized = query_embedding / np.linalg.norm(query_embedding)

            # Search each corpus stage
            for stage_name, corpus in corpora.items():
                # Compute distances to all chunks in this corpus
                distances = np.array([
                    distance_fn(query_normalized, corpus.embeddings[i])
                    for i in range(len(corpus.embeddings))
                ])

                # Get top-k indices
                top_indices = np.argsort(distances)[:k]

                # Create results
                for rank, idx in enumerate(top_indices, start=1):
                    all_results.append(SimilarityResult(
                        sample_id=sample_id,
                        variant=variant_name,
                        corpus_stage=stage_name,
                        distance=float(distances[idx]),
                        nearest_chunk_text=corpus.chunks[idx],
                        rank=rank
                    ))

        logger.info(f"  Processed {len(samples)} samples\n")

    logger.info(f"Total results: {len(all_results)}")
    logger.info("=" * 80 + "\n")
    return all_results


def save_results(results: list[SimilarityResult], output_path: Path):
    """Save results to CSV.

    Args:
        results: List of SimilarityResult objects
        output_path: Path to output CSV file
    """
    logger.info("=" * 80)
    logger.info(f"Saving results to {output_path}")
    logger.info("=" * 80)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'sample_id',
            'variant',
            'corpus_stage',
            'rank',
            'distance',
            'chunk_text'
        ])

        # Data
        for result in results:
            writer.writerow([
                result.sample_id,
                result.variant,
                result.corpus_stage,
                result.rank,
                result.distance,
                result.nearest_chunk_text[:200]  # Truncate for CSV readability
            ])

    logger.info(f"Saved {len(results)} results")
    logger.info("=" * 80 + "\n")


def main():
    import importlib.util
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load config
    if len(sys.argv) > 1 and sys.argv[1].endswith('.py'):
        # Load config from experiment file
        spec = importlib.util.spec_from_file_location("exp_config", sys.argv[1])
        if spec is None:
            raise ImportError(f"Could not load spec from {sys.argv[1]}")
        if spec.loader is None:
            raise ImportError(f"Spec has no loader: {sys.argv[1]}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config: Config = module.config
    else:
        # Use default config
        config = Config()

    logger.info("\n" + "=" * 80)
    logger.info("GSM8K Corpus Similarity Measurement")
    logger.info("=" * 80)
    logger.info(f"Samples: {config.similarity.num_gsm8k_samples}")
    logger.info(f"Corpus sizes: {config.similarity.corpus_sizes}")
    logger.info(f"Chunking: {config.similarity.chunking_strategy}")
    logger.info(f"Distance: {config.similarity.distance_metric}")
    logger.info(f"k neighbors: {config.similarity.k_neighbors}")
    logger.info("=" * 80 + "\n")

    try:
        # Load model
        logger.info("Loading sentence transformer model...")
        model = SentenceTransformer(
            config.embedding.model,
            device=config.embedding.device
        )
        logger.info(f"Model: {config.embedding.model}\n")

        # 1. Load corpus chunks
        corpus_chunks = load_corpus_chunks(config)

        # 2. Embed corpus (with caching)
        cache_dir = config.similarity.output_dir / "embeddings_cache"
        corpora = embed_corpus(corpus_chunks, model, cache_dir)

        # 3. Load GSM8K samples
        gsm8k_samples = load_gsm8k_samples(
            num_samples=config.similarity.num_gsm8k_samples,
            split=config.similarity.gsm8k_split
        )

        # 4. Generate model answers (optional - requires API calls)
        model_answers = None
        if config.similarity.include_model_answers:
            model_answers = generate_model_answers(gsm8k_samples, config)

        # 5. Embed GSM8K variants
        gsm8k_embeddings = embed_gsm8k_samples(gsm8k_samples, model, model_answers)

        # 6. Search all
        results = search_all(
            samples=gsm8k_samples,
            gsm8k_embeddings=gsm8k_embeddings,
            corpora=corpora,
            model=model,
            k=config.similarity.k_neighbors,
            distance_metric=config.similarity.distance_metric
        )

        # 7. Save results
        output_path = config.similarity.output_dir / config.similarity.output_file
        save_results(results, output_path)

        # Summary
        num_variants = len(gsm8k_embeddings)
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"GSM8K samples: {len(gsm8k_samples)}")
        variant_list = list(gsm8k_embeddings.keys())
        logger.info(f"Variants per sample: {num_variants} ({', '.join(variant_list)})")
        if model_answers:
            logger.info("  (includes model-generated answers)")
        logger.info(f"Corpus stages: {list(corpora.keys())}")
        logger.info(f"Results per sample: {config.similarity.k_neighbors * num_variants * len(corpora)}")
        logger.info(f"Total results: {len(results)}")
        logger.info(f"Output: {output_path}")
        logger.info("=" * 80)

        logger.info("\nDone!")
        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
