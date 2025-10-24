"""Annotation engine for mapping model outputs to corpus clusters."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from types import ModuleType
from typing import Iterable, List, Optional

import numpy as np

from corpus_index import CorpusIndex

syntok_segmenter: Optional[ModuleType] = None
try:
    from syntok import segmenter as syntok_segmenter  # type: ignore
except ModuleNotFoundError:
    pass  # Already None


logger = logging.getLogger(__name__)


@dataclass
class ClusterAnnotation:
    """Single annotation linking a text span to a corpus cluster."""

    text_span: str
    cluster_id: str
    cluster_name: str
    cluster_depth: int
    distance: float
    rank: int
    corpus_stage: Optional[str]
    nearest_chunk_idx: int
    nearest_chunk_text: str
    avg_logprob: Optional[float] = None


@dataclass
class AnnotatedOutput:
    """Full annotation bundle for a single model output."""

    prompt: Optional[str]
    text: str
    annotations: List[ClusterAnnotation]
    timestamp: str
    corpus_index_path: str
    k: int
    phrase_level: bool

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "text": self.text,
            "annotations": [asdict(a) for a in self.annotations],
            "annotation_metadata": {
                "timestamp": self.timestamp,
                "corpus_index_path": self.corpus_index_path,
                "k": self.k,
                "phrase_level": self.phrase_level,
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "AnnotatedOutput":
        metadata = payload.get("annotation_metadata", {})
        annotations = [ClusterAnnotation(**item) for item in payload.get("annotations", [])]
        return cls(
            prompt=payload.get("prompt"),
            text=payload.get("text", ""),
            annotations=annotations,
            timestamp=metadata.get("timestamp", ""),
            corpus_index_path=metadata.get("corpus_index_path", ""),
            k=metadata.get("k", 0),
            phrase_level=metadata.get("phrase_level", False),
        )


_warned_no_segmenter = False


def _split_sentences_fallback(text: str) -> List[str]:
    """Simple regex-based sentence splitter used when syntok is unavailable."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentence-like spans using syntok when available."""
    if not text.strip():
        return []

    if syntok_segmenter is not None:
        sentences: List[str] = []
        for paragraph in syntok_segmenter.analyze(text):  # type: ignore[attr-defined]
            for sentence in paragraph:
                sentence_text = "".join(token.spacing + token.value for token in sentence).strip()
                if sentence_text:
                    sentences.append(sentence_text)
        if sentences:
            return sentences

    global _warned_no_segmenter
    if syntok_segmenter is None and not _warned_no_segmenter:
        logger.warning("syntok not installed; using simple sentence splitter")
        _warned_no_segmenter = True

    return _split_sentences_fallback(text)


def compute_distances(query_emb: np.ndarray, corpus_embs: np.ndarray) -> np.ndarray:
    """Compute cosine distances between a query embedding and corpus embeddings."""
    assert query_emb.ndim == 1, "Query embedding must be a 1D vector"
    assert corpus_embs.ndim == 2, "Corpus embeddings must be a 2D array"
    similarities = corpus_embs @ query_emb
    return 1.0 - similarities


def annotate_text(
    corpus_index: CorpusIndex,
    text: str,
    *,
    k: int = 3,
    phrase_level: bool = True,
    prompt: Optional[str] = None,
    logprobs: Optional[Iterable[float]] = None,
    nearest_chunk_preview_chars: int = 200,
) -> AnnotatedOutput:
    """Annotate text with nearest training corpus clusters."""
    assert k > 0, "k must be positive"
    assert nearest_chunk_preview_chars > 0, "Preview length must be positive"

    phrases: List[str]
    if phrase_level:
        phrases = split_into_sentences(text)
        if not phrases and text:
            phrases = [text]
    else:
        phrases = [text] if text else []

    annotations: List[ClusterAnnotation] = []

    if not phrases:
        timestamp = datetime.now(timezone.utc).isoformat()
        return AnnotatedOutput(
            prompt=prompt,
            text=text,
            annotations=annotations,
            timestamp=timestamp,
            corpus_index_path=str(corpus_index.index_path),
            k=k,
            phrase_level=phrase_level,
        )

    phrase_embeddings = corpus_index.model.encode(
        phrases,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    num_chunks = corpus_index.num_chunks()
    effective_k = min(k, num_chunks)
    assert effective_k > 0, "Corpus index must contain at least one chunk"

    for phrase_idx, (phrase, phrase_emb) in enumerate(zip(phrases, phrase_embeddings)):
        distances = compute_distances(phrase_emb, corpus_index.embeddings)
        top_indices = np.argsort(distances)[:effective_k]

        for rank, chunk_idx in enumerate(top_indices, start=1):
            chunk_idx_int = int(chunk_idx)
            cluster_id = corpus_index.chunk_to_cluster.get(chunk_idx_int, "unknown")
            cluster_info = corpus_index.get_cluster_info(cluster_id)

            metadata = corpus_index.metadata[chunk_idx_int]
            corpus_stage = metadata.get("corpus_stage")

            nearest_chunk = corpus_index.chunks[chunk_idx_int]
            preview = nearest_chunk[:nearest_chunk_preview_chars]

            annotation = ClusterAnnotation(
                text_span=phrase,
                cluster_id=cluster_id,
                cluster_name=cluster_info.get("name", "Unknown"),
                cluster_depth=cluster_info.get("depth", 0),
                distance=float(distances[chunk_idx_int]),
                rank=rank,
                corpus_stage=corpus_stage,
                nearest_chunk_idx=chunk_idx_int,
                nearest_chunk_text=preview,
                avg_logprob=None,
            )
            annotations.append(annotation)

    timestamp = datetime.now(timezone.utc).isoformat()
    return AnnotatedOutput(
        prompt=prompt,
        text=text,
        annotations=annotations,
        timestamp=timestamp,
        corpus_index_path=str(corpus_index.index_path),
        k=k,
        phrase_level=phrase_level,
    )
