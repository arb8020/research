"""Unit tests for annotation helpers (unittest-based)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from annotation import annotate_text, split_into_sentences
from cluster_corpus import ClusterNode, build_chunk_to_cluster_map
from corpus_index import CorpusIndex
from formatting import format_annotations_compact


class _StubModel:
    """Simple embedding stub for tests."""

    def encode(self, sentences, *, normalize_embeddings, convert_to_numpy):
        vectors = []
        for sentence in sentences:
            lower = sentence.lower()
            if "alpha" in lower:
                vectors.append(np.array([1.0, 0.0], dtype=np.float32))
            else:
                vectors.append(np.array([0.0, 1.0], dtype=np.float32))
        return np.vstack(vectors)


class AnnotationEngineTests(unittest.TestCase):
    def test_build_chunk_to_cluster_map_handles_noise(self):
        root = ClusterNode(
            cluster_id="0",
            depth=0,
            parent_id=None,
            indices=[0, 1, 2],
            centroid=np.zeros(2, dtype=np.float32),
            size=3,
            silhouette_score=0.5,
            children=[],
            noise_indices=[2],
        )
        child_a = ClusterNode(
            cluster_id="0.0",
            depth=1,
            parent_id="0",
            indices=[0],
            centroid=np.zeros(2, dtype=np.float32),
            size=1,
            silhouette_score=1.0,
            children=[],
            noise_indices=[],
        )
        child_b = ClusterNode(
            cluster_id="0.1",
            depth=1,
            parent_id="0",
            indices=[1],
            centroid=np.zeros(2, dtype=np.float32),
            size=1,
            silhouette_score=1.0,
            children=[],
            noise_indices=[],
        )
        root.children = [child_a, child_b]

        mapping = build_chunk_to_cluster_map(root)

        self.assertEqual(mapping[0], "0.0")
        self.assertEqual(mapping[1], "0.1")
        self.assertEqual(mapping[2], "0")  # Noise assigned to parent cluster

    def test_annotate_text_matches_expected_cluster(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            chunks = ["Alpha content", "Beta content"]
            metadata = [
                {"corpus_stage": "pretrain"},
                {"corpus_stage": "midtrain"},
            ]
            tree = {
                "cluster_id": "0",
                "depth": 0,
                "name": "Root",
                "children": [
                    {
                        "cluster_id": "0.alpha",
                        "depth": 1,
                        "name": "Alpha Cluster",
                        "children": [],
                    },
                    {
                        "cluster_id": "0.beta",
                        "depth": 1,
                        "name": "Beta Cluster",
                        "children": [],
                    },
                ],
            }
            chunk_to_cluster = {0: "0.alpha", 1: "0.beta"}

            index = CorpusIndex(
                embeddings=embeddings,
                chunks=chunks,
                metadata=metadata,
                tree=tree,
                chunk_to_cluster=chunk_to_cluster,
                model=_StubModel(),
                index_path=Path(tmpdir),
            )
            index._cluster_lookup = index._build_cluster_lookup(index.tree)

            result = annotate_text(index, "Alpha example.", k=1, phrase_level=False)

            self.assertTrue(result.annotations)
            top_ann = result.annotations[0]
            self.assertEqual(top_ann.cluster_name, "Alpha Cluster")
            self.assertEqual(top_ann.nearest_chunk_idx, 0)

            summary = format_annotations_compact(result)
            self.assertIn("Alpha Cluster", summary)

    def test_split_into_sentences_handles_blank_text(self):
        self.assertEqual(split_into_sentences(""), [])


if __name__ == "__main__":
    unittest.main()

