#!/usr/bin/env python3
"""LLM-based cluster naming and inspection utilities."""

import asyncio
import json
import logging
import numpy as np
import os
import random
from pathlib import Path
from typing import List, Optional, Dict
from dotenv import load_dotenv

from cluster_corpus import ClusterNode, get_cache_key, get_max_depth
from rollout import Endpoint, Rollout, Message, generate
from config import Config

logger = logging.getLogger(__name__)


# ────────────────────────────── Cluster Sampling ──────────────────────────────


def sample_cluster_texts(
    node: ClusterNode,
    embeddings: np.ndarray,
    texts: List[str],
    k: int = 5,
    strategy: str = "centroid"
) -> List[str]:
    """
    Sample representative texts from cluster.

    Args:
        node: Cluster node
        embeddings: Full corpus embeddings (N, D)
        texts: Full corpus texts (N,)
        k: Number of samples
        strategy: "centroid" (closest to center) or "random"

    Returns:
        List of sampled text strings
    """
    assert strategy in ("centroid", "random"), f"Unknown strategy: {strategy}"

    if len(node.indices) <= k:
        return [texts[i] for i in node.indices]

    if strategy == "random":
        # Random sampling
        sampled_indices = random.sample(node.indices, k)
        return [texts[i] for i in sampled_indices]

    # Centroid-based sampling: find k closest to cluster centroid
    cluster_embeddings = embeddings[node.indices]
    distances = np.linalg.norm(cluster_embeddings - node.centroid, axis=1)

    # Get k closest indices
    top_k_local = np.argsort(distances)[:k]
    sampled_indices = [node.indices[i] for i in top_k_local]

    return [texts[i] for i in sampled_indices]


def sample_noise_points(
    node: ClusterNode,
    texts: List[str],
    k: int = 5
) -> List[str]:
    """
    Sample random texts from noise points.

    Args:
        node: Cluster node
        texts: Full corpus texts (N,)
        k: Number of samples

    Returns:
        List of sampled noise point texts
    """
    if len(node.noise_indices) == 0:
        return []

    if len(node.noise_indices) <= k:
        return [texts[i] for i in node.noise_indices]

    sampled_indices = random.sample(node.noise_indices, k)
    return [texts[i] for i in sampled_indices]


# ────────────────────────────── LLM Naming ──────────────────────────────


async def generate_cluster_name(
    node: ClusterNode,
    embeddings: np.ndarray,
    texts: List[str],
    endpoint: Endpoint,
    parent_name: Optional[str] = None,
    num_samples: int = 5
) -> str:
    """
    Generate cluster name using LLM.

    Args:
        node: Cluster node
        embeddings: Full corpus embeddings
        texts: Full corpus texts
        endpoint: LLM endpoint (from rollout.py)
        parent_name: Parent cluster name (for subclusters)
        num_samples: Number of sample texts to show LLM

    Returns:
        Generated cluster name (2-5 words)
    """
    # Sample representative texts
    samples = sample_cluster_texts(node, embeddings, texts, k=num_samples, strategy="centroid")

    # Build prompt
    if parent_name:
        context = f"Parent cluster: \"{parent_name}\"\n\n"
    else:
        context = ""

    prompt = f"""{context}You are analyzing a cluster of training corpus texts. Based on these examples, provide a concise 2-5 word label describing the common theme.

Examples from this cluster (cluster has {node.size} total texts):
"""

    for i, text in enumerate(samples, 1):
        # Truncate long texts
        snippet = text[:200] + "..." if len(text) > 200 else text
        prompt += f"{i}. {snippet}\n"

    prompt += "\nCluster label (2-5 words only):"

    # Generate using rollout.py
    rollout = Rollout(messages=[Message(role="user", content=prompt)])

    try:
        updated_rollout = await generate(endpoint, rollout)

        # Extract label
        label = updated_rollout.get_last_message_content()
        if label:
            label = label.strip().strip('"').strip("'")
            # Truncate if too long
            words = label.split()
            if len(words) > 5:
                label = " ".join(words[:5])

        return label or f"Cluster {node.cluster_id}"

    except Exception as e:
        logger.error(f"Error generating name for cluster {node.cluster_id}: {e}")
        return f"Cluster {node.cluster_id}"


async def name_cluster_tree(
    root: ClusterNode,
    embeddings: np.ndarray,
    texts: List[str],
    endpoint: Endpoint
) -> ClusterNode:
    """
    Name all clusters in tree using breadth-first traversal.

    Process level-by-level to ensure parent names are available for subclusters.
    Use asyncio.gather() to parallelize within each level.

    Args:
        root: Root cluster node
        embeddings: Full corpus embeddings
        texts: Full corpus texts
        endpoint: LLM endpoint

    Returns:
        Root node with all names populated
    """
    # Get max depth
    max_depth = get_max_depth(root)

    logger.info(f"Naming clusters (max depth: {max_depth})")

    # Breadth-first traversal by depth
    for depth in range(max_depth + 1):
        # Collect all nodes at this depth
        nodes_at_depth = []
        collect_nodes_at_depth(root, depth, nodes_at_depth)

        logger.info(f"Naming {len(nodes_at_depth)} clusters at depth {depth}")

        # Name all nodes at this depth in parallel
        tasks = []
        for node in nodes_at_depth:
            # Get parent name if available
            parent_name = None
            if node.parent_id:
                parent_node = find_node_by_id(root, node.parent_id)
                if parent_node:
                    parent_name = parent_node.name

            tasks.append(generate_cluster_name(node, embeddings, texts, endpoint, parent_name))

        # Await all names for this level
        names = await asyncio.gather(*tasks)

        # Assign names
        for node, name in zip(nodes_at_depth, names):
            node.name = name
            logger.info(f"  {node.cluster_id}: \"{name}\" (size={node.size})")

    return root


# ────────────────────────────── Tree Traversal Helpers ──────────────────────────────


def collect_nodes_at_depth(node: ClusterNode, target_depth: int, result: List[ClusterNode]):
    """Collect all nodes at target depth (recursive helper)."""
    if node.depth == target_depth:
        result.append(node)
    for child in node.children:
        collect_nodes_at_depth(child, target_depth, result)


def find_node_by_id(node: ClusterNode, cluster_id: str) -> Optional[ClusterNode]:
    """Find node by cluster_id (recursive helper)."""
    if node.cluster_id == cluster_id:
        return node
    for child in node.children:
        found = find_node_by_id(child, cluster_id)
        if found:
            return found
    return None


def list_all_clusters(node: ClusterNode) -> List[ClusterNode]:
    """Get flat list of all clusters (depth-first)."""
    result = [node]
    for child in node.children:
        result.extend(list_all_clusters(child))
    return result


# ────────────────────────────── Inspection Utilities ──────────────────────────────


def print_cluster_tree(node: ClusterNode, indent: int = 0):
    """Pretty-print cluster tree."""
    prefix = "  " * indent
    name_str = f' "{node.name}"' if node.name else ""
    noise_str = f" ({len(node.noise_indices)} noise)" if node.noise_indices else ""

    print(f"{prefix}[{node.cluster_id}]{name_str} size={node.size}, sil={node.silhouette_score:.3f}{noise_str}")

    for child in node.children:
        print_cluster_tree(child, indent + 1)


def inspect_cluster(
    cluster_id: str,
    root: ClusterNode,
    texts: List[str],
    num_samples: int = 5,
    show_noise: bool = False
):
    """
    Inspect a specific cluster by ID.

    Args:
        cluster_id: Cluster ID to inspect
        root: Root cluster node
        texts: Full corpus texts
        num_samples: Number of samples to show
        show_noise: Whether to show noise points
    """
    node = find_node_by_id(root, cluster_id)

    if node is None:
        print(f"Cluster '{cluster_id}' not found")
        return

    print("="*80)
    print(f"Cluster: {cluster_id}")
    if node.name:
        print(f"Name: {node.name}")
    print(f"Size: {node.size}")
    print(f"Depth: {node.depth}")
    print(f"Silhouette: {node.silhouette_score:.3f}")
    print(f"Noise points: {len(node.noise_indices)}")
    print(f"Children: {len(node.children)}")
    print("="*80)

    # Show random samples
    print(f"\nRandom samples ({min(num_samples, len(node.indices))} of {len(node.indices)}):\n")

    sample_indices = random.sample(node.indices, min(num_samples, len(node.indices)))

    for i, idx in enumerate(sample_indices, 1):
        text = texts[idx]
        snippet = text[:300] + "..." if len(text) > 300 else text
        print(f"{i}. {snippet}\n")

    # Show noise samples if requested
    if show_noise and node.noise_indices:
        print(f"\nNoise samples ({min(num_samples, len(node.noise_indices))} of {len(node.noise_indices)}):\n")

        noise_sample_indices = random.sample(node.noise_indices, min(num_samples, len(node.noise_indices)))

        for i, idx in enumerate(noise_sample_indices, 1):
            text = texts[idx]
            snippet = text[:300] + "..." if len(text) > 300 else text
            print(f"{i}. {snippet}\n")


# ────────────────────────────── Main CLI ──────────────────────────────


async def main():
    import sys
    import importlib.util
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser(description="Name clusters and inspect cluster tree")
    parser.add_argument("config", help="Config file (e.g., configs/clustering_01_tiny.py)")
    parser.add_argument("--name", action="store_true", help="Generate LLM names for clusters")
    parser.add_argument("--inspect", help="Inspect specific cluster by ID")
    parser.add_argument("--tree", action="store_true", help="Print cluster tree")
    parser.add_argument("--list", action="store_true", help="List all cluster IDs")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to show (default: 5)")
    parser.add_argument("--show-noise", action="store_true", help="Show noise points when inspecting")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load config
    spec = importlib.util.spec_from_file_location("exp_config", args.config)
    if spec is None:
        raise ImportError(f"Could not load spec from {args.config}")
    if spec.loader is None:
        raise ImportError(f"Spec has no loader: {args.config}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config: Config = getattr(module, "config")

    # Get cache directory
    cache_key = get_cache_key(config)
    cache_dir = config.clustering.cache_dir / cache_key
    tree_path = cache_dir / "tree.json"

    if not tree_path.exists():
        logger.error(f"Cluster tree not found at {tree_path}")
        logger.error("Run cluster_corpus.py first to generate clusters")
        return 1

    # Load embeddings and texts
    embedding_cache_dir = config.clustering.embedding_cache_dir / cache_key
    embeddings_path = embedding_cache_dir / "embeddings.npy"
    metadata_path = embedding_cache_dir / "metadata.jsonl"

    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)

    logger.info(f"Loading texts from {config.data.processed_dir / config.data.output_file}")

    # Load original chunks
    original_chunks = []
    chunks_path = config.data.processed_dir / config.data.output_file
    with open(chunks_path, 'r') as f:
        for line in f:
            original_chunks.append(json.loads(line))

    # Re-chunk to match embeddings (same logic as cluster_corpus.py)
    from transformers import AutoTokenizer
    from chunking import chunk_text

    tokenizer = AutoTokenizer.from_pretrained(config.clustering.embedding_model)

    texts = []
    for orig_chunk in original_chunks:
        sub_chunks = chunk_text(
            text=orig_chunk['text'],
            strategy=config.clustering.chunking_strategy,
            chunk_size=config.clustering.chunk_max_tokens,
            overlap_pct=config.clustering.chunk_overlap_pct,
            tokenizer=tokenizer
        )
        texts.extend(sub_chunks)

    logger.info(f"Loaded {len(texts)} texts")

    # Load cluster tree from JSON (reconstruct ClusterNode structure)
    logger.info(f"Loading cluster tree from {tree_path}")
    with open(tree_path) as f:
        tree_dict = json.load(f)

    # Reconstruct ClusterNode tree
    root = reconstruct_tree(tree_dict, embeddings, texts)

    # Execute requested action
    if args.name:
        success_marker = Path(".naming_complete")
        failure_marker = Path(".naming_failed")
        for marker in (success_marker, failure_marker):
            if marker.exists():
                marker.unlink()

        try:
            logger.info("Generating cluster names...")

            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                logger.error("OPENAI_API_KEY not set. Add it to .env file or export it")
                return 1

            endpoint = Endpoint(
                model=config.clustering.naming_model,
                api_base=config.clustering.naming_api_base,
                api_key=api_key,
                temperature=config.clustering.naming_temperature,
                max_tokens=config.clustering.naming_max_tokens
            )

            root = await name_cluster_tree(root, embeddings, texts, endpoint)

            from cluster_corpus import save_cluster_tree
            logger.info(f"Saving updated tree to {tree_path}")
            save_cluster_tree(root, texts, [], tree_path)
            success_marker.touch()
            logger.info(f"✅ Naming complete, marker: {success_marker}")
            logger.info("Done!")
        except Exception as exc:
            failure_marker.touch()
            logger.error(f"❌ Naming failed, marker: {failure_marker} ({exc})")
            raise

    elif args.inspect:
        # Inspect specific cluster
        inspect_cluster(args.inspect, root, texts, num_samples=args.samples, show_noise=args.show_noise)

    elif args.tree:
        # Print tree
        print_cluster_tree(root)

    elif args.list:
        # List all clusters
        all_clusters = list_all_clusters(root)
        print(f"Found {len(all_clusters)} clusters:\n")
        for node in all_clusters:
            name_str = f' "{node.name}"' if node.name else ""
            print(f"  {node.cluster_id}{name_str} (size={node.size}, depth={node.depth})")

    else:
        parser.print_help()

    return 0


def reconstruct_tree(tree_dict: Dict, embeddings: np.ndarray, texts: List[str]) -> ClusterNode:
    """Reconstruct ClusterNode tree from JSON dict.

    Note: We don't have the original indices stored, so we'll use empty lists.
    This is fine for inspection purposes.
    """
    def reconstruct_node(d: Dict, parent_id: Optional[str] = None) -> ClusterNode:
        # Create node (indices now stored in JSON)
        node = ClusterNode(
            cluster_id=d["cluster_id"],
            depth=d["depth"],
            parent_id=d.get("parent_id"),
            indices=d.get("indices", []),  # Load indices from JSON
            centroid=np.zeros(embeddings.shape[1]),  # Dummy centroid (not needed for inspection)
            size=d["size"],
            silhouette_score=d["silhouette_score"],
            children=[],
            noise_indices=d.get("noise_indices", []),  # Load noise indices from JSON
            name=d.get("name", ""),
            _skip_validation=True  # Skip validation for reconstruction
        )

        # Recursively reconstruct children
        for child_dict in d.get("children", []):
            child_node = reconstruct_node(child_dict, node.cluster_id)
            node.children.append(child_node)

        return node

    return reconstruct_node(tree_dict)


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
