"""Quick demo of CFG-based pretraining.

This script demonstrates:
1. Loading a CFG config
2. Generating sequences
3. Verifying sequences
4. Computing ground-truth probabilities
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "PhysicsLM4" / "data-synthetic-pretrain" / "Lano-cfg"))

from data_cfg import CFG_Config
from cfg_generator import build_cfg_loader, CFGConfig


def demo_basic():
    """Demo 1: Basic CFG loading and generation."""
    print("=" * 60)
    print("Demo 1: Basic CFG Loading and Generation")
    print("=" * 60)
    
    # Load a CFG config
    cfg_path = Path(__file__).parent.parent.parent / "PhysicsLM4" / "data-synthetic-pretrain" / "Lano-cfg" / "configs" / "cfg3f.json"
    config = CFG_Config.from_graph(str(cfg_path))
    
    print(f"\nCFG loaded from: {cfg_path}")
    print(f"  Depth: {config.depth}")
    print(f"  Num symbols: {config.num_sym}")
    print(f"  Vocab size: {config.vocab_size}")
    
    # Generate a sequence
    rng = random.Random(42)
    seq = config.generate_onedata_pure(rng)
    
    print(f"\nGenerated sequence (length: {len(seq)}):")
    print(f"  {seq[:50]}...")
    
    # Verify the sequence
    correct, _, _, _ = config.solve_dp_noneq_fast(seq, no_debug=True)
    print(f"\nValid CFG: {correct == 0}")
    
    # Corrupt the sequence and verify again
    corrupted = seq.copy()
    corrupted[0] = corrupted[0] % 3 + 1  # Flip first token
    correct_corrupted, _, _, _ = config.solve_dp_noneq_fast(corrupted, no_debug=True)
    print(f"Corrupted sequence valid: {correct_corrupted == 0}")


def demo_dataloader():
    """Demo 2: Using the CFG DataLoader."""
    print("\n" + "=" * 60)
    print("Demo 2: CFG DataLoader")
    print("=" * 60)
    
    cfg_path = Path(__file__).parent.parent.parent / "PhysicsLM4" / "data-synthetic-pretrain" / "Lano-cfg" / "configs" / "cfg3f.json"
    
    # Build data loader
    loader = build_cfg_loader(
        cfg_path=cfg_path,
        seq_len=64,
        batch_size=2,
        device="cpu",
        seed=42,
    )
    
    print(f"\nDataLoader created:")
    print(f"  Seq len: {loader.seq_len}")
    print(f"  Batch size: {loader.batch_size}")
    print(f"  CFG vocab size: {loader.config.vocab_size}")
    
    # Get a batch
    input_ids, labels = loader.next()
    
    print(f"\nBatch generated:")
    print(f"  Input IDs shape: {input_ids.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  First sequence: {input_ids[0].tolist()[:20]}...")
    print(f"  Labels match (shifted by 1): {(input_ids[:, 1:] == labels[:, :-1]).all().item()}")


def demo_ground_truth_probs():
    """Demo 3: Computing ground-truth probabilities."""
    print("\n" + "=" * 60)
    print("Demo 3: Ground-Truth Next-Token Probabilities")
    print("=" * 60)
    
    cfg_path = Path(__file__).parent.parent.parent / "PhysicsLM4" / "data-synthetic-pretrain" / "Lano-cfg" / "configs" / "cfg3f.json"
    config = CFG_Config.from_graph(str(cfg_path))
    
    # Generate a short sequence
    rng = random.Random(123)
    seq = config.generate_onedata_pure(rng)
    seq = seq[:10]  # Short sequence for demo
    
    print(f"\nSequence: {seq}")
    
    # Compute ground-truth probabilities
    target_dist, probs_chosen = config.solve_dp_prob_highprecision(seq, debug=False)
    
    print(f"\nGround-truth next-token distribution (first 5 positions):")
    for i in range(min(5, len(seq))):
        dist = target_dist[i]
        print(f"  Position {i}: EOS={dist[0]:.3f}, T1={dist[1]:.3f}, T2={dist[2]:.3f}, T3={dist[3]:.3f}")
        print(f"    -> Chose token {seq[i]} with prob {probs_chosen[i]:.6f}")


def demo_custom_cfg():
    """Demo 4: Creating a custom CFG."""
    print("\n" + "=" * 60)
    print("Demo 4: Creating Custom CFG")
    print("=" * 60)
    
    # Create a simple CFG
    config = CFGConfig(
        depth=4,
        num_sym=3,
        vocab_size=3,
        deg_min=2,
        deg_max=2,
        len_min=1,
        len_max=2,
        num_sym_mode=2,
    )
    
    print(f"\nCustom CFG created:")
    print(f"  Depth: {config.depth}")
    print(f"  Num symbols: {config.num_sym}")
    print(f"  Total nodes: {config.count}")
    
    # Generate sequences
    rng = random.Random(42)
    for i in range(3):
        seq = config.generate_sequence(rng)
        print(f"  Sequence {i+1}: {seq[:20]}... (len={len(seq)})")
    
    # Save and reload
    test_path = "/tmp/test_cfg.json"
    config.save_graph(test_path)
    print(f"\nSaved to: {test_path}")
    
    loaded = CFGConfig.from_graph(test_path)
    print(f"Reloaded: depth={loaded.depth}, num_sym={loaded.num_sym}")


def main():
    """Run all demos."""
    demo_basic()
    demo_dataloader()
    demo_ground_truth_probs()
    demo_custom_cfg()
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
