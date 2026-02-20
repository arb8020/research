"""Evaluation utilities for CFG-based pretraining.

Provides:
1. CFG sequence verification (check if sequence satisfies CFG)
2. Next-token prediction accuracy using ground-truth CFG probabilities
3. KL-divergence between model predictions and CFG ground truth

Usage:
    python eval_cfg.py --cfg configs/cfg3f.json --checkpoint output/cfg_tiny/checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Import from rollouts
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "rollouts"))
from rollouts.pretrain.models.llama import forward, init_weights
from rollouts.pretrain.config import ModelConfig

# Import from PhysicsLM4
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "PhysicsLM4" / "data-synthetic-pretrain" / "Lano-cfg"))
from data_cfg import CFG_Config as PhysicsCFGConfig

from cfg_generator import CFGConfig


def verify_sequence(cfg_path: str | Path, sequence: list[int]) -> bool:
    """Verify if a sequence satisfies the CFG.
    
    Args:
        cfg_path: Path to CFG JSON file
        sequence: Token sequence to verify
        
    Returns:
        True if sequence satisfies CFG, False otherwise
    """
    config = PhysicsCFGConfig.from_graph(str(cfg_path))
    correct, _, _, _ = config.solve_dp_noneq_fast(sequence, no_debug=True)
    return correct == 0


def compute_ground_truth_probs(cfg_path: str | Path, sequence: list[int]) -> tuple[np.ndarray, list[float]]:
    """Compute ground-truth next-token probabilities from CFG.
    
    Args:
        cfg_path: Path to CFG JSON file
        sequence: Token sequence
        
    Returns:
        target_dist: [seq_len, vocab_size+1] probability distribution (includes EOS)
        probs_chosen: [seq_len] probability of the actual next token at each position
    """
    config = PhysicsCFGConfig.from_graph(str(cfg_path))
    target_dist, probs_chosen = config.solve_dp_prob_highprecision(sequence, debug=False)
    return np.array(target_dist), probs_chosen


def evaluate_model(
    model_weights: dict[str, torch.Tensor],
    model_config: ModelConfig,
    cfg_path: str | Path,
    num_sequences: int = 10,
    seq_len: int = 128,
    device: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    """Evaluate model on CFG-based metrics.
    
    Args:
        model_weights: Model weights dict
        model_config: Model configuration
        cfg_path: Path to CFG JSON file
        num_sequences: Number of sequences to evaluate
        seq_len: Sequence length
        device: Device to run on
        
    Returns:
        Dictionary of metrics
    """
    import random
    
    # Load CFG
    cfg_config = PhysicsCFGConfig.from_graph(str(cfg_path))
    rng = random.Random(42)
    
    total_loss = 0.0
    total_acc = 0.0
    total_tokens = 0
    total_kl = 0.0
    
    for _ in range(num_sequences):
        # Generate sequence
        seq = cfg_config.generate_onedata_pure(rng)
        seq = seq[:seq_len]
        if len(seq) < seq_len:
            seq = seq + [cfg_config.eos_token] * (seq_len - len(seq))
        
        # Get ground-truth probabilities
        try:
            target_dist, _ = compute_ground_truth_probs(cfg_path, seq)
        except Exception as e:
            print(f"Warning: Could not compute ground-truth probs: {e}")
            target_dist = None
        
        # Prepare tensors
        input_ids = torch.tensor([seq[:-1]], device=device)
        labels = torch.tensor([seq[1:]], device=device)
        
        # Forward pass
        with torch.no_grad():
            logits = forward(input_ids, model_weights, model_config)
            
            # Compute loss
            loss = F.cross_entropy(
                logits.reshape(-1, model_config.vocab_size),
                labels.reshape(-1),
                reduction='sum'
            )
            
            # Compute accuracy
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).sum().item()
            
            # Compute KL-divergence if we have ground-truth
            if target_dist is not None:
                # Get model probabilities
                log_probs = F.log_softmax(logits, dim=-1)
                probs = torch.exp(log_probs)
                
                # Convert target_dist to torch (skip EOS for now)
                target_torch = torch.tensor(target_dist[1:], dtype=torch.float32, device=device)
                
                # KL(target || model) = sum(target * (log(target) - log(model)))
                # We need to align vocab sizes
                min_vocab = min(target_torch.shape[1], probs.shape[2])
                kl = F.kl_div(
                    log_probs[0, :, :min_vocab],
                    target_torch[:, :min_vocab],
                    reduction='sum',
                    log_target=False
                )
                total_kl += kl.item()
            
            total_loss += loss.item()
            total_acc += acc
            total_tokens += labels.numel()
    
    metrics = {
        "loss": total_loss / total_tokens,
        "perplexity": np.exp(total_loss / total_tokens),
        "accuracy": total_acc / total_tokens,
    }
    
    if total_kl > 0:
        metrics["kl_divergence"] = total_kl / total_tokens
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate CFG-based pretraining")
    parser.add_argument("--cfg", required=True, help="Path to CFG JSON file")
    parser.add_argument("--checkpoint", help="Path to model checkpoint")
    parser.add_argument("--num-seq", type=int, default=10, help="Number of sequences to evaluate")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--verify", help="Verify a sequence (comma-separated tokens)")
    args = parser.parse_args()
    
    if args.verify:
        # Verify a specific sequence
        sequence = [int(x.strip()) for x in args.verify.split(",")]
        is_valid = verify_sequence(args.cfg, sequence)
        print(f"Sequence: {sequence}")
        print(f"Valid CFG: {is_valid}")
        return
    
    if args.checkpoint:
        # Evaluate model checkpoint
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        
        # Load config from checkpoint
        config_dict = ckpt.get("config", {})
        model_config_dict = config_dict.get("config", {}).get("model", {})
        
        model_config = ModelConfig(
            dim=model_config_dict.get("dim", 256),
            n_layers=model_config_dict.get("n_layers", 4),
            n_heads=model_config_dict.get("n_heads", 4),
            vocab_size=model_config_dict.get("vocab_size", 7),
        )
        
        # Load weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = init_weights(model_config, device, torch.float32)
        for key, value in ckpt["weights"].items():
            weights[key].copy_(value.to(device))
        
        print(f"Evaluating on {args.num_seq} sequences...")
        metrics = evaluate_model(
            weights,
            model_config,
            args.cfg,
            num_sequences=args.num_seq,
            seq_len=args.seq_len,
            device=device,
        )
        
        print("\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    else:
        # Just test CFG generation
        print(f"Testing CFG: {args.cfg}")
        cfg_config = PhysicsCFGConfig.from_graph(args.cfg)
        
        rng = __import__('random').Random(42)
        seq = cfg_config.generate_onedata_pure(rng)
        print(f"Generated sequence (len={len(seq)}): {seq[:50]}...")
        
        is_valid = verify_sequence(args.cfg, seq)
        print(f"Valid CFG: {is_valid}")


if __name__ == "__main__":
    main()
