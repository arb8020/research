"""Test loading all available CFG configs."""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "PhysicsLM4" / "data-synthetic-pretrain" / "Lano-cfg"))

from data_cfg import CFG_Config


def test_all_cfgs():
    """Test loading all CFG configs."""
    cfg_dir = Path(__file__).parent.parent.parent / "PhysicsLM4" / "data-synthetic-pretrain" / "Lano-cfg" / "configs"
    
    cfg_files = sorted(cfg_dir.glob("*.json"))
    
    print(f"Found {len(cfg_files)} CFG config files:\n")
    
    for cfg_file in cfg_files:
        try:
            config = CFG_Config.from_graph(str(cfg_file))
            
            # Try to generate a sequence
            import random
            rng = random.Random(42)
            seq = config.generate_onedata_pure(rng)
            
            # Verify it
            correct, _, _, _ = config.solve_dp_noneq_fast(seq, no_debug=True)
            is_valid = (correct == 0)
            
            print(f"✓ {cfg_file.name:12s} - depth={config.depth}, vocab={config.vocab_size}, "
                  f"seq_len={len(seq)}, valid={is_valid}")
            
        except Exception as e:
            print(f"✗ {cfg_file.name:12s} - ERROR: {e}")


if __name__ == "__main__":
    test_all_cfgs()
