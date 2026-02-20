"""Demo of iGSM retry/correction data."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Setup paths
sys.path = [p for p in sys.path if not (('rollouts' in p or 'research' in p) and 'site-packages' not in p)]
sys.path.insert(0, "/tmp/iGSM")
sys.path.insert(0, str(Path(__file__).parent))
for mod_name in ['matplotlib', 'matplotlib.pyplot', 'matplotlib.patches', 'matplotlib.lines', 'matplotlib.colors', 'matplotlib.cm']:
    sys.modules[mod_name] = MagicMock()

from igsm_retry_generator import build_igsm_retry_loader
from tools.tools import tokenizer


def demo_retry():
    """Show example of retry data."""
    print("=" * 70)
    print("iGSM Retry/Correction Data Demo")
    print("=" * 70)
    
    # Generate with retries
    loader = build_igsm_retry_loader(
        difficulty="med",
        retry_rate=0.5,  # High rate for demo
        retry_type="strong",
        seq_len=512,
        batch_size=1,
        device="cpu",
        seed=42,
    )
    
    input_ids, _ = loader.next()
    tokens = input_ids[0].tolist()
    
    # Parse tokens
    prob_start = 222
    sol_start = 223
    ans_start = 224
    eos = 50256
    
    idx_222 = tokens.index(prob_start)
    idx_223 = tokens.index(sol_start)
    idx_224 = tokens.index(ans_start)
    idx_eos = tokens.index(eos) if eos in tokens else len(tokens)
    
    prob_tokens = tokens[idx_222+1:idx_223]
    sol_tokens = tokens[idx_223+1:idx_224]
    ans_tokens = tokens[idx_224+1:idx_eos]
    
    prob_text = tokenizer.decode(prob_tokens)
    sol_text = tokenizer.decode(sol_tokens)
    ans_text = tokenizer.decode(ans_tokens)
    
    print("\n📋 PROBLEM:")
    print("-" * 70)
    print(prob_text)
    
    print("\n🔧 SOLUTION (with retries):")
    print("-" * 70)
    print(sol_text)
    
    print("\n📊 ANSWER:")
    print("-" * 70)
    print(f"   {ans_text.strip()}")
    
    print("\n" + "=" * 70)
    print("How it works:")
    print("=" * 70)
    print("""
The model learns to:
1. Make a step in the solution
2. Realize it's using the WRONG parameter
3. Say "BACK" to indicate a retry
4. Use the CORRECT parameter instead

This teaches the model to recover from mistakes during reasoning.

The retry tokens are inserted as:
  [2896, 500] + param_tokens + [355] + [" BACK."]

In the solution text, this appears as:
  " Define <wrong_param> as ... BACK. Define <correct_param> as ..."
    """)


if __name__ == "__main__":
    demo_retry()
