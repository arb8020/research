# nanochat vs rollouts/pretrain Gap Analysis

## What rollouts/pretrain HAS:
- `train.py` - basic training loop with distributed support
- `config.py` - dataclass configs with fingerprinting
- `runtime.py` - distributed setup (torchrun, all_reduce)
- `schedule.py` - LR scheduler
- `models/llama.py` - functional Llama model
- `configs/` - Python config files (tiny, small)
- Checkpoint save/resume, gradient accumulation, mixed precision (bf16)

## What nanochat HAS that rollouts is MISSING:

### High Priority (core training features):

1. **Muon Optimizer** (`optim.py`)
   - Much more efficient than AdamW for transformer weights
   - `MuonAdamW` (single GPU) and `DistMuonAdamW` (distributed)
   - Uses Polar Express orthogonalization + variance reduction
   - AdamW for embeddings, Muon for matrix params
   - Fused kernels via torch.compile

2. **BPE Tokenizer** (`tokenizer.py`)
   - Custom tokenizer training + tiktoken inference
   - Special tokens for chat (user_start, assistant_end, python_start, etc.)
   - HuggingFace training, tiktoken fast inference

3. **Better Dataloader** (`dataloader.py`)
   - BOS-aligned best-fit packing
   - Every row starts with BOS token
   - Best-fit algorithm minimizes cropping (~35% vs naive)
   - 100% utilization (no padding)

4. **Evaluation Pipeline** (`core_eval.py`, `loss_eval.py`)
   - CORE metric evaluation
   - BPB (bits per byte) evaluation on val set

### Medium Priority (architecture enhancements):

5. **Modern Model Features** (`gpt.py`)
   - QK Norm (normalize Q and K after RoPE)
   - ReLU^2 activation (instead of GELU/SwiGLU)
   - Value Embeddings (ResFormer-style, alternating layers)
   - Per-layer residual/x0 lambdas (learnable scalars)
   - Sliding window attention patterns (e.g., "SSSL")
   - Untied embedding/lm_head weights
   - Flash Attention 3 integration
   - Logit softcap (15.0)

6. **Inference Engine** (`engine.py`)
   - KV cache for efficient generation
   - Calculator tool integration (python REPL)
   - Batched generation with multiple samples
   - Tool use state machine

7. **Checkpoint Manager** (`checkpoint_manager.py`)
   - More sophisticated save/load
   - Model phases (base, chat, etc.)

### Lower Priority (nice to have):

8. **Evals/Tasks** (`tasks/`)
   - ARC, GSM8K, MMLU, HumanEval, SpellingBee
   - Common task infrastructure

9. **Chat/SFT/RL scripts** (`scripts/`)
   - `chat_sft.py` - supervised finetuning
   - `chat_rl.py` - reinforcement learning
   - `chat_cli.py`, `chat_web.py` - inference interfaces

10. **FP8 Training** (`fp8.py`)
    - H100+ GPU training with FP8
    - Tensorwise and rowwise scaling recipes

11. **Reporting/UI** (`report.py`, `ui.html`)
    - Training visualization
    - Web interface for chat

## Recommended Implementation Order:

1. **Muon optimizer** - biggest training efficiency win
2. **Better dataloader** - BOS-aligned packing
3. **Tokenizer** - needed for real data
4. **QK Norm + ReLU^2** - simple arch improvements
5. **Eval pipeline** - measure progress
6. **Inference engine** - for sampling/eval
7. **SFT/RL** - finetuning capabilities
