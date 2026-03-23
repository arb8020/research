# Known Green RL Configs

This file is the small supported set we have actually re-run to green on Modal.

If a config is not listed here, treat it as exploratory until it is revalidated.

## Green Now

### Dense Qwen3 on TorchTitan + vLLM

- Config: [qwen/grpo_qwen3_0_6b_torchtitan_modal_witness.py](/Users/chiraagbalu/research/rollouts/examples/rl/qwen/grpo_qwen3_0_6b_torchtitan_modal_witness.py)
- Stack:
  - trainer: `torchtitan`
  - inference: `qed-vllm`
  - provider: `modal`
- What this proves:
  - reverse-text RL loop completes
  - checkpointing works
  - runtime weight sync works

### Dense Qwen3 on Megatron + Slime-SGLang

- Config: [qwen/grpo_qwen3_0_6b_modal_megatron_witness.py](/Users/chiraagbalu/research/rollouts/examples/rl/qwen/grpo_qwen3_0_6b_modal_megatron_witness.py)
- Stack:
  - trainer: `megatron`
  - inference: `slime-sglang`
  - provider: `modal`
- What this proves:
  - reverse-text RL loop completes
  - isolated runtime NCCL weight sync works
  - the dense Megatron/SGLang path is alive again

## Not Green Yet

These are still important, but they are not part of the trusted minimal surface:

- small MoE witnesses
- GLM MoE / Megatron / Slime-SGLang
- any Prime RL backend integration
- RunPod witnesses not revalidated in this cleanup pass

## Why This File Exists

We had too many witness-shaped configs and too much ambiguity about what was
actually healthy. This file is intentionally narrow.

The current supported surface is:

1. one dense TorchTitan + vLLM witness
2. one dense Megatron + Slime-SGLang witness

That is enough to cover the trainer/runtime space we have actually repaired so
far without pretending the rest of the matrix is green.

Thin researcher-facing entrypoints for these now live in:

- [configs/trusted/rl/qwen3_0_6b_torchtitan_modal.py](/Users/chiraagbalu/research/rollouts/configs/trusted/rl/qwen3_0_6b_torchtitan_modal.py)
- [configs/trusted/rl/qwen3_0_6b_megatron_modal.py](/Users/chiraagbalu/research/rollouts/configs/trusted/rl/qwen3_0_6b_megatron_modal.py)
