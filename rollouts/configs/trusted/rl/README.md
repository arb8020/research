# Trusted RL Configs

This directory is the small researcher-facing RL surface we currently trust.

These configs are intentionally thin. They re-export the known-green workload
shapes from `examples/rl/` so people have a boring place to start from without
copying witness files directly.

Current trusted configs:

- `qwen3_0_6b_torchtitan_modal.py`
  - dense reverse-text RL
  - trainer: TorchTitan
  - inference: QED-vLLM
  - provider: Modal
- `qwen3_0_6b_megatron_modal.py`
  - dense reverse-text RL
  - trainer: Megatron
  - inference: Slime-SGLang
  - provider: Modal

Each trusted config now also exports a `worker_topology` sketch so eval and RL
can converge on the same "allocation -> named workers -> semantic roles"
denotation over time.

Usage:

```bash
python -m argus run --config configs/trusted/rl/qwen3_0_6b_torchtitan_modal.py
python -m argus run --config configs/trusted/rl/qwen3_0_6b_megatron_modal.py
```

Copy one of these into a new config file when experimenting. Keep the copied
config thin:

- import task logic from `examples/rl/*/base_config.py`
- import known-good stack shapes from this directory or the underlying witness
- change only the semantic axes you actually mean to vary

Do not treat older witness files in `examples/rl/` as the default public
surface. Those still exist for debugging and historical bringup.
