# RunPod+Bifrost Training Recovery TODO

Goal: get the two broken training paths back to a truthful working state using RunPod+Bifrost while `modal_runner` is being rewritten.

Success means all of the following are true:

- `torchtitan x qed-vllm` boots on RunPod via Bifrost.
- `megatron x slime-sglang` boots on RunPod via Bifrost.
- each training loop takes at least a few GRPO steps
- inference readiness works from the actual realization entrypoint
- weight sync works on the active path, not just server startup

## Stage 1: Remote realization smokes

- Add a narrow RunPod+Bifrost smoke for `qed-vllm`.
- Add a narrow RunPod+Bifrost smoke for `slime-sglang`.
- Follow the existing remote pattern from `examples/eval/reverse_text/run_runpod.py`.
- Provision with `bifrost.acquire_node(...)`.
- Push the workspace with `bifrost.push(..., allow_dirty=True)`.
- Bootstrap only the deps needed to launch the realization.
- Launch the exact realization module with `bifrost.submit(ProcessSpec(...))`.

`qed-vllm` smoke must verify:

- `GET /health`
- `GET /weight_update_schema`
- one minimal OpenAI-compatible generation request if practical
- `POST /receive_weight_update` only if we wire a truthful NCCL setup for the smoke

`slime-sglang` smoke must verify:

- `GET /health`
- one basic OpenAI-compatible generation request

Keep both smokes narrow:

- launch
- wait for readiness
- send one request
- exit

Non-goals for these smokes:

- fixing lifecycle semantics
- fixing logging semantics
- proving full RL correctness

## Stage 2: RunPod witness configs

- Add or adapt a RunPod witness config for `torchtitan x qed-vllm`.
- Keep the existing realization split explicit:
  - `inference.backend="vllm"`
  - `inference.realization="qed-vllm"`
  - `checkpoint.inference_sync_realization="vllm_custom_nccl_broadcast"`
- Ensure the config is Bifrost-shaped rather than Modal-shaped.
- Reuse the smallest model that still exercises the real path.

- Revalidate the existing RunPod witness for `megatron x slime-sglang`.
- Keep the realization explicit:
  - `inference.backend="sglang"`
  - `inference.realization="slime-sglang"`

## Stage 3: Training-loop bring-up

- Run `megatron x slime-sglang` on RunPod.
- Confirm the training loop boots, launches inference, and completes a few GRPO steps.
- Confirm the inference server answers requests during training.
- Confirm the active weight-sync path succeeds at least once.

- Run `torchtitan x qed-vllm` on RunPod.
- Confirm the training loop boots, launches inference, and completes a few GRPO steps.
- Confirm the patched vLLM server exposes the expected routes during training.
- Confirm one real NCCL-backed weight update succeeds.

## Stage 4: Debugging cut if failures remain

- If readiness fails, debug the realization smoke before touching GRPO.
- If training boots but sync fails, debug the sync realization contract before changing loop logic.
- If the server launches only under one environment shape, fix deps/runtime layout in config rather than adding fallback logic.
- Do not reintroduce `modal_runner` assumptions into the RunPod path.

## Expected deliverables

- one `qed-vllm` RunPod smoke
- one `slime-sglang` RunPod smoke
- one RunPod witness config for `torchtitan x qed-vllm`
- verified RunPod witness for `megatron x slime-sglang`
- short notes on any remaining failure boundary if either loop still breaks
