# Inference Engine Assignment

This assignment asks you to replace the stub inference logic in
[skeleton_server.py](/Users/chiraagbalu/research/rollouts/rollouts/inference/skeleton_server.py)
with a real inference engine implementation.

The serving contract is already implemented for you. You should usually only
edit `generate_reply()`.

## What You Implement

Edit:

- [skeleton_server.py](/Users/chiraagbalu/research/rollouts/rollouts/inference/skeleton_server.py)

Replace the default deterministic stub in `generate_reply()` with real model
execution.

The untouched starter is intentionally:

- protocol-complete
- deterministic
- wrong on the task

That means the eval harness can already talk to it correctly before you add any
real inference logic.

## HTTP Contract

Your server must satisfy:

- `GET /health` returns `200` with `{"status": "ok"}`
- `POST /v1/chat/completions` accepts OpenAI-compatible chat requests
- when `stream=True`, it must return SSE chunks
- responses must include `logprobs`
- each generated token should include `top_logprobs`

The eval harness uses `stream=True` by default.

The starter already handles:

- FastAPI routing
- SSE framing
- OpenAI-compatible response shape
- logprob/top-k shaping

Do not rewrite that surface unless you have a specific reason.

## Function Contract

`generate_reply()` receives:

- `messages`: conversation so far
- `model`: model name/path from the launch command
- `max_tokens`: max generation length
- `temperature`: decoding temperature
- `**kwargs`: extra request fields like `top_p`

It must return:

- `GenerationResult(reply_text, finish_reason, token_logprobs)`

Where:

- `finish_reason` is `"stop"` or `"length"`
- `token_logprobs` describes the selected token logprob and top-k alternatives

## How To Run

From `/Users/chiraagbalu/research/rollouts`:

Run directly through the eval runner:

```bash
../.venv/bin/python -m rollouts.eval.run --config examples/inference/eval_skeleton_server.py
```

Or run through Argus:

```bash
../.venv/bin/python -m argus run --config /Users/chiraagbalu/research/rollouts/examples/inference/eval_skeleton_server.py
```

Argus is the control-plane wrapper. For this assignment, the important point is
that both commands run the same eval config. If you just want the shortest path,
use `rollouts.eval.run`.

## What The Harness Does

The eval config at
[eval_skeleton_server.py](/Users/chiraagbalu/research/rollouts/examples/inference/eval_skeleton_server.py)
will:

1. launch your server on Modal
2. wait for `/health`
3. send a small reverse-text workload
4. score the outputs
5. write artifacts to `results/...`

The default workload is intentionally small. It is a smoke witness, not the
full benchmark.

## What Good Looks Like

There are two separate bars:

Mechanical correctness:

- server starts
- health check passes
- SSE works
- eval completes without provider errors
- logprobs/top-k fields are present

Task correctness:

- reverse-text outputs are actually correct

The untouched starter should pass the first bar and fail the second.

## Where To Look After A Run

The eval writes:

- `report.json`: summary metrics
- `samples/*.json`: per-sample artifacts

The per-sample artifact now includes a semantic trace in:

- `metadata.semantic_trace`

That trace is the raw material for correctness/fidelity checks. It may include:

- output tokens
- selected-token logprobs
- top-k candidates
- finish reason
- token IDs when the backend provides them

## Reference Baselines

Reference implementations and smoke witnesses:

- [gold_server.py](/Users/chiraagbalu/research/rollouts/rollouts/inference/gold_server.py)
- [eval_gold_server_smoke.py](/Users/chiraagbalu/research/rollouts/examples/inference/eval_gold_server_smoke.py)
- [eval_slime_sglang_smoke.py](/Users/chiraagbalu/research/rollouts/examples/inference/eval_slime_sglang_smoke.py)
- [eval_qed_vllm_smoke.py](/Users/chiraagbalu/research/rollouts/examples/inference/eval_qed_vllm_smoke.py)

Useful minimum backend set today:

- HF gold baseline
- SGLang
- vLLM

Not recommended as a primary target yet:

- mini-sglang
- TRT-LLM

## Recommended Workflow

1. Make the untouched starter pass mechanically.
2. Replace the stub with real inference.
3. Re-run the smoke eval.
4. Inspect `report.json` and `samples/*.json`.
5. Optimize latency/throughput without breaking logprob behavior.

## Current Grading Direction

Short-term, we care about:

- serving contract correctness
- task correctness on the smoke workload
- useful runtime metrics
- logprob/top-k behavior

Longer-term, the semantic trace artifacts will support stricter deployment
verification against a reference backend.
