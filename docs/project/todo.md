# Project TODO

'~' - in progress blocked by subtask (indent subtask)
'-' - in progress
'x' - finished


[~] run rl run with glm 5
  [x] get static preflight working
  [x] fix GLM-4.7 config for 2xB200
  [~] fix model download bottleneck (GLM-4.7 timed out downloading 48 shards)
    [x] increase SGLang startup timeout (InferenceConfig.startup_timeout, default 300s)
    [x] pass HF_TOKEN to remote instances (writes to ~/.cache/huggingface/token)
    [~] add RunPod network volume support to broker (persistent HF cache)
      researched: broker uses direct GraphQL not SDK
      add: network_volume_id, network_volume_mount_path to ProvisionRequest
      modify: broker/providers/runpod.py:477-513 pod_input construction
      [ ] create network volume in RunPod dashboard (one-time)
      [ ] add fields to broker/types.py:ProvisionRequest
      [ ] thread through bifrost/GPUQuery
      [ ] add to RunPod GraphQL mutation payload
  [ ] once GLM-4.7 works, scale to 8-GPU
  [ ] then multi-node
  [ ] then GLM-5 on 8xB200

[~] reduce cold start iteration time (~10 min currently)
  current breakdown: CUDA download ~3.5 min, python deps ~1 min, model download ~5 min
  [ ] RunPod network volumes for HF cache (see above - same implementation)
  [ ] custom Docker image with CUDA 12.8 pre-installed (avoids 4GB CUDA download)
  [~] Modal for faster iteration (sandbox caching, no SSH wait)
    [x] fix modal sandbox exec hanging (was pending sandboxes blocking - added cleanup)
    [x] add observability: warning if GPU verify >10s, verbose=True
    [x] upgrade modal SDK to 1.3.4
    [x] fix snapshot API (_experimental_snapshot_directory, _experimental_mount_image)
    [x] implement directory snapshot caching for HF weights
        - snapshot created: im-01KJ9FCRBD7NCN2Y8Z8B29HBP8 (GLM-4.7-Flash, ~60GB)
        - cached to Modal Dict, persists 30 days
    [x] test snapshot mounting on subsequent runs - works! skips 13min download
    [x] fix preflight to recognize megatron sharding (was estimating 383GB, now 57GB)
    [x] fix megatron worker fork order (workers must spawn before CUDA init)
        - CUDA contexts don't survive fork() - children inherit broken state
        - moved spawn_megatron_workers() to top of _grpo_train_async, before SGLang
    [x] add uncommitted changes warning to modal_runner (matches RunPod behavior)
    [-] testing megatron backend with GLM-4.7-Flash
        - added mbridge dep for HF->Megatron weight conversion
        - workers now survive fork, running setup_megatron_model()

[~] make engine_v2.py faster
  [~] benchmark engine_v2 vs sglang
    [x] instrument modal runner w JSONL wide event logging (server_start, server_log, progress, benchmark_done)
    [x] switch to uv_pip_install (58s builds vs 5+ min, caching works)
    [x] run engine_v2 benchmark (0.7 req/s, TTFT p50=103ms, E2E p50=1156ms on A100)
    [-] run sglang benchmark (image build failing - need to debug)
    [ ] compare results, identify bottlenecks
  [ ] profile engine_v2 with torch profiler / nsight
  [ ] study kestrel engine design (/tmp/kestrel/)




--

[ ] compress code where possible
[ ] get precommit hooks passing
[ ] clean up + make evals code nice
[ ] run evals


[ ] themes (compact)
[ ] toggleable tool results (ideally scrollable)
[ ]

[ ] implement opentui instead of bespoke tui
[ ] get smoke test runs for RL on single gpu node
[ ] get multi-node working for RL
[ ] pretraining?
[ ] support skills
[ ] sometimes bash commands get stuck
