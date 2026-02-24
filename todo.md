'~' - in progress blocked by subtask (indent subtask)
'-' - in progress
'x' - finished


[~] run rl run with glm 5
  [x] get static preflight working
  [x] fix GLM-4.7 config for 2xB200
  [~] fix model download bottleneck (GLM-4.7 timed out downloading 48 shards)
    [x] increase SGLang startup timeout (InferenceConfig.startup_timeout, default 300s)
    [x] pass HF_TOKEN to remote instances (writes to ~/.cache/huggingface/token)
    [ ] add RunPod network volume support to broker (persistent HF cache)
  [ ] once GLM-4.7 works, scale to 8-GPU
  [ ] then multi-node
  [ ] then GLM-5 on 8xB200

[~] reduce cold start iteration time (~10 min currently)
  current breakdown: CUDA download ~3.5 min, python deps ~1 min, model download ~5 min
  [ ] RunPod network volumes for HF cache (avoids re-downloading models)
  [ ] custom Docker image with CUDA 12.8 pre-installed (avoids 4GB CUDA download)
  [ ] Modal for faster iteration (sandbox caching, no SSH wait)

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
