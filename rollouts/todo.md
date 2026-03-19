'x' - done
'~' - in progress, subtasks blocking
'-' - in progress

# 2026-03-18
---

[ ] Before the next trace/frontend refactor phase, do an explicit feature-parity pass against `dreadnode/agent-lens`
    - Repo: https://github.com/dreadnode/agent-lens
    - Relevant surfaces to match or intentionally reject:
      - run list + filtering/search
      - run overview with session/fork relationships
      - trajectory viewer with thinking blocks, tool calls, and observations
      - API request/response capture inspection
      - file state / memory diffs across sessions
      - replay / resample workflow support and provenance
    - Decision needed before moving on: which of these belong in `rollouts/frontend/` now vs later

# 2026-02-26
---

[ ] Try smaller model for Modal (mini-glm-moe or similar from prime-rl)
[ ] Add network volume support to RunPod for persistent model cache
[ ] Consider publishing miniray to PyPI or using uv workspace sync on remote

# 2026-02-25
---

[x] Fix miniray import error on RunPod deployment
    - Root cause: bifrost deploys full workspace but Python sys.path didn't include workspace root
    - Fix: Added sys.path.insert(0, workspace_root) in run.py
    - Also added workspace to PYTHONPATH in env_vars (though shlex.quote prevents shell expansion)

[x] Enable preflight for Megatron backend
    - Subprocess GPU probe keeps CUDA out of main process, safe to run

[x] Fix memory estimation for Megatron distributed optimizer
    - Megatron defaults to use_distributed_optimizer=True (ZeRO-style sharding)
    - Updated preflight to account for this

[x] Add config validation to preflight
    - EP must divide num_experts evenly
    - GPU count must match TP * PP * EP for Megatron
    - Memory estimation fails hard instead of warning

[~] Test GLM Megatron training with EP=4 on RunPod 8x H100
    - Got past miniray import
    - Megatron workers spawned
    - Inference engines started loading model
    - Blocked: downloading 60GB model weights every run (no network volume)
    - Blocked: port 40999 already in use (stale process from crashed run)

[ ] clean up external driver integration
[ ] clean up cost counting integrations
[ ] themes (compact)

[-] support opencode zen
[-] support vercel ai gateway
