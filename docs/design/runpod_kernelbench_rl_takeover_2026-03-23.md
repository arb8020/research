RunPod KernelBench RL takeover note

Date: 2026-03-23

Goal
- Get Charisma-owned KernelBench RL configs running end-to-end on RunPod SSH.
- Keep Modal on `image_owned`.
- Move RunPod SSH to an honest `managed_venv` runtime instead of the old hybrid.

What is already fixed
- RunPod direct SSH bootstrap now works for custom images via `dockerArgs`.
- External Charisma project staging on SSH works.
- Remote editable path normalization works.
- SSH launch no longer tries to do fake CUDA reconciliation for explicit image-backed runtimes.
- RunPod SSH now uses a managed venv path:
  - `/root/.bifrost/venvs/rollouts-rl/bin/python`
- Managed-venv pip layers are refreshed each SSH run instead of trusting stale manifest state.
- Charisma RL now auto-fetches KernelBench-v3 if the pinned checkout is missing.

Relevant research commits
- `698ff307` `broker: use docker args for runpod ssh bootstrap`
- `65655f26` `argus: normalize remote manifest exec result`
- `0bb467f7` `argus: use managed venv for ssh runtimes`
- `c9aab3a1` `argus: trust explicit ssh images for cuda`
- `a0d0cfd4` `argus: use absolute ssh venv paths`
- `e6cd415b` `argus: always refresh managed ssh venv deps`
- `fac3e5ad` `rollouts: make mbridge explicit in megatron runtime`

Relevant charisma commits
- `2a8525e` `charisma: use managed venv deps for runpod rl`
- `7701ba2` `charisma: fetch kernelbench data for rl runs`
- `a4514e7` `charisma: include mbridge in megatron rl deps`

Current config under test
- `/Users/chiraagbalu/silares_stuff/charisma/charisma/configs/kernelbench_v3/rl_v3_smoke_cuda_qwen06_codeblock_runpod.py`

Current RunPod command
```bash
uv run python -m argus run \
  --config /Users/chiraagbalu/silares_stuff/charisma/charisma/configs/kernelbench_v3/rl_v3_smoke_cuda_qwen06_codeblock_runpod.py \
  --node-id runpod:nfevriq6k7twum \
  --tail \
  --no-hf-token \
  --force-deploy-committed
```

Last fully confirmed remote failure
- Run id: `run_20260323-161946`
- Remote path:
  - `/root/.bifrost/workspaces/rollouts-rl/rollouts/results/rl/run_20260323-161946/training.log`
- Failure:
  - `ImportError: mbridge is required for Megatron backend`

Important nuance
- That failure was observed before `fac3e5ad` and `a4514e7`.
- After those commits, one new rerun was started and then interrupted before we captured a new remote outcome.
- So `mbridge` is now explicit in both runtime profiles, but that fix is not yet verified on RunPod.

Why `mbridge` surfaced
- Modal was using the Slime image as an `image_owned` Python runtime.
- RunPod now uses a clean managed venv on top of the same boot image.
- Packages baked into the image interpreter are not automatically visible in that venv.
- Making `mbridge` explicit is the honest first step.

Open question after the next rerun
- If RunPod still fails on `mbridge`, inspect the image-installed copy before assuming the upstream pip ref is semantically identical.
- Concretely:
  - compare the image interpreter’s `mbridge.__file__` and version/commit
  - compare that to what the managed venv installs
- If Slime carries a patched `mbridge`, the next fix should model that as an image-coupled runtime input rather than silently substituting upstream.

Remote node state when this note was written
- RunPod node id: `runpod:nfevriq6k7twum`
- Public SSH observed earlier:
  - host `154.54.102.30`
  - port `15264`
- This node should be considered disposable; terminate it if not actively debugging.

Known unrelated local dirt
- The `research` worktree has unrelated user changes and untracked files.
- Do not revert them.
- Use `--force-deploy-committed` for continued remote runs unless those changes are intentionally committed.

Next action
1. Rerun the config above on a fresh or reused RunPod node.
2. If `mbridge` is fixed, record the next real failure boundary.
3. If `mbridge` still fails, inspect the image-installed copy before changing the package source again.
