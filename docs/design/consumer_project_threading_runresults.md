# consumer-project-threading: run results

Branch: `consumer-project-threading` (6 commits ahead of `main`).

## What shipped

1. `resolve_consumer_project(config_path)` helper in
   `rollouts.remote_runtime`. Walks up from the config file, honors uv
   workspace membership so it lands on `~/research/` (workspace root) for
   monorepo configs and on the consumer's pyproject dir for external
   consumers (courier, charisma, etc).
2. Threaded `consumer_project_root` through the launch chain: argus CLI →
   `build_local_workload_plan` → eval/serving supervisors →
   `realize_worker_backed_endpoint` → `_realize_ssh_endpoint` /
   `_realize_modal_endpoint` → bifrost.
3. Three REPO_ROOT lies removed at the threaded sites:
   - `launch_plan.build_local_{eval,serving}` now compute results dir
     under `consumer_project_root / "results" / <kind> / <run>`.
   - supervisors pass `cwd=consumer_project_root` to the child
     subprocess.
   - `realize_worker_backed_endpoint` passes the consumer to bifrost as
     `extra_python_projects` (SSH) / `extra_source_roots` (Modal) when
     the consumer lives outside the primary workspace.
4. Argus CLI honors CWD-relative `--config` paths (was REPO_ROOT-relative
   only).
5. Remote ProcessSpec cwd: `workspace.root` instead of
   `workspace.root / "rollouts"` (the latter assumed the bundle always
   contained a `rollouts/` subdir, false for external consumers).
6. Modal session now receives `request.extra_source_roots` (was previously
   set on the request but dropped into `ModalExecutionSession(...)` with
   the default empty tuple, so nothing was materialized).

## Proof

Ran end-to-end with `--force-deploy-committed`, observing the `run_end`
`status=ok` event + populated `workloads/<name>/{engine_report,report,...}.json`.

**From ~/research (monorepo consumer, regression check):**
- `tau2_retail_c4_smoke.py` → 4 samples, ok. Results at
  `~/research/results/serving/run_20260421-071652/`.
- `kimi_verifier_deepseek_v32_mi355x_tiny.py` → 2 samples, 1 tool_calls +
  1 stop on MI355X SSH. Results at
  `~/research/results/serving/run_20260421-072012/`.

**From ~/silares_stuff/courier (external consumer, courier on the
`consumer-project-test` branch with vendored research_deps synced to
these changes; branch already torn down):**
- `kimi_verifier_deepseek_v32_mi355x_tiny.py` → 2 samples, 2 stop.
  Results at
  `~/silares_stuff/courier/results/serving/run_20260421-072609/`.
- `harbor_tb2_modal_c4_smoke.py` → 4 samples, mean_reward=0.5.
  Results at
  `~/silares_stuff/courier/results/serving/run_20260421-073033/`.
- `bench_deepseek_v3_2_amd_mi355x_sharegpt_tiny.py` → 2 samples, ok.
  Results at
  `~/silares_stuff/courier/results/eval/run_20260421-073718/`.
- `tau2_retail_c4_smoke.py` → **blocked** by pre-existing python 3.13
  `audioop` incompat in the courier venv's tau2 dep (ModuleNotFoundError
  at config import). Unrelated to our change; would need `audioop-lts`
  shim in courier's deps.

## To merge for real

1. PR the branch.
2. Bump `third_party/research_deps/manifest.toml` in courier to a revision
   that includes these commits, then `./scripts/sync_research_deps.sh`.
3. Courier's production invocations (`cd ~/silares_stuff/courier &&
   uv run python -m argus run --config courier/...`) will then work as
   described.

## Deferred (flagged for a later pass)

- `argus.run._external_config_projects` still exists — used only by the
  top-level training deploy path which we didn't thread through. Fine for
  now, duplicates the consumer-project concept.
- Bifrost `ModalExecutionRequest.extra_source_roots: tuple[str, ...]` vs
  `WorkspaceMaterializationSpec.extra_python_projects: tuple[PPM, ...]`
  divergence intact. Modal path is stringly-typed tarball upload; SSH path
  is structured editable install. Unifying requires Modal to grow the
  editable-install semantics; deferred.
- `REPO_ROOT = Path(__file__)...` constants unchanged in argus/rollouts —
  still used for package-relative asset lookup. The lie is no longer
  ambient; it's now a legitimate "where is this module installed" read.
